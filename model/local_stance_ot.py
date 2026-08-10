"""Local support/deny conflict patterns via optimal transport.

The branch treats every labelled parent-child reply as a local discourse event.
Hard stance labels split the events in an anchor's descendant neighbourhood into
support and deny empirical measures; they are *not* recursively composed into a
root-relative stance.  Entropic optimal transport then matches the two local
measures and encodes their semantic discrepancy, balance, and transport cost.

Both node-level anchor representations and a graph-level vector are returned so
the branch can be used alone or fused with an existing graph-level branch.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax


class LocalStanceOptimalTransportBranch(nn.Module):
    """Encode local support/deny event distributions with Sinkhorn OT.

    The existing dataset convention is used throughout: ``0`` denotes support
    and ``1`` denotes deny.  For anchor ``i``, an event belongs to its local
    field when the event's child is a descendant of ``i`` within
    ``ot_local_hops``.  Thus a label always retains its direct-parent meaning;
    no XOR/parity conversion is performed.
    """

    def __init__(self, hidden_dim, max_hop, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_hop = max(1, int(max_hop))
        self.local_hops = min(
            self.max_hop,
            max(1, int(getattr(args, "ot_local_hops", 3))),
        )
        self.entropic_regularization = max(
            1e-4,
            float(getattr(args, "ot_entropic_regularization", 0.1)),
        )
        self.sinkhorn_iterations = max(
            1,
            int(getattr(args, "ot_sinkhorn_iterations", 20)),
        )
        self.structure_cost_weight = max(
            0.0,
            float(getattr(args, "ot_structure_cost_weight", 0.25)),
        )
        self.max_events_per_side = max(
            0,
            int(getattr(args, "ot_max_events_per_side", 16)),
        )
        self.detach_transport_plan = bool(
            getattr(args, "ot_detach_transport_plan", False)
        )
        self.dropout = max(
            0.0,
            float(getattr(args, "ot_dropout", getattr(args, "dropout", 0.1))),
        )
        self.eps = 1e-6

        # A stance event is represented by its two endpoint semantics.  The
        # stance label only assigns the event to a measure; it is deliberately
        # excluded from the ground-cost embedding to avoid a trivial label cost.
        self.event_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.ground_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        # support ratio, deny ratio, balance, OT cost, plan entropy,
        # neighbourhood size, mean relative depth, and sampling coverage.
        self.stat_projection = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 7, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.node_norm = nn.LayerNorm(self.hidden_dim)
        self.node_residual_gate = nn.Parameter(torch.tensor(0.0))

        self.graph_attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.graph_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.last_nodes = None
        self.last_graph = None
        self.last_transport_cost = None
        self.last_transport_entropy = None
        self.last_support_mass = None
        self.last_deny_mass = None
        self.last_selected_support_mass = None
        self.last_selected_deny_mass = None
        self.last_coverage = None
        self.last_balance = None
        self.last_conflict_strength = None
        self.last_active_anchor = None
        self.last_attention = None
        self.last_transport_anchors = None
        self.last_transport_plans = None

    @staticmethod
    def _batch_size(batch):
        return 0 if batch.numel() == 0 else int(batch.max().item()) + 1

    def _sinkhorn(self, cost):
        """Return a balanced entropic transport plan for one cost matrix."""
        num_support, num_deny = cost.shape
        if num_support == 0 or num_deny == 0:
            return cost.new_zeros(cost.shape)
        support_mask = torch.ones(
            (1, num_support), dtype=torch.bool, device=cost.device
        )
        deny_mask = torch.ones(
            (1, num_deny), dtype=torch.bool, device=cost.device
        )
        return self._sinkhorn_batched(
            cost.unsqueeze(0), support_mask, deny_mask
        ).squeeze(0)

    def _sinkhorn_batched(self, cost, support_mask, deny_mask):
        """Vectorized log-domain Sinkhorn for padded local measures."""
        support_count = support_mask.sum(dim=1).clamp_min(1).to(cost.dtype)
        deny_count = deny_mask.sum(dim=1).clamp_min(1).to(cost.dtype)
        log_a = -support_count.log().unsqueeze(1).expand_as(support_mask)
        log_b = -deny_count.log().unsqueeze(1).expand_as(deny_mask)
        negative_large = torch.finfo(cost.dtype).min / 4.0
        log_a = log_a.masked_fill(~support_mask, negative_large)
        log_b = log_b.masked_fill(~deny_mask, negative_large)
        pair_mask = support_mask.unsqueeze(2) & deny_mask.unsqueeze(1)
        log_kernel = (-cost / self.entropic_regularization).masked_fill(
            ~pair_mask, negative_large
        )
        log_u = torch.zeros_like(log_a).masked_fill(
            ~support_mask, negative_large
        )
        log_v = torch.zeros_like(log_b).masked_fill(
            ~deny_mask, negative_large
        )
        for _ in range(self.sinkhorn_iterations):
            row_normalizer = torch.logsumexp(
                log_kernel + log_v.unsqueeze(1), dim=2
            )
            row_normalizer = row_normalizer.masked_fill(
                ~support_mask, 0.0
            )
            log_u = (log_a - row_normalizer).masked_fill(
                ~support_mask, negative_large
            )
            column_normalizer = torch.logsumexp(
                log_kernel + log_u.unsqueeze(2), dim=1
            )
            column_normalizer = column_normalizer.masked_fill(
                ~deny_mask, 0.0
            )
            log_v = (log_b - column_normalizer).masked_fill(
                ~deny_mask, negative_large
            )
        plan = torch.exp(
            log_kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
        )
        plan = plan * pair_mask.to(dtype=plan.dtype)
        return plan / plan.sum(dim=(1, 2), keepdim=True).clamp_min(self.eps)

    def _valid_events(self, node_hidden, edge_index, edge_stance, depth, batch):
        device = node_hidden.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        if edge_index.numel() == 0:
            return empty, empty, empty, node_hidden.new_zeros(
                (0, self.hidden_dim)
            )
        if edge_stance is None:
            raise ValueError(
                "classification_fusion_mode includes 'ot', but the batch has "
                "no edge_stance hard labels"
            )

        labels = edge_stance.view(-1).long().to(device=device)
        if labels.numel() != edge_index.size(1):
            raise ValueError(
                "edge_stance must contain one hard label per edge for the OT "
                "branch; got {} labels for {} edges".format(
                    labels.numel(), edge_index.size(1)
                )
            )
        src, dst = edge_index.long()
        valid = (
            (depth[src] >= 0)
            & (depth[dst] == depth[src] + 1)
            & (batch[src] == batch[dst])
            & ((labels == 0) | (labels == 1))
        )
        edge_ids = valid.nonzero(as_tuple=False).view(-1)
        if edge_ids.numel() == 0:
            return empty, empty, empty, node_hidden.new_zeros(
                (0, self.hidden_dim)
            )

        parent = src[edge_ids]
        child = dst[edge_ids]
        event_hidden = self.event_encoder(
            torch.cat(
                (
                    node_hidden[parent],
                    node_hidden[child],
                    (node_hidden[parent] - node_hidden[child]).abs(),
                    node_hidden[parent] * node_hidden[child],
                ),
                dim=-1,
            )
        )
        return parent, child, labels[edge_ids], event_hidden

    def _local_memberships(self, parent, child, num_nodes):
        """Map descendant stance events to all local ancestor anchors."""
        device = parent.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        if child.numel() == 0:
            return empty, empty, empty

        parent_of_node = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        parent_of_node[child] = parent
        event_id = torch.arange(child.numel(), device=device)
        current_anchor = parent.clone()
        anchors = []
        events = []
        distances = []
        for distance in range(1, self.local_hops + 1):
            valid = current_anchor >= 0
            if not valid.any():
                break
            anchors.append(current_anchor[valid])
            events.append(event_id[valid])
            distances.append(
                torch.full_like(event_id[valid], distance)
            )
            next_anchor = torch.full_like(current_anchor, -1)
            next_anchor[valid] = parent_of_node[current_anchor[valid]]
            current_anchor = next_anchor
        if not anchors:
            return empty, empty, empty
        return (
            torch.cat(anchors),
            torch.cat(events),
            torch.cat(distances),
        )

    def _dense_side(
        self,
        side,
        anchor_index,
        event_index,
        relative_depth,
        labels,
        event_child,
        num_nodes,
    ):
        """Pack one stance side into anchor-major padded event indices."""
        side_membership = labels[event_index] == side
        side_anchor = anchor_index[side_membership]
        side_event = event_index[side_membership]
        side_depth = relative_depth[side_membership]
        raw_count = torch.bincount(side_anchor, minlength=num_nodes)
        if side_anchor.numel() == 0:
            return (
                side_event.new_full((num_nodes, 1), -1),
                side_depth.new_zeros((num_nodes, 1)),
                torch.zeros(
                    (num_nodes, 1), dtype=torch.bool, device=side_event.device
                ),
                raw_count,
            )

        # Anchor-first, then nearest-depth-first ordering gives each anchor a
        # deterministic local truncation without a Python/Sinkhorn loop.
        # Child id is a stable final tie-breaker.  Without it, truncation among
        # same-depth events would depend on the arbitrary edge column order.
        side_child = event_child[side_event]
        sort_key = (
            (side_anchor * (self.local_hops + 1) + side_depth)
            * (num_nodes + 1)
            + side_child
        )
        order = torch.argsort(sort_key)
        side_anchor = side_anchor[order]
        side_event = side_event[order]
        side_depth = side_depth[order]
        group_start = raw_count.cumsum(dim=0) - raw_count
        position = torch.arange(
            side_anchor.numel(), device=side_anchor.device
        ) - group_start[side_anchor]
        if self.max_events_per_side > 0:
            width = self.max_events_per_side
            keep = position < width
            side_anchor = side_anchor[keep]
            side_event = side_event[keep]
            side_depth = side_depth[keep]
            position = position[keep]
        else:
            width = max(1, int(raw_count.max().item()))

        dense_event = side_event.new_full((num_nodes, width), -1)
        dense_depth = side_depth.new_zeros((num_nodes, width))
        dense_mask = torch.zeros(
            (num_nodes, width),
            dtype=torch.bool,
            device=side_event.device,
        )
        dense_event[side_anchor, position] = side_event
        dense_depth[side_anchor, position] = side_depth
        dense_mask[side_anchor, position] = True
        return dense_event, dense_depth, dense_mask, raw_count

    def forward(
        self,
        node_hidden,
        edge_index,
        edge_stance,
        depth,
        batch,
        roots,
    ):
        num_nodes = node_hidden.size(0)
        batch_size = self._batch_size(batch)
        parent, child, labels, event_hidden = self._valid_events(
            node_hidden,
            edge_index,
            edge_stance,
            depth,
            batch,
        )
        anchor_index, event_index, relative_depth = self._local_memberships(
            parent,
            child,
            num_nodes,
        )

        (
            support_event,
            support_depth,
            support_mask,
            support_count,
        ) = self._dense_side(
            0,
            anchor_index,
            event_index,
            relative_depth,
            labels,
            child,
            num_nodes,
        )
        (
            deny_event,
            deny_depth,
            deny_mask,
            deny_count,
        ) = self._dense_side(
            1,
            anchor_index,
            event_index,
            relative_depth,
            labels,
            child,
            num_nodes,
        )
        support_mass = support_count.to(dtype=node_hidden.dtype)
        deny_mass = deny_count.to(dtype=node_hidden.dtype)
        selected_support_mass = support_mask.sum(dim=1).to(node_hidden.dtype)
        selected_deny_mass = deny_mask.sum(dim=1).to(node_hidden.dtype)
        total_mass = support_mass + deny_mass
        selected_total_mass = selected_support_mass + selected_deny_mass
        coverage = selected_total_mass / total_mass.clamp_min(1.0)
        active_anchor = total_mass > 0
        transport_anchor_mask = (support_mass > 0) & (deny_mass > 0)
        transport_anchors = transport_anchor_mask.nonzero(
            as_tuple=False
        ).view(-1)

        if event_hidden.size(0) == 0:
            support_dense = node_hidden.new_zeros(
                (num_nodes, support_event.size(1), self.hidden_dim)
            )
            deny_dense = node_hidden.new_zeros(
                (num_nodes, deny_event.size(1), self.hidden_dim)
            )
        else:
            support_dense = event_hidden[support_event.clamp_min(0)]
            deny_dense = event_hidden[deny_event.clamp_min(0)]
            support_dense = support_dense * support_mask.unsqueeze(-1)
            deny_dense = deny_dense * deny_mask.unsqueeze(-1)

        support_selected_count = support_mask.sum(dim=1).clamp_min(1)
        deny_selected_count = deny_mask.sum(dim=1).clamp_min(1)
        support_mean = support_dense.sum(dim=1) / (
            support_selected_count.unsqueeze(-1).to(node_hidden.dtype)
        )
        deny_mean = deny_dense.sum(dim=1) / (
            deny_selected_count.unsqueeze(-1).to(node_hidden.dtype)
        )
        selected_count = support_mask.sum(dim=1) + deny_mask.sum(dim=1)
        mean_relative_depth = (
            (support_depth * support_mask).sum(dim=1)
            + (deny_depth * deny_mask).sum(dim=1)
        ).to(node_hidden.dtype) / selected_count.clamp_min(1).to(
            node_hidden.dtype
        )

        balance = (
            4.0
            * support_mass
            * deny_mass
            / total_mass.pow(2).clamp_min(1.0)
        )
        transport_difference = node_hidden.new_zeros(
            (num_nodes, self.hidden_dim)
        )
        transport_cost = node_hidden.new_zeros(num_nodes)
        transport_entropy = node_hidden.new_zeros(num_nodes)
        if transport_anchors.numel() > 0:
            active_support = support_dense[transport_anchors]
            active_deny = deny_dense[transport_anchors]
            active_support_mask = support_mask[transport_anchors]
            active_deny_mask = deny_mask[transport_anchors]
            support_ground = F.normalize(
                self.ground_projection(active_support),
                dim=-1,
                eps=self.eps,
            )
            deny_ground = F.normalize(
                self.ground_projection(active_deny),
                dim=-1,
                eps=self.eps,
            )
            semantic_cost = 1.0 - torch.bmm(
                support_ground,
                deny_ground.transpose(1, 2),
            )
            structural_cost = (
                support_depth[transport_anchors].unsqueeze(2)
                - deny_depth[transport_anchors].unsqueeze(1)
            ).abs().to(node_hidden.dtype) / float(self.local_hops)
            ground_cost = semantic_cost + (
                self.structure_cost_weight * structural_cost
            )
            plan = self._sinkhorn_batched(
                ground_cost,
                active_support_mask,
                active_deny_mask,
            )
            representation_plan = (
                plan.detach() if self.detach_transport_plan else plan
            )

            row_mass = representation_plan.sum(dim=2, keepdim=True)
            column_mass = representation_plan.sum(dim=1, keepdim=True)
            matched_deny = torch.bmm(representation_plan, active_deny)
            matched_deny = matched_deny / row_mass.clamp_min(self.eps)
            matched_support = torch.bmm(
                representation_plan.transpose(1, 2), active_support
            )
            matched_support = matched_support / column_mass.transpose(
                1, 2
            ).clamp_min(self.eps)
            support_displacement = (
                (active_support - matched_deny).abs() * row_mass
            ).sum(dim=1)
            deny_displacement = (
                (active_deny - matched_support).abs()
                * column_mass.transpose(1, 2)
            ).sum(dim=1)
            active_difference = 0.5 * (
                support_displacement + deny_displacement
            )
            transport_difference = transport_difference.index_copy(
                0, transport_anchors, active_difference
            )

            active_cost = (representation_plan * ground_cost).sum(dim=(1, 2))
            transport_cost = transport_cost.index_copy(
                0, transport_anchors, active_cost
            )
            active_entropy = -(
                representation_plan
                * representation_plan.clamp_min(self.eps).log()
            ).sum(dim=(1, 2))
            support_cardinality = active_support_mask.sum(dim=1)
            deny_cardinality = active_deny_mask.sum(dim=1)
            max_entropy = (
                support_cardinality * deny_cardinality
            ).clamp_min(1).to(node_hidden.dtype).log()
            active_entropy = torch.where(
                max_entropy > 0,
                active_entropy / max_entropy.clamp_min(self.eps),
                torch.zeros_like(active_entropy),
            )
            transport_entropy = transport_entropy.index_copy(
                0, transport_anchors, active_entropy
            )
            detached_plans = plan.detach()
        else:
            detached_plans = node_hidden.new_zeros(
                (0, support_event.size(1), deny_event.size(1))
            )

        support_ratio = support_mass / total_mass.clamp_min(1.0)
        deny_ratio = deny_mass / total_mass.clamp_min(1.0)
        normalized_count = torch.tanh(total_mass / 10.0)
        normalized_depth = mean_relative_depth / float(self.local_hops)
        conflict_strength = balance * (
            transport_cost / (2.0 + self.structure_cost_weight)
        ).clamp(0.0, 1.0)
        stats = torch.stack(
            (
                support_ratio,
                deny_ratio,
                balance,
                transport_cost,
                transport_entropy,
                normalized_count,
                normalized_depth,
                coverage,
            ),
            dim=-1,
        )
        stat_hidden = self.stat_projection(stats)
        local_update = self.node_fusion(
            torch.cat(
                (
                    node_hidden,
                    support_mean,
                    deny_mean,
                    (support_mean - deny_mean).abs(),
                    support_mean * deny_mean,
                    transport_difference,
                    stat_hidden,
                ),
                dim=-1,
            )
        )
        residual_gate = torch.sigmoid(self.node_residual_gate)
        updated_nodes = self.node_norm(
            node_hidden + residual_gate * local_update
        )
        ot_nodes = torch.where(
            active_anchor.unsqueeze(-1),
            updated_nodes,
            node_hidden,
        )

        root_hidden = node_hidden[roots]
        root_for_node = root_hidden[batch.long()]
        attention_score = self.graph_attention(
            torch.cat(
                (
                    ot_nodes,
                    root_for_node,
                    (ot_nodes - root_for_node).abs(),
                    ot_nodes * root_for_node,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        # Pool only anchors that own a non-empty local event field.  A root
        # fallback keeps single-node / unlabelled graphs well-defined without
        # allowing unrelated leaf semantics to dominate the OT branch.
        attention_mask = active_anchor.clone()
        graph_has_anchor = global_add_pool(
            attention_mask.to(node_hidden.dtype).unsqueeze(-1),
            batch,
            size=batch_size,
        ).squeeze(-1) > 0
        if (~graph_has_anchor).any():
            attention_mask[roots[~graph_has_anchor]] = True
        attention_score = attention_score.masked_fill(
            ~attention_mask,
            torch.finfo(attention_score.dtype).min / 4.0,
        )
        attention_weight = softmax(attention_score, batch)
        attention_weight = attention_weight * attention_mask.to(
            attention_weight.dtype
        )
        pooled = global_add_pool(
            ot_nodes * attention_weight.unsqueeze(-1),
            batch,
            size=batch_size,
        )
        ot_graph = self.graph_fusion(
            torch.cat(
                (
                    root_hidden,
                    pooled,
                    (root_hidden - pooled).abs(),
                    root_hidden * pooled,
                ),
                dim=-1,
            )
        )

        zero = node_hidden.new_zeros(())
        self.last_nodes = ot_nodes
        self.last_graph = ot_graph
        self.last_transport_cost = transport_cost
        self.last_transport_entropy = transport_entropy
        self.last_support_mass = support_mass
        self.last_deny_mass = deny_mass
        self.last_selected_support_mass = selected_support_mass
        self.last_selected_deny_mass = selected_deny_mass
        self.last_coverage = coverage
        self.last_balance = balance
        self.last_conflict_strength = conflict_strength
        self.last_active_anchor = active_anchor
        self.last_attention = attention_weight
        self.last_transport_anchors = transport_anchors
        self.last_transport_plans = detached_plans
        return ot_graph, ot_nodes, {
            "aux_loss": zero,
            "nodes": ot_nodes,
            "graph": ot_graph,
            "transport_cost": transport_cost,
            "transport_entropy": transport_entropy,
            "support_mass": support_mass,
            "deny_mass": deny_mass,
            "selected_support_mass": selected_support_mass,
            "selected_deny_mass": selected_deny_mass,
            "coverage": coverage,
            "balance": balance,
            "conflict_strength": conflict_strength,
            "active_anchor": active_anchor,
            "attention": attention_weight,
            "transport_anchors": self.last_transport_anchors,
            "transport_plans": detached_plans,
        }

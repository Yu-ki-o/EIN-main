import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateAuxSameDiffEnhancer(nn.Module):
    """
    LLM-state supervised same/different node enhancement.

    The LLM-derived node_state is used only as auxiliary supervision. Routing
    still comes from the current node representation, so the graph encoder can
    learn relation weights instead of hard-copying preprocessing labels.
    """

    def __init__(self, hidden_dim, args=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_states = int(getattr(args, 'state_aux_num_states', 2))
        self.dropout = float(getattr(args, 'dropout', 0.0))

        self.temp = max(1e-3, float(getattr(args, 'state_aux_temp', 1.0)))
        self.detach_route = bool(getattr(args, 'state_aux_detach_route', False))
        self.same_path_blend = max(0.0, float(getattr(args, 'same_path_blend', 1.0)))
        self.diff_path_blend = max(0.0, float(getattr(args, 'diff_path_blend', 1.0)))
        self.gate_blend = min(
            max(float(getattr(args, 'same_diff_gate_blend', 1.0)), 0.0),
            1.0,
        )

        self.lambda_node_state = max(
            0.0,
            float(getattr(args, 'lambda_node_state_aux', 0.1)),
        )
        self.lambda_edge_same = max(
            0.0,
            float(getattr(args, 'lambda_edge_same_aux', 0.1)),
        )
        self.lambda_sep = max(
            0.0,
            float(getattr(args, 'lambda_same_diff_sep', 0.01)),
        )

        self.logvar_min = float(getattr(args, 'same_dulreg_logvar_min', -8.0))
        self.logvar_max = float(getattr(args, 'same_dulreg_logvar_max', 8.0))
        if self.logvar_min > self.logvar_max:
            self.logvar_min, self.logvar_max = self.logvar_max, self.logvar_min
        self.precision_max = max(
            1.0,
            float(getattr(args, 'same_dulreg_precision_max', 1e4)),
        )
        self.dulreg_blend = min(
            max(float(getattr(args, 'same_dulreg_blend', 1.0)), 0.0),
            1.0,
        )
        self.transition_attention = bool(
            getattr(args, 'state_aux_transition_attention', False)
        )
        self.transition_attention_blend = max(
            0.0,
            float(getattr(args, 'state_aux_transition_attention_blend', 1.0)),
        )
        self.reply_target_attention = bool(
            getattr(args, 'state_aux_reply_target_attention', False)
        )
        self.reply_target_attention_blend = max(
            0.0,
            float(getattr(args, 'state_aux_reply_target_attention_blend', 1.0)),
        )
        self.cross_view_attention = bool(
            getattr(args, 'state_aux_cross_view_attention', False)
        )
        self.cross_view_attention_blend = max(
            0.0,
            float(getattr(args, 'state_aux_cross_view_attention_blend', 1.0)),
        )
        self.cross_view_topk_ratio = min(
            max(float(getattr(args, 'state_aux_cross_view_topk_ratio', 0.3)), 0.0),
            1.0,
        )
        self.cross_view_min_nodes = max(
            1,
            int(getattr(args, 'state_aux_cross_view_min_nodes', 1)),
        )
        self.edge_relation_routing = bool(
            getattr(args, 'state_aux_edge_relation_routing', False)
        )
        self.use_node_state_aux = bool(
            getattr(args, 'state_aux_use_node_state_aux', not self.edge_relation_routing)
        )

        self.state_head = nn.Linear(hidden_dim, self.num_states)
        if self.edge_relation_routing:
            self.edge_relation_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.same_uncertainty_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.diff_msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.same_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.diff_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.same_path_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.diff_path_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.same_summary_norm = nn.LayerNorm(hidden_dim)
        self.diff_summary_norm = nn.LayerNorm(hidden_dim)
        self.same_path_norm = nn.LayerNorm(hidden_dim)
        self.diff_path_norm = nn.LayerNorm(hidden_dim)
        if self.transition_attention or self.reply_target_attention:
            self.transition_query = nn.Linear(hidden_dim, hidden_dim)
            self.transition_key = nn.Linear(hidden_dim, hidden_dim)
            self.transition_value = nn.Linear(hidden_dim, hidden_dim)
            self.transition_out = nn.Linear(hidden_dim, hidden_dim)
            self.transition_norm = nn.LayerNorm(hidden_dim)
        if self.cross_view_attention:
            self.cross_same_score = nn.Linear(hidden_dim, 1)
            self.cross_diff_score = nn.Linear(hidden_dim, 1)
            self.cross_query = nn.Linear(hidden_dim, hidden_dim)
            self.cross_key = nn.Linear(hidden_dim, hidden_dim)
            self.cross_value = nn.Linear(hidden_dim, hidden_dim)
            self.cross_out = nn.Linear(hidden_dim, hidden_dim)
            self.cross_same_norm = nn.LayerNorm(hidden_dim)
            self.cross_diff_norm = nn.LayerNorm(hidden_dim)

        self._last_aux_loss = None
        self._last_node_state_loss = None
        self._last_edge_same_loss = None
        self._last_sep_loss = None
        self._last_same_weight_mean = None
        self._last_logvar_mean = None
        self._last_precision_mean = None

    def auxiliary_loss(self):
        if self._last_aux_loss is None:
            return None
        return self._last_aux_loss

    def _zero(self, h):
        return h.new_zeros(())

    def _valid_node_mask(self, node_state, num_nodes):
        if node_state is None:
            return None
        node_state = node_state.view(-1).long()
        if node_state.size(0) != num_nodes:
            return None
        return (node_state >= 0) & (node_state < self.num_states)

    def _aggregate_diff(self, h, edge_index, diff_weight):
        src, dst = edge_index
        msg = self.diff_msg_mlp(h[src])
        weight = diff_weight.view(-1, 1)
        weighted = weight * msg

        out = h.new_zeros(h.size())
        denom = h.new_zeros(h.size(0), 1)
        out.index_add_(0, dst, weighted)
        denom.index_add_(0, dst, weight)

        aggr = out / denom.clamp_min(1e-6)
        has_msg = denom > 0
        return torch.where(has_msg, aggr, h)

    def _aggregate_same_uncertain(self, h, edge_index, same_weight):
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        edge_feat = torch.cat([h_src, h_dst, h_src - h_dst, h_src * h_dst], dim=-1)
        edge_stats = self.same_uncertainty_mlp(edge_feat)
        msg_mu, raw_logvar = edge_stats.chunk(2, dim=-1)
        logvar = raw_logvar.clamp(self.logvar_min, self.logvar_max)
        precision = torch.exp(-logvar).clamp(max=self.precision_max)

        weight = same_weight.view(-1, 1) * precision
        weighted_mu = h.new_zeros(h.size())
        precision_sum = h.new_zeros(h.size())
        weighted_mu.index_add_(0, dst, weight * msg_mu)
        precision_sum.index_add_(0, dst, weight)

        dulreg = weighted_mu / precision_sum.clamp_min(1e-6)
        valid_dim = precision_sum > 0
        dulreg = torch.where(valid_dim, dulreg, h)

        if self.dulreg_blend < 1.0:
            basic = h.new_zeros(h.size())
            basic_denom = h.new_zeros(h.size(0), 1)
            scalar_weight = same_weight.view(-1, 1)
            basic.index_add_(0, dst, scalar_weight * h_src)
            basic_denom.index_add_(0, dst, scalar_weight)
            basic = basic / basic_denom.clamp_min(1e-6)
            basic = torch.where(basic_denom > 0, basic, h)
            dulreg = (1.0 - self.dulreg_blend) * basic + self.dulreg_blend * dulreg

        return dulreg, logvar.mean().detach(), precision.mean().detach()

    def _edge_same_probability(self, h, edge_index):
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        edge_feat = torch.cat(
            [h_src, h_dst, h_dst - h_src, h_src * h_dst],
            dim=-1,
        )
        return torch.sigmoid(self.edge_relation_head(edge_feat).view(-1))

    def _directed_tree_edges(self, edge_index):
        src, dst = edge_index
        valid = src != dst
        src = src[valid]
        dst = dst[valid]
        if src.numel() == 0:
            return src, dst

        # Batched propagation trees use monotonically increasing node ids in this
        # repo. When an undirected edge_index is supplied, this keeps parent->child
        # edges and drops the synthetic reverse edges.
        forward = src < dst
        return src[forward], dst[forward]

    def _directed_tree_edges_with_attr(self, edge_index, edge_attr):
        src, dst = edge_index
        valid = src != dst
        src = src[valid]
        dst = dst[valid]
        edge_attr = edge_attr[valid]
        if src.numel() == 0:
            return src, dst, edge_attr

        forward = src < dst
        return src[forward], dst[forward], edge_attr[forward]

    def _apply_transition_attention(
        self,
        z_same,
        z_diff,
        tree_edge_index,
        route_probs=None,
        tree_same_weight=None,
    ):
        if tree_same_weight is None:
            src, dst = self._directed_tree_edges(tree_edge_index)
        else:
            src, dst, edge_same_weight = self._directed_tree_edges_with_attr(
                tree_edge_index,
                tree_same_weight,
            )
        if src.numel() == 0:
            return z_diff

        num_nodes = z_diff.size(0)
        parent_of = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=z_diff.device,
        )
        parent_of[dst.long()] = src.long()
        same_to_parent = None
        if tree_same_weight is not None:
            same_to_parent = z_diff.new_full((num_nodes,), -1.0)
            same_to_parent[dst.long()] = edge_same_weight.to(dtype=z_diff.dtype)

        child = torch.arange(num_nodes, device=z_diff.device)
        parent = parent_of
        valid = parent >= 0
        grandparent = parent_of[parent.clamp_min(0)]
        valid = valid & (grandparent >= 0)
        if not valid.any():
            return z_diff

        child = child[valid]
        parent = parent[valid]
        grandparent = grandparent[valid]

        if same_to_parent is not None:
            same_child_parent = same_to_parent[child].clamp(0.0, 1.0)
            same_parent_grand = same_to_parent[parent].clamp(0.0, 1.0)
        else:
            same_child_parent = (route_probs[child] * route_probs[parent]).sum(dim=-1)
            same_parent_grand = (route_probs[parent] * route_probs[grandparent]).sum(dim=-1)
        diff_child_parent = (1.0 - same_child_parent).clamp(0.0, 1.0)
        diff_parent_grand = (1.0 - same_parent_grand).clamp(0.0, 1.0)

        active = diff_child_parent.view(-1, 1)
        if active.max().item() <= 0:
            return z_diff

        scale = self.hidden_dim ** -0.5
        query = self.transition_query(z_diff[parent])
        diff_key = self.transition_key(z_diff[grandparent])
        same_key = self.transition_key(z_same[grandparent])
        diff_score = (query * diff_key).sum(dim=-1) * scale
        same_score = (query * same_key).sum(dim=-1) * scale
        diff_score = diff_score + torch.log(diff_parent_grand.clamp_min(1e-6))
        same_score = same_score + torch.log(same_parent_grand.clamp_min(1e-6))

        attn = F.softmax(torch.stack((diff_score, same_score), dim=-1), dim=-1)
        diff_value = self.transition_value(z_diff[grandparent])
        same_value = self.transition_value(z_same[grandparent])
        context = attn[:, :1] * diff_value + attn[:, 1:] * same_value
        context = active * context

        out = z_diff.new_zeros(z_diff.size())
        denom = z_diff.new_zeros(num_nodes, 1)
        out.index_add_(0, parent, context)
        denom.index_add_(0, parent, active)
        out = out / denom.clamp_min(1e-6)

        delta = self.transition_out(out)
        delta = F.dropout(delta, self.dropout, training=self.training)
        updated = self.transition_norm(
            z_diff + self.transition_attention_blend * delta
        )
        return torch.where(denom > 0, updated, z_diff)

    def _apply_reply_target_attention(
        self,
        z_same,
        z_diff,
        tree_edge_index,
        route_probs=None,
        tree_same_weight=None,
    ):
        if tree_same_weight is None:
            src, dst = self._directed_tree_edges(tree_edge_index)
        else:
            src, dst, edge_same_weight = self._directed_tree_edges_with_attr(
                tree_edge_index,
                tree_same_weight,
            )
        if src.numel() == 0:
            return z_diff

        num_nodes = z_diff.size(0)
        parent_of = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=z_diff.device,
        )
        parent_of[dst.long()] = src.long()
        same_to_parent = None
        if tree_same_weight is not None:
            same_to_parent = z_diff.new_full((num_nodes,), -1.0)
            same_to_parent[dst.long()] = edge_same_weight.to(dtype=z_diff.dtype)

        child = torch.arange(num_nodes, device=z_diff.device)
        parent = parent_of
        valid = parent >= 0
        grandparent = parent_of[parent.clamp_min(0)]
        valid = valid & (grandparent >= 0)
        if not valid.any():
            return z_diff

        child = child[valid]
        parent = parent[valid]
        grandparent = grandparent[valid]

        if same_to_parent is not None:
            same_child_parent = same_to_parent[child].clamp(0.0, 1.0)
            same_parent_grand = same_to_parent[parent].clamp(0.0, 1.0)
        else:
            same_child_parent = (route_probs[child] * route_probs[parent]).sum(dim=-1)
            same_parent_grand = (route_probs[parent] * route_probs[grandparent]).sum(dim=-1)
        diff_child_parent = (1.0 - same_child_parent).clamp(0.0, 1.0)
        diff_parent_grand = (1.0 - same_parent_grand).clamp(0.0, 1.0)

        active = diff_child_parent.view(-1, 1)
        if active.max().item() <= 0:
            return z_diff

        scale = self.hidden_dim ** -0.5
        query = self.transition_query(z_diff[child])
        parent_key = self.transition_key(z_diff[parent])
        same_grand_key = self.transition_key(z_same[grandparent])
        diff_grand_key = self.transition_key(z_diff[grandparent])

        parent_score = (query * parent_key).sum(dim=-1) * scale
        same_grand_score = (query * same_grand_key).sum(dim=-1) * scale
        diff_grand_score = (query * diff_grand_key).sum(dim=-1) * scale
        same_grand_score = same_grand_score + torch.log(same_parent_grand.clamp_min(1e-6))
        diff_grand_score = diff_grand_score + torch.log(diff_parent_grand.clamp_min(1e-6))

        scores = torch.stack(
            (parent_score, same_grand_score, diff_grand_score),
            dim=-1,
        )
        attn = F.softmax(scores, dim=-1)

        parent_value = self.transition_value(z_diff[parent])
        same_grand_value = self.transition_value(z_same[grandparent])
        diff_grand_value = self.transition_value(z_diff[grandparent])
        values = torch.stack(
            (parent_value, same_grand_value, diff_grand_value),
            dim=1,
        )
        context = (attn.unsqueeze(-1) * values).sum(dim=1)
        context = active * context

        out = z_diff.new_zeros(z_diff.size())
        denom = z_diff.new_zeros(num_nodes, 1)
        out.index_add_(0, child, context)
        denom.index_add_(0, child, active)
        out = out / denom.clamp_min(1e-6)

        delta = self.transition_out(out)
        delta = F.dropout(delta, self.dropout, training=self.training)
        updated = self.transition_norm(
            z_diff + self.reply_target_attention_blend * delta
        )
        return torch.where(denom > 0, updated, z_diff)

    def _topk_mask_by_graph(self, scores, batch):
        scores = scores.view(-1)
        num_nodes = scores.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=scores.device)

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=scores.device)
        for graph_id in torch.unique(batch):
            idx = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            k = int(math.ceil(float(idx.numel()) * self.cross_view_topk_ratio))
            k = max(self.cross_view_min_nodes, k)
            k = min(k, int(idx.numel()))
            top_idx = torch.topk(scores[idx], k=k, largest=True).indices
            mask[idx[top_idx]] = True
        return mask

    def _attend_selected_by_graph(self, query_nodes, key_value_nodes, selected_mask, batch):
        num_nodes = query_nodes.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=query_nodes.device)

        out = query_nodes.new_zeros(query_nodes.size())
        has_context = torch.zeros(num_nodes, 1, dtype=torch.bool, device=query_nodes.device)
        scale = self.hidden_dim ** -0.5

        for graph_id in torch.unique(batch):
            query_idx = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            selected_idx = ((batch == graph_id) & selected_mask).nonzero(as_tuple=False).view(-1)
            if query_idx.numel() == 0 or selected_idx.numel() == 0:
                continue

            query = self.cross_query(query_nodes[query_idx])
            key = self.cross_key(key_value_nodes[selected_idx])
            value = self.cross_value(key_value_nodes[selected_idx])
            attn = torch.matmul(query, key.transpose(0, 1)) * scale
            attn = F.softmax(attn, dim=-1)
            out[query_idx] = torch.matmul(attn, value)
            has_context[query_idx] = True

        return out, has_context

    def _apply_cross_view_attention(self, z_same, z_diff, batch):
        same_scores = self.cross_same_score(z_same)
        diff_scores = self.cross_diff_score(z_diff)
        same_selected = self._topk_mask_by_graph(same_scores, batch)
        diff_selected = self._topk_mask_by_graph(diff_scores, batch)

        same_context, same_has_context = self._attend_selected_by_graph(
            z_same,
            z_diff,
            diff_selected,
            batch,
        )
        diff_context, diff_has_context = self._attend_selected_by_graph(
            z_diff,
            z_same,
            same_selected,
            batch,
        )

        same_delta = self.cross_out(same_context)
        diff_delta = self.cross_out(diff_context)
        same_delta = F.dropout(same_delta, self.dropout, training=self.training)
        diff_delta = F.dropout(diff_delta, self.dropout, training=self.training)

        z_same_updated = self.cross_same_norm(
            z_same + self.cross_view_attention_blend * same_delta
        )
        z_diff_updated = self.cross_diff_norm(
            z_diff + self.cross_view_attention_blend * diff_delta
        )
        z_same = torch.where(same_has_context, z_same_updated, z_same)
        z_diff = torch.where(diff_has_context, z_diff_updated, z_diff)
        return z_same, z_diff

    def forward(
        self,
        h,
        edge_index,
        node_state=None,
        edge_stance=None,
        return_views=False,
        tree_edge_index=None,
        batch=None,
    ):
        zero = self._zero(h)
        self._last_aux_loss = zero
        self._last_node_state_loss = zero
        self._last_edge_same_loss = zero
        self._last_sep_loss = zero
        self._last_same_weight_mean = zero.detach()
        self._last_logvar_mean = zero.detach()
        self._last_precision_mean = zero.detach()

        logits = self.state_head(h)
        probs = F.softmax(logits / self.temp, dim=-1)
        valid_node = self._valid_node_mask(node_state, h.size(0))

        node_loss = zero
        if self.use_node_state_aux and valid_node is not None and valid_node.any():
            node_loss = F.cross_entropy(logits[valid_node], node_state.view(-1)[valid_node].long())

        if edge_index.numel() == 0:
            aux = self.lambda_node_state * node_loss
            self._last_aux_loss = aux
            self._last_node_state_loss = node_loss.detach()
            if return_views:
                return h, h
            return h

        src, dst = edge_index
        route_probs = probs.detach() if self.detach_route else probs
        tree_same_weight = None
        if self.edge_relation_routing:
            same_weight = self._edge_same_probability(h, edge_index).clamp(0.0, 1.0)
            if tree_edge_index is not None:
                tree_same_weight = self._edge_same_probability(h, tree_edge_index).clamp(0.0, 1.0)
        else:
            same_weight = (route_probs[src] * route_probs[dst]).sum(dim=-1).clamp(0.0, 1.0)
        diff_weight = (1.0 - same_weight).clamp(0.0, 1.0)

        edge_loss = zero
        if self.edge_relation_routing and edge_stance is not None:
            edge_stance = edge_stance.view(-1).long()
            if edge_stance.size(0) == same_weight.size(0):
                valid_edge = (edge_stance == 0) | (edge_stance == 1)
                if valid_edge.any():
                    true_same = (edge_stance == 0).to(dtype=h.dtype)
                    edge_loss = F.binary_cross_entropy(
                        same_weight[valid_edge],
                        true_same[valid_edge],
                    )
        elif valid_node is not None:
            valid_edge = valid_node[src] & valid_node[dst]
            if valid_edge.any():
                flat_state = node_state.view(-1).long()
                true_same = (flat_state[src] == flat_state[dst]).to(dtype=h.dtype)
                edge_loss = F.binary_cross_entropy(
                    same_weight[valid_edge],
                    true_same[valid_edge],
                )

        same_aggr, logvar_mean, precision_mean = self._aggregate_same_uncertain(
            h,
            edge_index,
            same_weight,
        )
        diff_aggr = self._aggregate_diff(h, edge_index, diff_weight)

        diff_summary = self.diff_summary_norm(diff_aggr - h)
        diff_gate = torch.sigmoid(self.diff_gate_mlp(torch.cat([h, diff_summary], dim=-1)))
        gated_diff_summary = diff_summary * diff_gate
        diff_delta = self.diff_path_mlp(torch.cat([h, gated_diff_summary], dim=-1))
        diff_delta = F.dropout(diff_delta, self.dropout, training=self.training)
        z_diff = self.diff_path_norm(h + self.diff_path_blend * diff_delta)

        same_summary = self.same_summary_norm(same_aggr - h)
        same_gate = torch.sigmoid(self.same_gate_mlp(torch.cat([h, same_summary], dim=-1)))
        guided_gate = same_gate * ((1.0 - self.gate_blend) + self.gate_blend * diff_gate)
        guided_same_summary = same_summary * guided_gate
        same_delta = self.same_path_mlp(torch.cat([h, guided_same_summary], dim=-1))
        same_delta = F.dropout(same_delta, self.dropout, training=self.training)
        z_same = self.same_path_norm(h + self.same_path_blend * same_delta)

        if self.transition_attention:
            if tree_edge_index is None:
                tree_edge_index = edge_index
            if self.edge_relation_routing and tree_same_weight is None:
                tree_same_weight = self._edge_same_probability(h, tree_edge_index).clamp(0.0, 1.0)
            z_diff = self._apply_transition_attention(
                z_same,
                z_diff,
                tree_edge_index,
                route_probs,
                tree_same_weight=tree_same_weight,
            )
        if self.reply_target_attention:
            if tree_edge_index is None:
                tree_edge_index = edge_index
            if self.edge_relation_routing and tree_same_weight is None:
                tree_same_weight = self._edge_same_probability(h, tree_edge_index).clamp(0.0, 1.0)
            z_diff = self._apply_reply_target_attention(
                z_same,
                z_diff,
                tree_edge_index,
                route_probs,
                tree_same_weight=tree_same_weight,
            )
        if self.cross_view_attention:
            z_same, z_diff = self._apply_cross_view_attention(z_same, z_diff, batch)

        sep_loss = (F.normalize(z_same, dim=1) * F.normalize(z_diff, dim=1)).sum(dim=1)
        sep_loss = sep_loss.pow(2).mean()

        aux = (
            self.lambda_node_state * node_loss
            + self.lambda_edge_same * edge_loss
            + self.lambda_sep * sep_loss
        )
        self._last_aux_loss = aux
        self._last_node_state_loss = node_loss.detach()
        self._last_edge_same_loss = edge_loss.detach()
        self._last_sep_loss = sep_loss.detach()
        self._last_same_weight_mean = same_weight.mean().detach()
        self._last_logvar_mean = logvar_mean
        self._last_precision_mean = precision_mean
        if return_views:
            return z_same, z_diff
        return z_same

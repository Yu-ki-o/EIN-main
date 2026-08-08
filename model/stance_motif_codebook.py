"""Local stance-motif codebook branch.

The branch deliberately does not compose support/deny labels with XOR.  A
root-to-leaf stance sequence is represented by its overlapping two-edge
windows instead.  For example, ``SSS`` produces ``SS -> SS`` and ``SDS``
produces ``SD -> DS``.

The LLM-derived hard edge labels determine one of four structural types
(``SS``, ``SD``, ``DS``, or ``DD``).  Within each type, instance semantics
select one of several learnable prototypes, which allows the same structural
pattern (especially ``DD``) to have more than one semantic realization.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax


class StanceMotifCodebookBranch(nn.Module):
    """Encode overlapping hard-label stance motifs into a graph vector.

    Edge labels follow the convention already used by the uncertainty semantic
    change models: ``0`` is support and ``1`` is deny.  A motif type id is
    therefore ``2 * first_label + second_label``:

    ``0=SS, 1=SD, 2=DS, 3=DD``.
    """

    motif_names = ("SS", "SD", "DS", "DD")

    def __init__(self, hidden_dim, max_hop, args=None):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_hop = max(2, int(max_hop))
        self.num_types = len(self.motif_names)
        self.num_prototypes = max(
            1,
            int(getattr(args, "codebook_prototypes_per_type", 4)),
        )
        self.dropout = max(
            0.0,
            float(
                getattr(
                    args,
                    "codebook_dropout",
                    getattr(args, "dropout", 0.1),
                )
            ),
        )
        self.prototype_loss_weight = max(
            0.0,
            float(getattr(args, "lambda_codebook_prototype_aux", 0.05)),
        )
        self.commitment_loss_weight = max(
            0.0,
            float(getattr(args, "lambda_codebook_commitment_aux", 0.0125)),
        )
        self.separation_loss_weight = max(
            0.0,
            float(getattr(args, "lambda_codebook_separation_aux", 0.01)),
        )
        self.data_initialize = bool(
            getattr(args, "codebook_data_initialize", True)
        )
        self.eps = 1e-6

        # Three node vectors, two absolute changes, and two pairwise products.
        motif_input_dim = self.hidden_dim * 7
        self.motif_encoder = nn.Sequential(
            nn.Linear(motif_input_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.codebook = nn.Parameter(
            torch.empty(
                self.num_types,
                self.num_prototypes,
                self.hidden_dim,
            )
        )
        nn.init.normal_(self.codebook, mean=0.0, std=self.hidden_dim ** -0.5)
        self.register_buffer(
            "codebook_initialized",
            torch.zeros(
                self.num_types,
                self.num_prototypes,
                dtype=torch.bool,
            ),
        )

        self.depth_embedding = nn.Embedding(
            self.max_hop + 1,
            self.hidden_dim,
        )
        self.prototype_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
        )
        self.prototype_scale = nn.Parameter(torch.tensor(1.0))
        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.path_encoder = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.empty_motif = nn.Parameter(torch.zeros(self.hidden_dim))
        self.graph_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.last_aux_loss = None
        self.last_prototype_loss = None
        self.last_commitment_loss = None
        self.last_separation_loss = None
        self.last_motif_types = None
        self.last_prototype_indices = None
        self.last_motif_depth = None
        self.last_motif_batch = None
        self.last_motif_attention = None
        self.last_motif_tokens = None
        self.last_path_tokens = None
        self.last_graph = None

    @staticmethod
    def _batch_size(batch):
        if batch.numel() == 0:
            return 0
        return int(batch.max().item()) + 1

    def _extract_motifs(self, edge_index, edge_stance, depth, batch):
        """Return all valid overlapping two-edge windows in the batch."""
        device = depth.device
        num_nodes = int(depth.numel())
        empty = torch.empty(0, dtype=torch.long, device=device)
        if edge_index.numel() == 0 or edge_stance is None:
            return {
                "grandparent": empty,
                "parent": empty,
                "child": empty,
                "type": empty,
                "depth": empty,
                "batch": empty,
                "parent_token": empty,
            }

        labels = edge_stance.view(-1).long().to(device=device)
        if labels.numel() != edge_index.size(1):
            raise ValueError(
                "edge_stance must contain one hard label per edge for the "
                "codebook branch; got {} labels for {} edges".format(
                    labels.numel(),
                    edge_index.size(1),
                )
            )

        src, dst = edge_index.long()
        valid_tree_edge = (
            (depth[src] >= 0)
            & (depth[dst] == depth[src] + 1)
            & (batch[src] == batch[dst])
        )
        tree_edge_ids = valid_tree_edge.nonzero(as_tuple=False).view(-1)
        if tree_edge_ids.numel() == 0:
            return {
                "grandparent": empty,
                "parent": empty,
                "child": empty,
                "type": empty,
                "depth": empty,
                "batch": empty,
                "parent_token": empty,
            }

        incoming_edge = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        incoming_edge[dst[tree_edge_ids]] = tree_edge_ids

        child = ((depth >= 2) & (incoming_edge >= 0)).nonzero(
            as_tuple=False
        ).view(-1)
        if child.numel() == 0:
            return {
                "grandparent": empty,
                "parent": empty,
                "child": empty,
                "type": empty,
                "depth": empty,
                "batch": empty,
                "parent_token": empty,
            }

        second_edge = incoming_edge[child]
        parent = src[second_edge]
        first_edge = incoming_edge[parent]
        valid_parent = first_edge >= 0
        child = child[valid_parent]
        second_edge = second_edge[valid_parent]
        parent = parent[valid_parent]
        first_edge = first_edge[valid_parent]
        if child.numel() == 0:
            return {
                "grandparent": empty,
                "parent": empty,
                "child": empty,
                "type": empty,
                "depth": empty,
                "batch": empty,
                "parent_token": empty,
            }

        first_label = labels[first_edge]
        second_label = labels[second_edge]
        valid_label = (
            ((first_label == 0) | (first_label == 1))
            & ((second_label == 0) | (second_label == 1))
        )
        child = child[valid_label]
        parent = parent[valid_label]
        first_edge = first_edge[valid_label]
        first_label = first_label[valid_label]
        second_label = second_label[valid_label]
        if child.numel() == 0:
            return {
                "grandparent": empty,
                "parent": empty,
                "child": empty,
                "type": empty,
                "depth": empty,
                "batch": empty,
                "parent_token": empty,
            }

        grandparent = src[first_edge]
        motif_type = first_label * 2 + second_label
        motif_depth = depth[child]
        motif_batch = batch[child].long()

        # A token's predecessor is the motif ending at its parent.  This makes
        # overlapping windows an ordered path: SDS -> SD followed by DS.
        node_to_token = torch.full(
            (num_nodes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        node_to_token[child] = torch.arange(child.numel(), device=device)
        parent_token = node_to_token[parent]
        return {
            "grandparent": grandparent,
            "parent": parent,
            "child": child,
            "type": motif_type,
            "depth": motif_depth,
            "batch": motif_batch,
            "parent_token": parent_token,
        }

    def _initialize_from_batch(self, semantic, motif_type):
        if not self.training or not self.data_initialize or semantic.numel() == 0:
            return
        with torch.no_grad():
            for type_id in range(self.num_types):
                missing = (~self.codebook_initialized[type_id]).nonzero(
                    as_tuple=False
                ).view(-1)
                candidates = semantic[motif_type == type_id]
                if missing.numel() == 0 or candidates.size(0) == 0:
                    continue
                fill_count = min(int(missing.numel()), int(candidates.size(0)))
                if self.num_prototypes == 1:
                    seeds = candidates.mean(dim=0, keepdim=True)
                    fill_count = 1
                else:
                    # Deterministic farthest-first seeds provide a data anchor
                    # without an offline k-means preprocessing pass.
                    chosen = [0]
                    while len(chosen) < fill_count:
                        selected = candidates[chosen]
                        distance = torch.cdist(candidates, selected).min(dim=1).values
                        distance[torch.tensor(chosen, device=distance.device)] = -1
                        chosen.append(int(distance.argmax().item()))
                    seeds = candidates[chosen]
                target = missing[:fill_count]
                self.codebook[type_id, target].copy_(seeds[:fill_count])
                self.codebook_initialized[type_id, target] = True

    def _select_prototypes(self, semantic, motif_type):
        typed_codebook = self.codebook[motif_type]
        semantic_normalized = F.normalize(semantic, dim=-1, eps=self.eps)
        prototype_normalized = F.normalize(
            typed_codebook,
            dim=-1,
            eps=self.eps,
        )
        similarity = torch.einsum(
            "th,tmh->tm",
            semantic_normalized,
            prototype_normalized,
        )
        initialized = self.codebook_initialized[motif_type]
        has_initialized = initialized.any(dim=-1, keepdim=True)
        allowed = initialized | (~has_initialized)
        similarity = similarity.masked_fill(~allowed, float("-inf"))
        prototype_index = similarity.argmax(dim=-1)
        token_index = torch.arange(semantic.size(0), device=semantic.device)
        selected = typed_codebook[token_index, prototype_index]
        return selected, prototype_index

    def _separation_loss(self):
        flat = F.normalize(
            self.codebook.reshape(-1, self.hidden_dim),
            dim=-1,
            eps=self.eps,
        )
        if flat.size(0) <= 1:
            return flat.new_zeros(())
        gram = flat.matmul(flat.t())
        identity = torch.eye(
            gram.size(0),
            device=gram.device,
            dtype=gram.dtype,
        )
        return (gram - identity).pow(2).sum() / (
            gram.numel() - gram.size(0)
        )

    def _encode_ordered_paths(self, tokens, motif_depth, parent_token):
        context = torch.zeros_like(tokens)
        if tokens.numel() == 0:
            return context
        min_depth = int(motif_depth.min().item())
        max_depth = int(motif_depth.max().item())
        for depth_id in range(min_depth, max_depth + 1):
            token_ids = (motif_depth == depth_id).nonzero(
                as_tuple=False
            ).view(-1)
            if token_ids.numel() == 0:
                continue
            predecessor = parent_token[token_ids]
            predecessor_hidden = tokens.new_zeros(
                token_ids.numel(),
                self.hidden_dim,
            )
            has_predecessor = predecessor >= 0
            if has_predecessor.any():
                predecessor_hidden[has_predecessor] = context[
                    predecessor[has_predecessor]
                ]
            updated = self.path_encoder(tokens[token_ids], predecessor_hidden)
            context = context.index_copy(0, token_ids, updated)
        return context

    def forward(
        self,
        node_hidden,
        edge_index,
        edge_stance,
        depth,
        batch,
        roots,
    ):
        if edge_stance is None and edge_index.numel() > 0:
            raise ValueError(
                "classification_fusion_mode includes 'codebook', but the "
                "batch has no edge_stance hard labels"
            )

        batch_size = self._batch_size(batch)
        root_hidden = node_hidden[roots]
        motifs = self._extract_motifs(
            edge_index,
            edge_stance,
            depth,
            batch,
        )

        if motifs["child"].numel() == 0:
            pooled = self.empty_motif.unsqueeze(0).expand(batch_size, -1)
            graph_hidden = self.graph_fusion(
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
            self.last_aux_loss = zero
            self.last_prototype_loss = zero
            self.last_commitment_loss = zero
            self.last_separation_loss = self._separation_loss()
            self.last_motif_types = motifs["type"]
            self.last_prototype_indices = motifs["type"]
            self.last_motif_depth = motifs["depth"]
            self.last_motif_batch = motifs["batch"]
            self.last_motif_attention = node_hidden.new_zeros((0,))
            self.last_motif_tokens = node_hidden.new_zeros(
                (0, self.hidden_dim)
            )
            self.last_path_tokens = node_hidden.new_zeros(
                (0, self.hidden_dim)
            )
            self.last_graph = graph_hidden
            return graph_hidden, {
                "aux_loss": zero,
                "prototype_loss": zero,
                "commitment_loss": zero,
                "separation_loss": self.last_separation_loss,
                "motif_type": motifs["type"],
                "prototype_index": motifs["type"],
                "motif_depth": motifs["depth"],
                "motif_batch": motifs["batch"],
                "attention": self.last_motif_attention,
                "tokens": self.last_motif_tokens,
                "path_tokens": self.last_path_tokens,
            }

        grandparent_hidden = node_hidden[motifs["grandparent"]]
        parent_hidden = node_hidden[motifs["parent"]]
        child_hidden = node_hidden[motifs["child"]]
        semantic = self.motif_encoder(
            torch.cat(
                (
                    grandparent_hidden,
                    parent_hidden,
                    child_hidden,
                    (grandparent_hidden - parent_hidden).abs(),
                    (parent_hidden - child_hidden).abs(),
                    grandparent_hidden * parent_hidden,
                    parent_hidden * child_hidden,
                ),
                dim=-1,
            )
        )
        self._initialize_from_batch(semantic, motifs["type"])
        prototype, prototype_index = self._select_prototypes(
            semantic,
            motifs["type"],
        )

        prototype_loss = F.mse_loss(prototype, semantic.detach())
        commitment_loss = F.mse_loss(semantic, prototype.detach())
        separation_loss = self._separation_loss()
        aux_loss = (
            self.prototype_loss_weight * prototype_loss
            + self.commitment_loss_weight * commitment_loss
            + self.separation_loss_weight * separation_loss
        )

        depth_id = motifs["depth"].clamp(0, self.max_hop)
        tokens = self.token_norm(
            semantic
            + self.prototype_scale
            * self.prototype_projection(prototype)
            + self.depth_embedding(depth_id)
        )
        path_tokens = self._encode_ordered_paths(
            tokens,
            motifs["depth"],
            motifs["parent_token"],
        )

        token_root = root_hidden[motifs["batch"]]
        attention_score = self.attention(
            torch.cat(
                (
                    path_tokens,
                    token_root,
                    (path_tokens - token_root).abs(),
                    path_tokens * token_root,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        attention_weight = softmax(attention_score, motifs["batch"])
        pooled = global_add_pool(
            path_tokens * attention_weight.unsqueeze(-1),
            motifs["batch"],
            size=batch_size,
        )
        motif_count = torch.bincount(
            motifs["batch"],
            minlength=batch_size,
        )
        no_motif = motif_count == 0
        if no_motif.any():
            pooled = pooled.clone()
            pooled[no_motif] = self.empty_motif

        graph_hidden = self.graph_fusion(
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

        self.last_aux_loss = aux_loss
        self.last_prototype_loss = prototype_loss
        self.last_commitment_loss = commitment_loss
        self.last_separation_loss = separation_loss
        self.last_motif_types = motifs["type"]
        self.last_prototype_indices = prototype_index
        self.last_motif_depth = motifs["depth"]
        self.last_motif_batch = motifs["batch"]
        self.last_motif_attention = attention_weight
        self.last_motif_tokens = tokens
        self.last_path_tokens = path_tokens
        self.last_graph = graph_hidden
        return graph_hidden, {
            "aux_loss": aux_loss,
            "prototype_loss": prototype_loss,
            "commitment_loss": commitment_loss,
            "separation_loss": separation_loss,
            "motif_type": motifs["type"],
            "prototype_index": prototype_index,
            "motif_depth": motifs["depth"],
            "motif_batch": motifs["batch"],
            "attention": attention_weight,
            "tokens": tokens,
            "path_tokens": path_tokens,
        }

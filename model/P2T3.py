import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils.p2t3 import build_p2t3_sequence_metadata


class FF(nn.Module):
    """Residual discriminator used by the released MI pre-training loss."""

    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


def sinusoidal_depth_embedding(max_depth, d_model):
    max_depth = max(0, int(max_depth))
    d_model = int(d_model)
    encoding = torch.zeros(max_depth + 1, d_model)
    position = torch.arange(max_depth + 1, dtype=torch.float32).unsqueeze(1)
    even_dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)
    angle = position / torch.pow(10000.0, even_dimensions / d_model)
    encoding[:, 0::2] = torch.sin(angle)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(angle[:, : encoding[:, 1::2].size(1)])
    return encoding


class P2T3(nn.Module):
    """Pre-Trained Propagation Tree Transformer adapted to EIN's PyG API.

    The released implementation expects pre-extracted conversation-chain
    sequences. This adapter consumes an EIN ``Data``/``Batch`` directly and
    uses cached P2T3 metadata when available, with an on-the-fly fallback for
    standalone graph objects.
    """

    def __init__(
        self,
        in_feats,
        hid_feats,
        out_feats,
        num_classes,
        args,
        device,
    ):
        super().__init__()
        self.args = args
        self.device = device
        self.in_feats = int(in_feats)
        self.d_model = int(getattr(args, "p2t3_d_model", hid_feats))
        self.num_classes = int(num_classes)
        self.num_layers = max(1, int(getattr(args, "p2t3_num_layers", 3)))
        self.num_heads = max(1, int(getattr(args, "p2t3_num_heads", 8)))
        self.dim_feedforward = max(
            self.d_model,
            int(getattr(args, "p2t3_dim_feedforward", self.d_model * 2)),
        )
        self.dropout = float(getattr(args, "p2t3_dropout", 0.1))
        self.max_sequence_length = max(
            1,
            int(getattr(args, "p2t3_max_sequence_length", 1000)),
        )
        self.max_chain_length = max(
            1,
            int(getattr(args, "p2t3_max_chain_length", 40)),
        )
        self.max_chain_identifiers = max(
            1,
            int(
                getattr(
                    args,
                    "p2t3_max_chain_identifiers",
                    self.d_model,
                )
            ),
        )
        self.max_depth = max(
            self.max_chain_length,
            int(getattr(args, "p2t3_max_depth", self.max_chain_length)),
        )
        self.use_chain_identifier = bool(
            getattr(args, "p2t3_use_chain_identifier", True)
        )
        self.use_depth_embedding = bool(
            getattr(args, "p2t3_use_depth_embedding", True)
        )
        self.use_type_embedding = bool(
            getattr(args, "p2t3_use_type_embedding", True)
        )
        self.unsup_weight = float(getattr(args, "p2t3_unsup_weight", 0.0))
        self.measure = str(getattr(args, "p2t3_measure", "JSD")).upper()
        if self.measure != "JSD":
            raise ValueError(
                "The EIN P2T3 adapter currently supports p2t3_measure=JSD, "
                "matching the released default."
            )
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "p2t3_d_model {} must be divisible by p2t3_num_heads {}.".format(
                    self.d_model,
                    self.num_heads,
                )
            )
        if self.max_chain_identifiers > self.d_model:
            raise ValueError(
                "The released additive chain encoding requires "
                "p2t3_max_chain_identifiers <= p2t3_d_model, got {} > {}.".format(
                    self.max_chain_identifiers,
                    self.d_model,
                )
            )

        self.input_projection = (
            nn.Identity()
            if self.in_feats == self.d_model
            else nn.Linear(self.in_feats, self.d_model)
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )

        gaussian = torch.randn(self.d_model, self.d_model)
        orthogonal, _ = torch.linalg.qr(gaussian, mode="reduced")
        self.register_buffer(
            "chain_identifier_matrix",
            orthogonal[: self.max_chain_identifiers],
        )
        self.register_buffer(
            "depth_embedding_matrix",
            sinusoidal_depth_embedding(self.max_depth, self.d_model),
            persistent=False,
        )
        type_embedding = torch.arange(3, dtype=torch.float32).view(3, 1)
        self.register_buffer(
            "type_embedding_matrix",
            type_embedding.expand(3, self.d_model).clone(),
            persistent=False,
        )

        self.local_d = FF(self.d_model, self.d_model)
        self.global_d = FF(self.d_model, self.d_model)
        self.lin_class = nn.Linear(self.d_model, self.num_classes)

        class_weights = getattr(args, "classification_class_weights", None)
        if class_weights is None:
            self.register_buffer(
                "classification_class_weights",
                torch.empty(0),
                persistent=False,
            )
        else:
            self.register_buffer(
                "classification_class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
                persistent=False,
            )

        self._last_sequence_lengths = None
        self._last_valid_mask = None
        self._last_level_one_mask = None
        self._last_root_embeddings = None
        self._auxiliary_term = None

    def init_optimizer(self, args):
        return torch.optim.Adam(
            self.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def classification_loss(self, output, target):
        weight = (
            self.classification_class_weights
            if self.classification_class_weights.numel() > 0
            else None
        )
        return F.nll_loss(output, target.view(-1).long(), weight=weight)

    def physics_loss(self, U, S, D, true_state):
        return self.lin_class.weight.new_zeros(())

    @staticmethod
    def _batch_vector(data):
        if hasattr(data, "batch"):
            return data.batch.long()
        return torch.zeros(
            data.x.size(0),
            dtype=torch.long,
            device=data.x.device,
        )

    @staticmethod
    def _graph_ptr(data):
        if hasattr(data, "ptr"):
            return data.ptr.long()
        return torch.tensor(
            [0, data.x.size(0)],
            dtype=torch.long,
            device=data.x.device,
        )

    def _cached_metadata(self, data):
        required = (
            "p2t3_node_id",
            "p2t3_chain_id",
            "p2t3_depth",
            "p2t3_type_id",
            "p2t3_level_one_mask",
            "p2t3_sequence_length",
        )
        if not all(hasattr(data, name) for name in required):
            return None
        return {
            name: getattr(data, name)
            for name in required
        }

    def _dynamic_metadata(self, data):
        ptr = self._graph_ptr(data)
        edge_index = getattr(
            data,
            "directed_edge_index",
            data.edge_index,
        ).long()
        metadata_by_graph = []
        for graph_id in range(ptr.numel() - 1):
            start = int(ptr[graph_id].item())
            end = int(ptr[graph_id + 1].item())
            edge_mask = (
                (edge_index[0] >= start)
                & (edge_index[0] < end)
                & (edge_index[1] >= start)
                & (edge_index[1] < end)
            )
            local_edges = edge_index[:, edge_mask] - start
            metadata_by_graph.append(
                build_p2t3_sequence_metadata(
                    local_edges,
                    end - start,
                    max_sequence_length=self.max_sequence_length,
                    max_chain_length=self.max_chain_length,
                    max_chain_identifiers=self.max_chain_identifiers,
                )
            )

        names = metadata_by_graph[0].keys()
        return {
            name: torch.cat(
                [metadata[name] for metadata in metadata_by_graph],
                dim=0,
            ).to(data.x.device)
            for name in names
        }

    def _sequence_metadata(self, data):
        metadata = self._cached_metadata(data)
        if metadata is None:
            metadata = self._dynamic_metadata(data)

        lengths = metadata["p2t3_sequence_length"].view(-1).long()
        total_tokens = int(lengths.sum().item())
        for name in (
            "p2t3_node_id",
            "p2t3_chain_id",
            "p2t3_depth",
            "p2t3_type_id",
            "p2t3_level_one_mask",
        ):
            if metadata[name].numel() != total_tokens:
                raise ValueError(
                    "Invalid cached P2T3 metadata: {} has {} values, "
                    "expected {}.".format(
                        name,
                        metadata[name].numel(),
                        total_tokens,
                    )
                )
        return metadata, lengths

    def _dense_token_input(self, data):
        metadata, lengths = self._sequence_metadata(data)
        num_graphs = lengths.numel()
        token_graph = torch.repeat_interleave(
            torch.arange(num_graphs, device=data.x.device),
            lengths,
        )
        token_starts = torch.cat(
            (
                lengths.new_zeros(1),
                lengths.cumsum(dim=0)[:-1],
            )
        )
        token_position = (
            torch.arange(token_graph.numel(), device=data.x.device)
            - torch.repeat_interleave(token_starts, lengths)
        )

        ptr = self._graph_ptr(data)
        local_node_id = metadata["p2t3_node_id"].long()
        global_node_id = ptr[token_graph] + local_node_id
        if (
            global_node_id.numel() == 0
            or global_node_id.min() < 0
            or global_node_id.max() >= data.x.size(0)
        ):
            raise ValueError("P2T3 sequence metadata references an invalid node.")

        token_hidden = self.input_projection(data.x.float())[global_node_id]
        chain_id = metadata["p2t3_chain_id"].long()
        depth = metadata["p2t3_depth"].long()
        type_id = metadata["p2t3_type_id"].long()
        if self.use_chain_identifier:
            if chain_id.max() >= self.max_chain_identifiers:
                raise ValueError(
                    "P2T3 chain identifier exceeds the configured matrix size."
                )
            token_hidden = token_hidden + self.chain_identifier_matrix[chain_id]
        if self.use_depth_embedding:
            token_hidden = token_hidden + self.depth_embedding_matrix[
                depth.clamp(0, self.max_depth)
            ]
        if self.use_type_embedding:
            token_hidden = token_hidden + self.type_embedding_matrix[
                type_id.clamp(0, 2)
            ]

        max_length = int(lengths.max().item())
        dense_input = token_hidden.new_zeros(
            num_graphs,
            max_length,
            self.d_model,
        )
        valid_mask = torch.zeros(
            num_graphs,
            max_length,
            dtype=torch.bool,
            device=data.x.device,
        )
        level_one_mask = torch.zeros_like(valid_mask)
        dense_input[token_graph, token_position] = token_hidden
        valid_mask[token_graph, token_position] = True
        level_one_mask[token_graph, token_position] = metadata[
            "p2t3_level_one_mask"
        ].bool()
        return dense_input, valid_mask, level_one_mask, lengths

    def encode_sequences(self, data):
        dense_input, valid_mask, level_one_mask, lengths = (
            self._dense_token_input(data)
        )
        encoded = self.transformer_encoder(
            dense_input,
            src_key_padding_mask=~valid_mask,
        )
        encoded = encoded.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return encoded, valid_mask, level_one_mask, lengths

    def _jsd_local_global_loss(
        self,
        encoded,
        level_one_mask,
        global_embeddings,
    ):
        num_graphs = global_embeddings.size(0)
        if num_graphs < 2 or not level_one_mask.any():
            return global_embeddings.new_zeros(())

        graph_grid = torch.arange(
            num_graphs,
            device=encoded.device,
        ).view(-1, 1).expand_as(level_one_mask)
        local_graph = graph_grid[level_one_mask]
        local_embeddings = encoded[level_one_mask]

        local_embeddings = self.local_d(local_embeddings)
        global_embeddings = self.global_d(global_embeddings)
        scores = torch.mm(local_embeddings, global_embeddings.t())

        local_index = torch.arange(scores.size(0), device=scores.device)
        positive_scores = scores[local_index, local_graph]
        negative_mask = torch.ones_like(scores, dtype=torch.bool)
        negative_mask[local_index, local_graph] = False
        negative_scores = scores[negative_mask]
        if negative_scores.numel() == 0:
            return scores.new_zeros(())

        log_two = math.log(2.0)
        positive_expectation = (
            log_two - F.softplus(-positive_scores)
        ).mean()
        negative_expectation = (
            F.softplus(-negative_scores) + negative_scores - log_two
        ).mean()
        return negative_expectation - positive_expectation

    def auxiliary_loss(self):
        if self._auxiliary_term is None:
            return self.lin_class.weight.new_zeros(())
        return self._auxiliary_term

    def pretraining_loss(self, data):
        encoded, valid_mask, level_one_mask, lengths = self.encode_sequences(data)
        root_embeddings = encoded[:, 0, :]
        loss = self._jsd_local_global_loss(
            encoded,
            level_one_mask,
            root_embeddings,
        )
        self._last_sequence_lengths = lengths.detach()
        self._last_valid_mask = valid_mask.detach()
        self._last_level_one_mask = level_one_mask.detach()
        self._last_root_embeddings = root_embeddings.detach()
        return loss

    def forward(self, data):
        encoded, valid_mask, level_one_mask, lengths = self.encode_sequences(data)
        root_embeddings = encoded[:, 0, :]
        output = F.log_softmax(self.lin_class(root_embeddings), dim=-1)

        if self.training and self.unsup_weight > 0.0:
            self._auxiliary_term = self.unsup_weight * (
                self._jsd_local_global_loss(
                    encoded,
                    level_one_mask,
                    root_embeddings,
                )
            )
        else:
            self._auxiliary_term = output.new_zeros(())

        self._last_sequence_lengths = lengths.detach()
        self._last_valid_mask = valid_mask.detach()
        self._last_level_one_mask = level_one_mask.detach()
        self._last_root_embeddings = root_embeddings.detach()

        max_hop = int(getattr(self.args, "max_hop", 1))
        placeholder = output.new_zeros(output.size(0), max_hop, 1)
        return output, placeholder, placeholder, placeholder

    def load_pretrained(self, checkpoint_path, map_location=None):
        checkpoint_path = os.fspath(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        if not isinstance(payload, dict):
            raise ValueError(
                "P2T3 checkpoint must contain a PyTorch state dictionary."
            )

        state_dict = {}
        for name, value in payload.items():
            if name.startswith("module."):
                name = name[len("module.") :]
            if name.startswith("lin_class."):
                continue
            state_dict[name] = value
        return self.load_state_dict(state_dict, strict=False)

    def __repr__(self):
        return self.__class__.__name__

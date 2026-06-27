import torch
from torch import nn


class MLPSemanticChangeEncoder(nn.Module):
    """
    Encodes node-level semantic change between two aligned graph views.

    The direction convention is always:

        delta = deny_nodes - support_nodes

    Only the signed change and its absolute magnitude are encoded. Edge or
    node uncertainty is intentionally kept outside this module so uncertainty
    routing/trend modeling can remain an independent, replaceable branch.
    """

    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = (
            self.input_dim if output_dim is None else int(output_dim)
        )
        mlp_hidden_dim = (
            self.input_dim if hidden_dim is None else int(hidden_dim)
        )

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if mlp_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        # Bias-free layers guarantee that identical support/deny views produce
        # an exact zero change representation.
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim * 2, mlp_hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden_dim, self.output_dim, bias=False),
        )

    def change_features(self, support_nodes, deny_nodes):
        self._validate_inputs(support_nodes, deny_nodes)
        delta = deny_nodes - support_nodes
        return torch.cat((delta, delta.abs()), dim=-1)

    def forward(self, support_nodes, deny_nodes):
        features = self.change_features(support_nodes, deny_nodes)
        return self.encoder(features)

    def _validate_inputs(self, support_nodes, deny_nodes):
        if support_nodes.shape != deny_nodes.shape:
            raise ValueError(
                "support_nodes and deny_nodes must have identical shapes, "
                "got {} and {}".format(
                    tuple(support_nodes.shape),
                    tuple(deny_nodes.shape),
                )
            )
        if support_nodes.size(-1) != self.input_dim:
            raise ValueError(
                "expected node feature dimension {}, got {}".format(
                    self.input_dim,
                    support_nodes.size(-1),
                )
            )


def build_semantic_change_encoder(
    encoder_name,
    input_dim,
    output_dim=None,
    hidden_dim=None,
    dropout=0.0,
):
    """
    Factory kept at the model boundary so the MLP can later be replaced
    without changing the caller's forward interface.
    """

    name = str(encoder_name).strip().lower()
    if name == "mlp":
        return MLPSemanticChangeEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(
        "unsupported semantic change encoder: {!r}".format(encoder_name)
    )

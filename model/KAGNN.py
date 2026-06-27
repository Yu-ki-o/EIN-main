import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.nn import global_add_pool, global_mean_pool


class KANLinear(nn.Module):
    """
    Efficient KAN linear layer used by the KAGNN graph-classification models.
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = float(grid_eps)

        h = (grid_range[1] - grid_range[0]) / self.grid_size
        grid = (
            torch.arange(
                -self.spline_order,
                self.grid_size + self.spline_order + 1,
            )
            * h
            + grid_range[0]
        )
        self.register_buffer(
            "grid",
            grid.expand(self.in_features, -1).contiguous(),
        )

        self.base_weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features)
        )
        self.spline_weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                self.grid_size + self.spline_order,
            )
        )
        if self.enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.empty(self.out_features, self.in_features)
            )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(
            self.base_weight,
            a=math.sqrt(5) * self.scale_base,
        )
        with torch.no_grad():
            noise = (
                torch.rand(
                    self.grid_size + 1,
                    self.in_features,
                    self.out_features,
                )
                - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (
                    self.scale_spline
                    if not self.enable_standalone_scale_spline
                    else 1.0
                )
                * self.curve2coeff(
                    self.grid.T[
                        self.spline_order : -self.spline_order
                    ],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline,
                )

    def b_splines(self, x):
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                "KANLinear expected input shape [batch, {}], got {}".format(
                    self.in_features,
                    tuple(x.shape),
                )
            )
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(
            x.dtype
        )
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - self.grid[:, : -(k + 1)])
                / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (self.grid[:, k + 1 :] - x)
                / (self.grid[:, k + 1 :] - self.grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x, y):
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                "curve2coeff expected x shape [batch, {}], got {}".format(
                    self.in_features,
                    tuple(x.shape),
                )
            )
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x):
        splines = self.b_splines(x).view(x.size(0), -1)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            splines,
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
                for in_features, out_features in zip(
                    layers_hidden,
                    layers_hidden[1:],
                )
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SplineLinear(nn.Linear):
    def __init__(self, in_features, out_features, init_scale=0.1):
        self.init_scale = float(init_scale)
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min=-2.0,
        grid_max=2.0,
        num_grids=8,
        denominator=None,
    ):
        super().__init__()
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)
        self.num_grids = int(num_grids)
        grid = torch.linspace(self.grid_min, self.grid_max, self.num_grids)
        self.register_buffer("grid", grid)
        if denominator is None:
            denominator = (self.grid_max - self.grid_min) / (
                self.num_grids - 1
            )
        self.denominator = float(denominator)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_grids=8,
        use_base_update=True,
        use_layernorm=True,
        spline_weight_init_scale=0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.layernorm = (
            nn.LayerNorm(self.input_dim)
            if use_layernorm and self.input_dim > 1
            else None
        )
        self.rbf = RadialBasisFunction(num_grids=num_grids)
        self.spline_linear = SplineLinear(
            self.input_dim * int(num_grids),
            self.output_dim,
            spline_weight_init_scale,
        )
        self.use_base_update = bool(use_base_update)
        if self.use_base_update:
            self.base_linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        if self.layernorm is not None:
            basis_input = self.layernorm(x)
        else:
            basis_input = x
        spline_basis = self.rbf(basis_input)
        out = self.spline_linear(
            spline_basis.view(*spline_basis.shape[:-2], -1)
        )
        if self.use_base_update:
            out = out + self.base_linear(F.silu(x))
        return out


class FastKAN(nn.Module):
    def __init__(self, layers_hidden, num_grids=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FastKANLayer(in_dim, out_dim, num_grids=num_grids)
                for in_dim, out_dim in zip(
                    layers_hidden[:-1],
                    layers_hidden[1:],
                )
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def make_kan(
    num_features,
    hidden_dim,
    out_dim,
    hidden_layers,
    grid_size,
    spline_order,
):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return KAN(
        layers_hidden=sizes,
        grid_size=grid_size,
        spline_order=spline_order,
    )


def make_fastkan(num_features, hidden_dim, out_dim, hidden_layers, grid_size):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return FastKAN(layers_hidden=sizes, num_grids=grid_size)


class KANLayer(KANLinear):
    def __init__(self, input_dim, output_dim, grid_size=4, spline_order=3):
        super().__init__(
            in_features=input_dim,
            out_features=output_dim,
            grid_size=grid_size,
            spline_order=spline_order,
        )


class FastKANConvLayer(FastKANLayer):
    def __init__(self, input_dim, output_dim, grid_size=4):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            num_grids=grid_size,
        )


class KAGCNLayer(GCNConv):
    def __init__(self, in_feat, out_feat, grid_size=4, spline_order=3):
        super().__init__(in_feat, out_feat)
        self.lin = KANLayer(in_feat, out_feat, grid_size, spline_order)


class FastKAGCNLayer(GCNConv):
    def __init__(self, in_feat, out_feat, grid_size=4):
        super().__init__(in_channels=in_feat, out_channels=out_feat)
        self.lin = FastKANConvLayer(in_feat, out_feat, grid_size)


class KAGATLayer(GATConv):
    def __init__(
        self,
        in_feat,
        out_feat,
        heads,
        grid_size=4,
        spline_order=3,
    ):
        super().__init__(in_feat, out_feat, heads)
        self.lin = KANLayer(
            in_feat,
            out_feat * heads,
            grid_size,
            spline_order,
        )


class FastKAGATLayer(GATConv):
    def __init__(self, in_feat, out_feat, heads, grid_size=4):
        super().__init__(in_feat, out_feat, heads)
        self.lin = FastKANConvLayer(in_feat, out_feat * heads, grid_size)


class KAGCN(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        grid_size,
        spline_order,
        dropout,
    ):
        super().__init__()
        self.n_layers = int(gnn_layers)
        layers = [
            KAGCNLayer(num_features, hidden_dim, grid_size, spline_order)
        ]
        for _ in range(self.n_layers - 1):
            layers.append(
                KAGCNLayer(hidden_dim, hidden_dim, grid_size, spline_order)
            )
        self.conv = nn.ModuleList(layers)
        self.readout = make_kan(
            hidden_dim,
            hidden_dim,
            num_classes,
            hidden_layers=1,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = data.x.float()
        for conv in self.conv:
            x = conv(x, data.edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        return self.readout(x)


class FastKAGCN(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        grid_size,
        dropout,
    ):
        super().__init__()
        self.n_layers = int(gnn_layers)
        layers = [FastKAGCNLayer(num_features, hidden_dim, grid_size)]
        for _ in range(self.n_layers - 1):
            layers.append(FastKAGCNLayer(hidden_dim, hidden_dim, grid_size))
        self.conv = nn.ModuleList(layers)
        self.readout = make_fastkan(
            hidden_dim,
            hidden_dim,
            num_classes,
            hidden_layers=1,
            grid_size=grid_size,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = data.x.float()
        for conv in self.conv:
            x = conv(x, data.edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        return self.readout(x)


class KAGAT(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        grid_size,
        spline_order,
        dropout,
        heads,
    ):
        super().__init__()
        self.n_layers = int(gnn_layers)
        self.heads = int(heads)
        layers = [
            KAGATLayer(
                num_features,
                hidden_dim,
                self.heads,
                grid_size,
                spline_order,
            )
        ]
        for _ in range(self.n_layers - 1):
            layers.append(
                KAGATLayer(
                    hidden_dim * self.heads,
                    hidden_dim,
                    self.heads,
                    grid_size,
                    spline_order,
                )
            )
        self.conv = nn.ModuleList(layers)
        self.readout = make_kan(
            hidden_dim * self.heads,
            hidden_dim,
            num_classes,
            hidden_layers=1,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = data.x.float()
        for conv in self.conv:
            x = conv(x, data.edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        return self.readout(x)


class FastKAGAT(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        grid_size,
        dropout,
        heads,
    ):
        super().__init__()
        self.n_layers = int(gnn_layers)
        self.heads = int(heads)
        layers = [
            FastKAGATLayer(
                num_features,
                hidden_dim,
                self.heads,
                grid_size,
            )
        ]
        for _ in range(self.n_layers - 1):
            layers.append(
                FastKAGATLayer(
                    hidden_dim * self.heads,
                    hidden_dim,
                    self.heads,
                    grid_size,
                )
            )
        self.conv = nn.ModuleList(layers)
        self.readout = make_fastkan(
            hidden_dim * self.heads,
            hidden_dim,
            num_classes,
            hidden_layers=1,
            grid_size=grid_size,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        x = data.x.float()
        for conv in self.conv:
            x = conv(x, data.edge_index)
            x = F.silu(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        return self.readout(x)


class KAGIN(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        hidden_layers,
        grid_size,
        spline_order,
        dropout,
    ):
        super().__init__()
        self.n_layers = int(gnn_layers)
        layers = [
            GINConv(
                make_kan(
                    num_features,
                    hidden_dim,
                    hidden_dim,
                    hidden_layers,
                    grid_size,
                    spline_order,
                )
            )
        ]
        for _ in range(self.n_layers - 1):
            layers.append(
                GINConv(
                    make_kan(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        hidden_layers,
                        grid_size,
                        spline_order,
                    )
                )
            )
        self.conv = nn.ModuleList(layers)
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers)]
        )
        self.readout = make_kan(
            hidden_dim,
            hidden_dim,
            num_classes,
            hidden_layers,
            grid_size,
            spline_order,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x.float()
        for conv, batch_norm in zip(self.conv, self.bn):
            x = conv(x, data.edge_index)
            x = batch_norm(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        return self.readout(x)


class FastKAGIN(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        hidden_layers,
        grid_size,
        dropout,
    ):
        super().__init__()
        self.n_layers = int(gnn_layers)
        layers = [
            GINConv(
                make_fastkan(
                    num_features,
                    hidden_dim,
                    hidden_dim,
                    hidden_layers,
                    grid_size,
                )
            )
        ]
        for _ in range(self.n_layers - 1):
            layers.append(
                GINConv(
                    make_fastkan(
                        hidden_dim,
                        hidden_dim,
                        hidden_dim,
                        hidden_layers,
                        grid_size,
                    )
                )
            )
        self.conv = nn.ModuleList(layers)
        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(self.n_layers)]
        )
        self.readout = make_fastkan(
            hidden_dim,
            hidden_dim,
            num_classes,
            hidden_layers,
            grid_size,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x.float()
        for conv, batch_norm in zip(self.conv, self.bn):
            x = conv(x, data.edge_index)
            x = batch_norm(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        return self.readout(x)


class KAGNN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.num_classes = int(num_classes)
        self.max_hop = int(getattr(args, "max_hop", 1))
        self.variant = str(getattr(args, "kagnn_variant", "KAGCN")).upper()
        self.gnn_layers = int(
            getattr(args, "kagnn_num_layers", getattr(args, "n_layers_conv", 2))
        )
        self.hidden_layers = int(getattr(args, "kagnn_hidden_layers", 1))
        self.grid_size = int(getattr(args, "kagnn_grid_size", 4))
        self.spline_order = int(getattr(args, "kagnn_spline_order", 3))
        self.heads = int(getattr(args, "kagnn_heads", 4))
        self.dropout = float(getattr(args, "dropout", 0.0))

        if self.variant == "KAGCN":
            self.backbone = KAGCN(
                self.gnn_layers,
                in_feats,
                hidden_dim,
                num_classes,
                self.grid_size,
                self.spline_order,
                self.dropout,
            )
        elif self.variant in {"FASTKAGCN", "FKAGCN"}:
            self.backbone = FastKAGCN(
                self.gnn_layers,
                in_feats,
                hidden_dim,
                num_classes,
                self.grid_size,
                self.dropout,
            )
        elif self.variant == "KAGAT":
            self.backbone = KAGAT(
                self.gnn_layers,
                in_feats,
                hidden_dim,
                num_classes,
                self.grid_size,
                self.spline_order,
                self.dropout,
                self.heads,
            )
        elif self.variant in {"FASTKAGAT", "FKAGAT"}:
            self.backbone = FastKAGAT(
                self.gnn_layers,
                in_feats,
                hidden_dim,
                num_classes,
                self.grid_size,
                self.dropout,
                self.heads,
            )
        elif self.variant == "KAGIN":
            self.backbone = KAGIN(
                self.gnn_layers,
                in_feats,
                hidden_dim,
                num_classes,
                self.hidden_layers,
                self.grid_size,
                self.spline_order,
                self.dropout,
            )
        elif self.variant in {"FASTKAGIN", "FKAGIN"}:
            self.backbone = FastKAGIN(
                self.gnn_layers,
                in_feats,
                hidden_dim,
                num_classes,
                self.hidden_layers,
                self.grid_size,
                self.dropout,
            )
        else:
            raise ValueError(
                "kagnn_variant must be one of KAGCN, FASTKAGCN, KAGAT, "
                "FASTKAGAT, KAGIN, FASTKAGIN; got {}".format(self.variant)
            )

        class_weights = getattr(args, "classification_class_weights", None)
        if class_weights is None:
            self.register_buffer("classification_class_weights", torch.empty(0))
        else:
            self.register_buffer(
                "classification_class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )

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
        return next(self.parameters()).new_zeros(())

    def _dummy_states(self, data):
        if hasattr(data, "y"):
            batch_size = data.y.view(-1).size(0)
        elif hasattr(data, "batch") and data.batch.numel() > 0:
            batch_size = int(data.batch.max().item()) + 1
        else:
            batch_size = 1
        return next(self.parameters()).new_zeros(
            batch_size,
            self.max_hop,
            1,
        )

    def forward(self, data):
        logits = self.backbone(data)
        out = F.log_softmax(logits, dim=-1)
        dummy = self._dummy_states(data)
        return out, dummy, dummy, dummy

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.variant)

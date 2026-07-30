"""Diagnose the state-conditioning bottleneck in the original EIN ResGCN.

The script provides two complementary checks:

1. A synthetic counterfactual check against ``model/EIN_ResGCN.py``.  It
   changes the per-hop support/deny trajectory while holding the total number
   of replies and the reported maximum depth fixed.
2. An optional audit over preprocessed ``source/*.json`` files.  It measures
   how often samples with the same ``(reply_count, max_depth)`` have different
   state trajectories or different rumor labels.

Examples
--------
Run the code-level counterfactual check:

    python scripts/diagnose_ein_state_bottleneck.py

Also audit a preprocessed dataset:

    python scripts/diagnose_ein_state_bottleneck.py \
        --source-dir dataset/DRWeibo/source
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from torch_geometric.data import Data as PyGData
    from model.EIN_ResGCN import ResGCN as NativeResGCN

    NATIVE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    PyGData = None
    NativeResGCN = None
    NATIVE_IMPORT_ERROR = exc


class _SyntheticDataset:
    def __init__(self, num_features: int):
        self.num_features = int(num_features)


class _FallbackEpidemiologyProbe(torch.nn.Module):
    """Exact probe of the EIN epidemiology data flow, without PyG.

    This is used only when the complete repository model cannot be imported
    because torch_geometric is unavailable.  The source contract is checked
    before this probe runs.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, max_hop: int):
        super().__init__()
        self.max_hop = int(max_hop)
        self.hidden_dim = int(hidden_dim)
        self.W_u0 = torch.nn.Linear(1, hidden_dim)
        self.W_s0 = torch.nn.Linear(1, hidden_dim)
        self.W_d0 = torch.nn.Linear(1, hidden_dim)
        self.W_u = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_s = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_d = torch.nn.Linear(hidden_dim, hidden_dim)
        self.W_x = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.l_u = torch.nn.Linear(hidden_dim, 1)
        self.l_s = torch.nn.Linear(hidden_dim, 1)
        self.l_d = torch.nn.Linear(hidden_dim, 1)
        self.raw_alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.raw_beta = torch.nn.Parameter(torch.tensor(0.5))
        # This small graph projection is not a replacement for ResGCN.  It
        # only demonstrates that a separate data-driven branch may still
        # react when node content changes.
        self.graph_projection = torch.nn.Linear(feature_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    @property
    def alpha(self):
        return torch.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return torch.sigmoid(self.raw_beta)

    def forward(self, data):
        user_state = data.user_state
        n_hop = data.num_hop
        u = torch.sum(user_state, dim=(1, 2))
        s = torch.zeros(user_state.shape[0], device=user_state.device)
        d = torch.zeros(user_state.shape[0], device=user_state.device)

        u_hidden = self.W_u0(u.unsqueeze(1))
        s_hidden = self.W_s0(s.unsqueeze(1))
        d_hidden = self.W_d0(d.unsqueeze(1))
        u_sequence = []
        s_sequence = []
        d_sequence = []
        for _ in range(self.max_hop):
            u_hidden = u_hidden - self.alpha * u_hidden
            u_hidden = u_hidden - self.beta * u_hidden
            u_hidden = self.W_u(u_hidden)
            s_hidden = s_hidden + self.alpha * u_hidden
            s_hidden = self.W_s(s_hidden)
            d_hidden = d_hidden + self.beta * u_hidden
            d_hidden = self.W_d(d_hidden)
            u_sequence.append(u_hidden)
            s_sequence.append(s_hidden)
            d_sequence.append(d_hidden)

        u_all = torch.stack(u_sequence, dim=1)
        s_all = torch.stack(s_sequence, dim=1)
        d_all = torch.stack(d_sequence, dim=1)
        hop_index = n_hop.view(-1).long().clamp(1, self.max_hop) - 1
        hop_index = hop_index.reshape(user_state.shape[0], 1, 1).expand(
            -1,
            -1,
            self.hidden_dim,
        )
        u_final = torch.gather(u_all, 1, hop_index).reshape(
            user_state.shape[0],
            self.hidden_dim,
        )
        s_final = torch.gather(s_all, 1, hop_index).reshape(
            user_state.shape[0],
            self.hidden_dim,
        )
        d_final = torch.gather(d_all, 1, hop_index).reshape(
            user_state.shape[0],
            self.hidden_dim,
        )
        xg = self.W_x(torch.cat((u_final, s_final, d_final), dim=1))

        graph_hidden = self.graph_projection(data.x.mean(dim=0, keepdim=True))
        output = torch.log_softmax(
            self.classifier(graph_hidden + xg),
            dim=-1,
        )
        return (
            output,
            self.l_u(u_all),
            self.l_s(s_all),
            self.l_d(d_all),
        )

    def physics_loss(self, u, s, d, true_state):
        pred_states = torch.stack((u, s, d), dim=2)
        logp_state = torch.log_softmax(pred_states, dim=2)
        state_sum = true_state.sum(dim=-1, keepdim=True)
        true_distribution = true_state / state_sum
        true_distribution[torch.isnan(true_distribution)] = 0
        mask = (state_sum != 0).to(dtype=logp_state.dtype)
        logp_state = (mask.unsqueeze(-2) * logp_state).squeeze(-1)
        kl = torch.nn.functional.kl_div(
            logp_state,
            true_distribution,
            reduction="none",
        )
        return kl.sum(dim=(1, 2), keepdim=True).mean()


def _model_args(hidden_dim: int, max_hop: int) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_dim=int(hidden_dim),
        max_hop=int(max_hop),
        init_alpha=0.5,
        init_beta=0.5,
        lamda=1.0,
    )


def _build_model(
    feature_dim: int = 8,
    hidden_dim: int = 16,
    max_hop: int = 3,
) -> tuple[torch.nn.Module, bool]:
    torch.manual_seed(7)
    native = NativeResGCN is not None
    if native:
        model = NativeResGCN(
            dataset=_SyntheticDataset(feature_dim),
            num_classes=2,
            hidden=hidden_dim,
            num_feat_layers=1,
            num_conv_layers=2,
            num_fc_layers=2,
            residual=True,
            global_pool="mean",
            dropout=0.0,
            edge_norm=True,
            args=_model_args(hidden_dim, max_hop),
            device=torch.device("cpu"),
        )
    else:
        _validate_original_source_contract()
        model = _FallbackEpidemiologyProbe(
            feature_dim,
            hidden_dim,
            max_hop,
        )
    model.eval()
    return model, native


def _validate_original_source_contract() -> None:
    model_path = REPO_ROOT / "model" / "EIN_ResGCN.py"
    source = model_path.read_text(encoding="utf-8")
    required_fragments = {
        "total-count reduction": "u = torch.sum(user_state, dim=(1, 2))",
        "support scalar initialization": "s = torch.zeros(user_state.shape[0])",
        "deny scalar initialization": "d = torch.zeros(user_state.shape[0])",
        "unknown recurrence": "U_ = U_ - self.alpha*U_ - self.beta*U_",
        "support recurrence": "S_ = S_ + self.alpha*U_",
        "deny recurrence": "D_ = D_ + self.beta*U_",
        "depth selection": "hop_ind = n_hop.view(-1).long()",
        "terminal-state concatenation": "xg = torch.cat((U_m, S_m, D_m), dim=1)",
    }
    missing = [
        name
        for name, fragment in required_fragments.items()
        if fragment not in source
    ]
    if missing:
        raise RuntimeError(
            "The original EIN source no longer matches the diagnostic "
            "contract; missing: {}".format(", ".join(missing))
        )


def _make_data(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    user_state: torch.Tensor,
    num_hop: int,
) -> Any:
    fields = {
        "x": x.clone(),
        "edge_index": edge_index.clone(),
        "batch": torch.zeros(x.size(0), dtype=torch.long),
        "user_state": user_state.unsqueeze(0).float(),
        "num_hop": torch.tensor([num_hop], dtype=torch.long),
        "y": torch.tensor([0], dtype=torch.long),
    }
    if PyGData is not None:
        return PyGData(**fields)
    return SimpleNamespace(**fields)


def _forward_with_xg(
    model: torch.nn.Module,
    data: Any,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    captured_z: list[torch.Tensor] = []
    captured_xg: list[torch.Tensor] = []

    def capture_xg(_module, inputs, output):
        if len(inputs) != 1:
            raise RuntimeError(
                "Expected W_x to receive one tensor, got {}".format(
                    len(inputs)
                )
            )
        captured_z.append(inputs[0].detach().clone())
        captured_xg.append(output.detach().clone())

    hook = model.W_x.register_forward_hook(capture_xg)
    try:
        with torch.no_grad():
            output, u, s, d = model(data)
    finally:
        hook.remove()

    if len(captured_z) != 1 or len(captured_xg) != 1:
        raise RuntimeError(
            "Expected exactly one W_x input/output pair, captured {}/{}"
            .format(
                len(captured_z),
                len(captured_xg),
            )
        )
    return output, u, s, d, captured_z[0], captured_xg[0]


def _max_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).abs().max().item())


def run_counterfactual_check(tolerance: float = 1e-7) -> bool:
    """Run direct invariance checks against the repository implementation."""

    model, native = _build_model()
    generator = torch.Generator().manual_seed(11)
    x_a = torch.randn(7, 8, generator=generator)
    x_b = torch.randn(7, 8, generator=generator)

    edge_a = torch.tensor(
        [[0, 0, 1, 1, 3, 3], [1, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )
    edge_b = torch.tensor(
        [[0, 0, 2, 2, 4, 4], [1, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )

    # Both trajectories contain six replies over three observed hops, but
    # their support/deny distributions are deliberately very different.
    support_heavy = torch.tensor(
        [[0, 2, 0], [0, 2, 1], [0, 1, 0]],
        dtype=torch.float32,
    )
    deny_heavy = torch.tensor(
        [[0, 0, 2], [0, 1, 2], [0, 0, 1]],
        dtype=torch.float32,
    )

    data_a = _make_data(x_a, edge_a, support_heavy, num_hop=3)
    state_counterfactual = _make_data(
        x_a,
        edge_a,
        deny_heavy,
        num_hop=3,
    )
    graph_counterfactual = _make_data(
        x_b,
        edge_b,
        deny_heavy,
        num_hop=3,
    )

    out_a, u_a, s_a, d_a, z_a, xg_a = _forward_with_xg(
        model,
        data_a,
    )
    (
        out_state,
        u_state,
        s_state,
        d_state,
        z_state,
        xg_state,
    ) = _forward_with_xg(model, state_counterfactual)
    out_graph, _, _, _, z_graph, xg_graph = _forward_with_xg(
        model,
        graph_counterfactual,
    )

    trajectory_difference = max(
        _max_abs_difference(u_a, u_state),
        _max_abs_difference(s_a, s_state),
        _max_abs_difference(d_a, d_state),
    )
    z_state_difference = _max_abs_difference(z_a, z_state)
    xg_state_difference = _max_abs_difference(xg_a, xg_state)
    output_state_difference = _max_abs_difference(out_a, out_state)
    z_graph_difference = _max_abs_difference(z_a, z_graph)
    xg_graph_difference = _max_abs_difference(xg_a, xg_graph)
    output_graph_difference = _max_abs_difference(out_a, out_graph)

    physics_a = float(
        model.physics_loss(u_a, s_a, d_a, data_a.user_state).item()
    )
    physics_state = float(
        model.physics_loss(
            u_state,
            s_state,
            d_state,
            state_counterfactual.user_state,
        ).item()
    )

    print("=== Original EIN ResGCN: synthetic counterfactual check ===")
    if native:
        print("Execution backend: native model/EIN_ResGCN.py")
    else:
        print(
            "Execution backend: source-validated epidemiology probe "
            "(native import unavailable: {})".format(NATIVE_IMPORT_ERROR)
        )
    print("Both samples: reply_count=6, max_depth=3")
    print(
        "Predicted U/S/D trajectory max |difference|: "
        "{:.10g}".format(trajectory_difference)
    )
    print(
        "Terminal concatenated state Z=[U_H;S_H;D_H] max |difference| "
        "after changing only the state trajectory: {:.10g}".format(
            z_state_difference
        )
    )
    print(
        "Epidemiology embedding x_g max |difference| after changing only "
        "the state trajectory: {:.10g}".format(xg_state_difference)
    )
    print(
        "Full prediction max |difference| after changing only the state "
        "trajectory: {:.10g}".format(output_state_difference)
    )
    print(
        "Terminal concatenated state Z max |difference| after also changing "
        "text/topology: {:.10g}".format(z_graph_difference)
    )
    print(
        "Epidemiology embedding x_g max |difference| after also changing "
        "text/topology: {:.10g}".format(xg_graph_difference)
    )
    print(
        "Full prediction max |difference| after changing text/topology: "
        "{:.10g}".format(output_graph_difference)
    )
    print(
        "Physics losses for the two different supervision targets: "
        "{:.6f} vs {:.6f}".format(physics_a, physics_state)
    )

    invariant = (
        trajectory_difference <= tolerance
        and z_state_difference <= tolerance
        and xg_state_difference <= tolerance
        and output_state_difference <= tolerance
        and z_graph_difference <= tolerance
        and xg_graph_difference <= tolerance
    )
    graph_branch_responds = output_graph_difference > tolerance

    if invariant and graph_branch_responds:
        print(
            "VERDICT: confirmed. For fixed reply count and depth, the "
            "original epidemiology branch is invariant to the event's "
            "per-hop state trajectory. The ordinary GNN branch can still "
            "respond to text/topology changes."
        )
        return True

    print(
        "VERDICT: not confirmed under tolerance {}. Inspect the reported "
        "differences and repository implementation.".format(tolerance)
    )
    return False


def _hop_number(raw_hop: Any) -> int:
    text = str(raw_hop)
    prefix = text.split("-", 1)[0]
    return int(prefix)


def _label_from_post(post: dict[str, Any]) -> Any:
    source = post.get("source", {})
    return source.get("label", post.get("label"))


def _state_rows(post: dict[str, Any]) -> list[tuple[int, int]]:
    raw_state = post.get("state", {})
    rows: list[tuple[int, int, int]] = []
    for raw_hop, counts in raw_state.items():
        rows.append(
            (
                _hop_number(raw_hop),
                int(counts.get("state_0", 0)),
                int(counts.get("state_1", 0)),
            )
        )
    rows.sort(key=lambda item: item[0])
    return [(support, deny) for _, support, deny in rows]


def _ratio_signature(
    rows: Iterable[tuple[int, int]],
    decimals: int,
) -> tuple[tuple[float, float], ...]:
    signature = []
    for support, deny in rows:
        total = support + deny
        if total == 0:
            signature.append((0.0, 0.0))
        else:
            signature.append(
                (
                    round(support / total, decimals),
                    round(deny / total, decimals),
                )
            )
    return tuple(signature)


def audit_source_directory(
    source_dir: Path,
    ratio_decimals: int = 2,
) -> None:
    """Audit real preprocessed JSON files without constructing embeddings."""

    records = []
    skipped = Counter()
    for path in sorted(source_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                post = json.load(handle)
        except (OSError, json.JSONDecodeError):
            skipped["unreadable_json"] += 1
            continue

        rows = _state_rows(post)
        if not rows:
            skipped["missing_state"] += 1
            continue
        label = _label_from_post(post)
        if label is None:
            skipped["missing_label"] += 1
            continue

        reply_count = sum(support + deny for support, deny in rows)
        max_depth = max(
            _hop_number(raw_hop)
            for raw_hop in post.get("state", {})
        )
        records.append(
            {
                "path": path,
                "label": str(label),
                "reply_count": reply_count,
                "max_depth": max_depth,
                "count_signature": tuple(rows),
                "ratio_signature": _ratio_signature(
                    rows,
                    ratio_decimals,
                ),
            }
        )

    if not records:
        print(
            "\n=== Real-data audit ===\n"
            "No usable preprocessed JSON files found in {}".format(source_dir)
        )
        if skipped:
            print("Skipped:", dict(skipped))
        return

    by_size_depth = defaultdict(list)
    by_ratio_trajectory = defaultdict(list)
    for record in records:
        by_size_depth[
            (record["reply_count"], record["max_depth"])
        ].append(record)
        by_ratio_trajectory[
            (
                record["reply_count"],
                record["max_depth"],
                record["ratio_signature"],
            )
        ].append(record)

    collision_groups = [
        group for group in by_size_depth.values() if len(group) >= 2
    ]
    mixed_label_groups = [
        group
        for group in collision_groups
        if len({item["label"] for item in group}) >= 2
    ]
    trajectory_diverse_groups = [
        group
        for group in collision_groups
        if len({item["count_signature"] for item in group}) >= 2
    ]
    mixed_and_diverse_groups = [
        group
        for group in mixed_label_groups
        if len({item["count_signature"] for item in group}) >= 2
    ]
    near_same_ratio_mixed_label_groups = [
        group
        for group in by_ratio_trajectory.values()
        if len(group) >= 2
        and len({item["label"] for item in group}) >= 2
    ]

    def sample_count(groups):
        return len(
            {
                str(item["path"])
                for group in groups
                for item in group
            }
        )

    total = len(records)
    print("\n=== Real-data audit: {} ===".format(source_dir))
    print("Usable samples:", total)
    print(
        "Samples sharing exact (reply_count, max_depth) with another sample: "
        "{} ({:.2%})".format(
            sample_count(collision_groups),
            sample_count(collision_groups) / total,
        )
    )
    print(
        "Mixed-label (reply_count, max_depth) groups: {} groups, {} samples "
        "({:.2%})".format(
            len(mixed_label_groups),
            sample_count(mixed_label_groups),
            sample_count(mixed_label_groups) / total,
        )
    )
    print(
        "State-trajectory-diverse (reply_count, max_depth) groups: {} groups, "
        "{} samples ({:.2%})".format(
            len(trajectory_diverse_groups),
            sample_count(trajectory_diverse_groups),
            sample_count(trajectory_diverse_groups) / total,
        )
    )
    print(
        "Groups that are both mixed-label and trajectory-diverse: {} groups, "
        "{} samples ({:.2%})".format(
            len(mixed_and_diverse_groups),
            sample_count(mixed_and_diverse_groups),
            sample_count(mixed_and_diverse_groups) / total,
        )
    )
    print(
        "Near-identical full ratio trajectories (rounded to {} decimals) "
        "with mixed labels: {} groups, {} samples ({:.2%})".format(
            ratio_decimals,
            len(near_same_ratio_mixed_label_groups),
            sample_count(near_same_ratio_mixed_label_groups),
            sample_count(near_same_ratio_mixed_label_groups) / total,
        )
    )
    if skipped:
        print("Skipped:", dict(skipped))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the original EIN epidemiology encoder's invariance and "
            "optionally audit a preprocessed dataset."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional directory containing preprocessed source/*.json files.",
    )
    parser.add_argument(
        "--ratio-decimals",
        type=int,
        default=2,
        help=(
            "Decimal precision used to define approximately equal per-hop "
            "support/deny ratio trajectories in the real-data audit."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-7,
        help="Numerical tolerance for the synthetic invariance check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    confirmed = run_counterfactual_check(args.tolerance)
    if args.source_dir is not None:
        audit_source_directory(
            args.source_dir,
            ratio_decimals=args.ratio_decimals,
        )
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())

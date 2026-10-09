#!/usr/bin/env python3
"""Small CPU check of GroupGain's staged training and unlabeled inference.

This checks execution on synthetic events or a trusted processed training cache.
It does not select a model, evaluate held-out data, or establish effectiveness.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import time

import torch
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.data.separate import separate


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.GroupGain import GroupGain, compute_gain_targets, freeze_teacher  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--data-cache", type=Path,
        help="Trusted PyG processed training data.pt; tensor storage is memory mapped.",
    )
    inputs.add_argument("--synthetic", action="store_true", help="Use synthetic events (default).")
    parser.add_argument("--max-graphs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, help="Also save the JSON result to this path.")
    args = parser.parse_args()
    if args.max_graphs < 1:
        parser.error("--max-graphs must be positive")
    return args


def synthetic_graphs():
    # Two distinguishable reply groups, an exact duplicate and a deeper branch.
    features = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    edges = torch.tensor([[0, 0, 0, 3], [1, 2, 3, 4]], dtype=torch.long)
    graphs = []
    for index, label in enumerate((0, 1)):
        x = features.clone()
        x[0] += index * 0.2
        graphs.append(Data(
            x=x, reply_edge_index=edges.clone(), root_index=torch.tensor([0]),
            y=torch.tensor([label]), graph_id="synthetic:{}".format(index), split="train",
        ))
    graphs.append(Data(
        x=features[:1].clone(), reply_edge_index=torch.empty((2, 0), dtype=torch.long),
        root_index=torch.tensor([0]), y=torch.tensor([0]),
        graph_id="synthetic:source-only", split="train",
    ))
    return graphs


def load_cached_graphs(path, max_graphs):
    path = path.expanduser().resolve()
    if {"val", "validation", "test"}.intersection(path.parts):
        raise ValueError("Smoke training requires a training cache, not a validation/test cache")
    # PyG Data needs its class during deserialization. Only use trusted caches.
    # mmap avoids copying an entire large cache's tensor storage into RAM.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], Data):
        data, slices = payload
        if slices is None:
            graphs = [data.clone()]
            total = 1
        else:
            total = int(slices["x"].numel()) - 1
            graphs = [
                separate(data.__class__, data, index, slices, decrement=False)
                for index in range(min(max_graphs, total))
            ]
    elif isinstance(payload, list) and all(isinstance(graph, Data) for graph in payload):
        total = len(payload)
        graphs = [graph.clone() for graph in payload[:max_graphs]]
    else:
        raise ValueError("Expected a PyG (Data, slices) training cache or a list of Data")
    if not graphs:
        raise ValueError("The cache contains no event graphs")
    for index, graph in enumerate(graphs):
        existing_split = getattr(graph, "split", None)
        if existing_split is not None and existing_split != "train":
            raise ValueError("The cached event is not marked as training data")
        graph.split = "train"
        if getattr(graph, "graph_id", None) is None:
            graph.graph_id = "{}::{}".format(path, index)
    return graphs, dict(path=str(path), total_cached_graphs=total, selected_cache_indices=list(range(len(graphs))))


def tensor_summary(values):
    values = values.detach().cpu().reshape(-1)
    if not values.numel():
        return dict(count=0, min=None, max=None, mean=None, std=None)
    return dict(
        count=values.numel(), min=float(values.min()), max=float(values.max()),
        mean=float(values.mean()), std=float(values.std(unbiased=False)),
    )


def gradient_sum(parameters):
    return sum(float(parameter.grad.detach().abs().sum())
               for parameter in parameters if parameter.grad is not None)


def run_smoke(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    if args.data_cache is None:
        graphs = synthetic_graphs()[:args.max_graphs]
        source = dict(kind="synthetic", total_available_graphs=3)
    else:
        graphs, source = load_cached_graphs(args.data_cache, args.max_graphs)
        source["kind"] = "processed_training_cache"
    for graph in graphs:
        graph.split = "train"
        if getattr(graph, "y", None) is None or graph.y.numel() != 1:
            raise ValueError("Each training event must have one numeric label")
    labels = [int(graph.y.item()) for graph in graphs]
    if min(labels) < 0:
        raise ValueError("Event labels must be nonnegative class indices")
    feature_dim = int(graphs[0].x.size(1))
    settings = dict(
        layers=2, dropout=0.2, correct_only=False, seed=args.seed,
        candidates_per_graph=2, contexts_per_candidate=1,
        full_context_probability=0.5, micro_batch_size=8,
        gate_temperature=0.2, group_pooling="max", stance_target="generic",
    )
    teacher = GroupGain(feature_dim, 16, max(2, max(labels) + 1),
                        variant="module_a_only", **settings)
    grouping_started = time.perf_counter()
    grouped = teacher.prepare(Batch.from_data_list(graphs))
    grouping_seconds = time.perf_counter() - grouping_started
    optimizer = teacher.init_optimizer()
    teacher_losses = []
    stage_started = time.perf_counter()
    teacher.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = teacher.teacher_training_loss(grouped, mask_probability=0.5)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite teacher loss")
        loss.backward()
        optimizer.step()
        teacher_losses.append(float(loss.detach()))
    teacher_seconds = time.perf_counter() - stage_started
    freeze_teacher(teacher)
    with torch.no_grad():
        _, teacher_details = teacher.forward_graphs(grouped, use_gain=False, return_details=True)
    teacher_gates_are_one = bool(teacher_details["gates"].eq(1).all())
    if not teacher_gates_are_one:
        raise RuntimeError("The reference classifier must have unit gates")

    student = GroupGain(feature_dim, 16, teacher.num_classes, variant="full_model", **settings)
    student.initialize_from_teacher(teacher)
    stage_started = time.perf_counter()
    targets = compute_gain_targets(
        teacher, grouped, candidates_per_graph=2, contexts_per_candidate=1,
        full_context_probability=0.5, correct_only=False, micro_batch_size=8, split="train",
    )
    gain_target_seconds = time.perf_counter() - stage_started

    # The short contribution-head warmup must not change the copied backbone.
    student.requires_grad_(False)
    head_parameters = student.gain_head_parameters()
    for parameter in head_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(head_parameters, lr=5e-4, weight_decay=1e-4)
    gain_head_losses = []
    stage_started = time.perf_counter()
    student.train()
    for _ in range(2):
        student.zero_grad(set_to_none=True)
        loss = student.gain_loss(grouped, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite contribution-head loss")
        if targets.samples:
            loss.backward()
            optimizer.step()
        gain_head_losses.append(float(loss.detach()))
    gain_head_seconds = time.perf_counter() - stage_started
    backbone_unchanged = all(
        torch.equal(parameter, dict(getattr(teacher, name).named_parameters())[key])
        for name in ("node_encoder", "layers", "classifier")
        for key, parameter in getattr(student, name).named_parameters()
    )
    if not backbone_unchanged:
        raise RuntimeError("Contribution warmup modified the frozen backbone")

    student.requires_grad_(True)
    student.zero_grad(set_to_none=True)
    optimizer = student.init_optimizer()
    joint_losses, joint_gradients = [], []
    stage_started = time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = student.compute_loss(grouped, teacher=teacher, gain_targets=targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite joint loss")
        loss.backward()
        joint_gradients.append(dict(
            classifier=gradient_sum(student.classifier.parameters()),
            gain_head=gradient_sum(student.gain_head.parameters()),
            context_encoder=gradient_sum(student.context_encoder.parameters()),
        ))
        optimizer.step()
        joint_losses.append(float(loss.detach()))
    joint_seconds = time.perf_counter() - stage_started
    teacher_has_no_gradients = all(parameter.grad is None for parameter in teacher.parameters())
    if not teacher_has_no_gradients:
        raise RuntimeError("The frozen reference teacher received gradients")
    if targets.samples and any(not any(step[name] > 0 for step in joint_gradients)
                               for name in ("classifier", "gain_head", "context_encoder")):
        raise RuntimeError("Joint training did not reach every classifier/contribution branch")

    student.eval()
    with torch.no_grad():
        expected, details = student.forward_graphs(grouped, return_details=True)
    gains = targets.values.detach().cpu().clone()
    gain_diagnostics = dict(targets.diagnostics)
    del teacher, targets  # Restored inference has neither of these utilities.
    unlabeled = []
    for graph in graphs:
        event = graph.clone()
        event.y = None
        unlabeled.append(event)
    with tempfile.TemporaryDirectory(prefix="group_gain_smoke_") as directory:
        checkpoint_path = Path(directory) / "student.pt"
        student.save_checkpoint(checkpoint_path, metadata={
            "scope": "smoke execution only", "data_kind": source["kind"], "label_mapping": None,
        })
        restored = GroupGain.load_checkpoint(checkpoint_path, map_location="cpu").eval()
        with torch.no_grad():
            inferred = restored(Batch.from_data_list(unlabeled))
        inference_matches = bool(torch.allclose(expected, inferred, atol=1e-6, rtol=1e-5))
        independent_checkpoint = not any("teacher" in key for key in restored.state_dict())
    if not inference_matches or not independent_checkpoint:
        raise RuntimeError("Checkpoint/unlabeled inference verification failed")
    root_only_indices = [index for index, graph in enumerate(grouped) if graph.num_groups == 1]
    root_only_readout_zero = all(bool(details["representation"][index, -16:].eq(0).all())
                                 for index in root_only_indices)
    root_gates_are_one = bool(details["gates"][details["root_mask"]].eq(1).all())
    if not root_only_readout_zero or not root_gates_are_one:
        raise RuntimeError("Source-only readout or source-gate verification failed")
    return dict(
        status="passed", scope="staged training smoke; no held-out performance evaluation",
        source=source, device="cpu", seed=args.seed,
        versions={"python": sys.version.split()[0], "torch": torch.__version__,
                  "torch_geometric": torch_geometric.__version__},
        settings=dict(settings, hidden_dim=16, teacher_steps=2, gain_head_steps=2 if gains.numel() else 0,
                      joint_steps=2, teacher_mask_probability=0.5, lambda_gain=1.0, lambda_rep=0.0),
        data=dict(graph_count=len(graphs), feature_dim=feature_dim, labels=labels,
                  label_counts=dict(Counter(labels)), label_mapping=None, split="train",
                  original_nodes=[graph.num_nodes for graph in grouped]),
        grouping=[dict(graph_id=graph.graph_id, **graph.diagnostics) for graph in grouped],
        losses=dict(teacher=teacher_losses, gain_head=gain_head_losses, joint=joint_losses),
        gain_targets=dict(values=gains.tolist(), summary=tensor_summary(gains), diagnostics=gain_diagnostics),
        reply_gates=tensor_summary(details["gates"][~details["root_mask"]]),
        joint_gradient_abs_sums=joint_gradients,
        verification=dict(
            teacher_gates_are_one=teacher_gates_are_one,
            teacher_has_no_gradients=teacher_has_no_gradients,
            gain_warmup_preserved_backbone=backbone_unchanged,
            unlabeled_restored_inference_allclose=inference_matches,
            inference_max_abs_difference=float((expected - inferred).abs().max()),
            checkpoint_has_no_teacher_parameters=independent_checkpoint,
            root_gates_are_one=root_gates_are_one,
            source_only_event_indices=root_only_indices,
            source_only_readout_zero=root_only_readout_zero,
        ),
        seconds=dict(grouping=grouping_seconds, teacher=teacher_seconds,
                     gain_targets=gain_target_seconds, gain_head=gain_head_seconds,
                     joint=joint_seconds, total=time.perf_counter() - started),
        gpu_peak_memory_bytes=None,
    )


def main():
    args = parse_args()
    report = run_smoke(args)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report is not None:
        path = args.report.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

"""Model properties for grouped conditional-gain rumor classification.

These synthetic checks exercise the restricted exact-copy property and gain
isolation. They are not an evaluation on a real rumor dataset.
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.GroupGain import EINGroupGain, GroupGain, compute_gain_targets, freeze_teacher


def make_model(**options):
    torch.manual_seed(29)
    values = dict(layers=2, dropout=0.0)
    values.update(options)
    return GroupGain(4, 8, 2, **values)


def graph(x, edges, root=0, label=0, graph_id="synthetic"):
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        reply_edge_index=torch.tensor(edges, dtype=torch.long).reshape(2, -1),
        root_index=torch.tensor([root]),
        graph_id=graph_id,
    )
    if label is not None:
        data.y = torch.tensor([label])
    return data


def multigroup_graph(label=0):
    # The root deliberately is not node 0; node 0 is a reproducible leaf.
    return graph(
        [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
         [0, 0, 0, 1], [0, -1, 0, 0]],
        [[2, 2, 2, 3], [0, 1, 3, 4]], root=2, label=label,
    )


def batch(graphs):
    return Batch.from_data_list(graphs)


def graph_structure(grouped):
    """Compare groups by their feature content, ignoring IDs and multiplicity."""
    keys = []
    for members in grouped.group_members:
        keys.append(tuple(sorted(set(tuple(grouped.x[node].tolist())
                                     for node in members))))
    edges = sorted((keys[source], keys[target], int(relation))
                   for (source, target), relation in zip(
                       grouped.edge_index.t().tolist(), grouped.edge_type.tolist()))
    return keys[0], sorted(keys[1:]), edges


def test_exact_leaf_copy_is_invariant_for_multiple_nonroot_groups():
    model = make_model().eval()
    original = multigroup_graph()
    original_graph = model.prepare(original)[0]
    before, details = model.forward_graphs([original_graph], return_details=True)
    assert original_graph.num_groups > 2
    assert details["u"].size(0) > 2
    for copies in (1, 5, 10, 50):
        repeated = deepcopy(original)
        repeated.x = torch.cat([original.x, original.x[0:1].repeat(copies, 1)])
        new_nodes = torch.arange(original.x.size(0), repeated.x.size(0))
        extra_edges = torch.stack([new_nodes.new_full((copies,), 2), new_nodes])
        repeated.reply_edge_index = torch.cat([original.reply_edge_index, extra_edges], dim=1)
        grouped = model.prepare(repeated)[0]
        assert grouped.num_groups == original_graph.num_groups
        assert graph_structure(grouped) == graph_structure(original_graph)
        after, duplicate_details = model.forward_graphs([grouped], return_details=True)
        torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
        # Canonical group ordering makes each pooled representation comparable.
        torch.testing.assert_close(duplicate_details["u"], details["u"],
                                   atol=1e-6, rtol=1e-5)


def test_node_permutation_preserves_group_graph_and_eval_logits():
    model = make_model().eval()
    original = multigroup_graph()
    permutation = torch.tensor([4, 1, 3, 0, 2])
    inverse = torch.argsort(permutation)
    relabeled = deepcopy(original)
    relabeled.x = original.x[permutation]
    relabeled.reply_edge_index = inverse[original.reply_edge_index]
    relabeled.root_index = inverse[original.root_index]
    first = model.prepare(original)[0]
    second = model.prepare(relabeled)[0]
    assert graph_structure(first) == graph_structure(second)
    torch.testing.assert_close(model(original), model(relabeled), atol=1e-6, rtol=1e-5)


def test_masked_groups_are_removed_before_message_passing():
    model = make_model().eval()
    grouped = model.prepare(multigroup_graph())[0]
    keep = torch.zeros(grouped.num_groups, dtype=torch.bool)
    keep[0] = True
    keep[grouped.node_to_group[3]] = True
    before, details = model.forward_graphs([grouped], [keep], return_details=True)
    assert details["u"].size(0) == 2
    assert details["z"].size(0) == 2
    assert details["batch"].numel() == 2
    altered = deepcopy(grouped)
    deleted = ~keep[grouped.node_to_group]
    altered.x[deleted] = torch.tensor([1000.0, -2000.0, 3000.0, -4000.0])
    after = model.forward_graphs([altered], [keep])
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)
    # An induced subgraph can retain a disconnected leaf and still encode it.
    keep.zero_()
    keep[0] = True
    keep[grouped.node_to_group[4]] = True
    disconnected = model.forward_graphs([grouped], [keep])
    assert disconnected.shape == (1, 2)
    assert torch.isfinite(disconnected).all()


def test_batch_matches_separate_graphs_and_inference_does_not_read_labels():
    model = make_model().eval()
    graphs = [multigroup_graph(),
              graph([[1, 0, 0, 0]], [[], []], label=1, graph_id="root"),
              graph([[1, 0, 0, 0], [0, 1, 0, 0]], [[0], [1]],
                    graph_id="single")]
    together = model(batch(graphs))
    separate = torch.cat([model(item) for item in graphs])
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    for item in graphs:
        before = model(item)
        item.y = 1 - item.y
        torch.testing.assert_close(before, model(item))
        del item.y
        torch.testing.assert_close(before, model(item))


def test_root_only_single_reply_and_zero_feature_graphs_are_finite():
    full = make_model().eval()
    only_root = graph([[1, 0, 0, 0]], [[], []])
    single = graph([[1, 0, 0, 0], [0, 1, 0, 0]], [[0], [1]])
    zeros = graph([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0], [1, 2]])
    for item in (only_root, single, zeros):
        logits, details = full.forward_graphs(full.prepare(item), return_details=True)
        assert logits.shape == (1, 2)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(details["gates"]).all()
        assert details["gates"][details["root_mask"]].eq(1).all()
    assert full.prepare(zeros)[0].num_groups == 3
    root_model = make_model(variant="root_only").eval()
    torch.testing.assert_close(root_model(multigroup_graph()), root_model(only_root))


class ControlledTeacher(GroupGain):
    """A frozen reference with a known signed improvement for its one reply."""

    def __init__(self, sign):
        super().__init__(4, 8, 2, layers=1, dropout=0.0, variant="module_a_only")
        self.sign = sign

    def forward_graphs(self, graphs, keep_masks=None, use_gain=None, return_details=False):
        values = []
        for index, item in enumerate(graphs):
            keep = (torch.ones(item.num_groups, dtype=torch.bool, device=item.x.device)
                    if keep_masks is None else keep_masks[index])
            count = keep[1:].sum().to(dtype=item.x.dtype)
            values.append(torch.stack([count * self.sign, count * 0]))
        return torch.stack(values)


def test_gain_targets_preserve_sign_and_equal_two_fresh_reference_losses():
    data = graph([[1, 0, 0, 0], [0, 1, 0, 0]], [[0], [1]])
    grouped = make_model().prepare(data)
    for sign in (-2.0, 2.0):
        teacher = freeze_teacher(ControlledTeacher(sign))
        targets = compute_gain_targets(teacher, grouped, candidates_per_graph=2,
                                       contexts_per_candidate=1, correct_only=False)
        assert targets.samples == [(0, 1, ())]
        assert targets.values.numel() == 1
        empty_logits = teacher.forward_graphs(grouped, [torch.tensor([True, False])])
        full_logits = teacher.forward_graphs(grouped, [torch.tensor([True, True])])
        expected = F.cross_entropy(empty_logits, data.y, reduction="none") - F.cross_entropy(
            full_logits, data.y, reduction="none")
        torch.testing.assert_close(targets.values, expected)
        assert targets.values.item() * sign > 0
        assert not targets.values.requires_grad
        assert all(parameter.grad is None for parameter in teacher.parameters())


def test_gain_targets_require_an_explicitly_frozen_eval_teacher():
    teacher = make_model(variant="module_a_only")
    grouped = teacher.prepare(multigroup_graph())
    with unittest.TestCase().assertRaises(ValueError):
        compute_gain_targets(teacher, grouped, correct_only=False)


def test_joint_training_has_classifier_gain_and_context_gradients_with_frozen_teacher():
    student = make_model().train()
    teacher = freeze_teacher(make_model(variant="module_a_only"))
    data = batch([multigroup_graph(0), multigroup_graph(1)])
    grouped = student.prepare(data)
    targets = compute_gain_targets(teacher, grouped, candidates_per_graph=2,
                                   contexts_per_candidate=1, correct_only=False,
                                   full_context_probability=1.0)
    loss = student.compute_loss(data, teacher=teacher, gain_targets=targets)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("classifier", "gain_head", "context_encoder"):
        parameters = list(getattr(student, name).parameters())
        gradients = [parameter.grad for parameter in parameters]
        assert gradients and all(item is not None and torch.isfinite(item).all()
                                 for item in gradients), name
        assert sum(item.abs().sum().item() for item in gradients) > 0, name
    assert all(parameter.grad is None and not parameter.requires_grad
               for parameter in teacher.parameters())


def test_gain_predictions_use_context_and_features_without_reading_labels():
    model = make_model().eval()
    grouped = model.prepare(multigroup_graph())
    contexts = [(0, 1, ()), (0, 1, tuple(range(2, grouped[0].num_groups)))]
    before = model.gain_predictions(grouped, contexts)
    assert before.shape == (2,)
    assert torch.isfinite(before).all()
    grouped[0].y = torch.tensor([1])
    torch.testing.assert_close(before, model.gain_predictions(grouped, contexts))
    grouped[0].y = None
    torch.testing.assert_close(before, model.gain_predictions(grouped, contexts))


def test_module_a_uses_unit_gates_and_full_model_gates_change_encoding():
    model = make_model().eval()
    grouped = model.prepare(multigroup_graph())
    ungated, plain = model.forward_graphs(grouped, use_gain=False, return_details=True)
    gated, details = model.forward_graphs(grouped, use_gain=True, return_details=True)
    assert plain["gates"].eq(1).all()
    assert details["gates"][details["root_mask"]].eq(1).all()
    assert (details["gates"][~details["root_mask"]] < 1).all()
    assert not torch.allclose(details["z"], plain["z"])
    assert not torch.allclose(gated, ungated)
    module_a = make_model(variant="module_a_only").eval()
    _, details = module_a.forward_graphs(module_a.prepare(multigroup_graph()),
                                        return_details=True)
    assert details["gates"].eq(1).all()


def test_message_only_and_readout_only_switches_control_separate_operations():
    message = make_model(gate_mode="message_only").eval()
    readout = make_model(gate_mode="readout_only").eval()
    both = make_model(gate_mode="both").eval()
    graphs = both.prepare(multigroup_graph())
    ungated, plain = both.forward_graphs(graphs, use_gain=False, return_details=True)
    message_logits, message_details = message.forward_graphs(graphs, return_details=True)
    readout_logits, readout_details = readout.forward_graphs(graphs, return_details=True)
    both_logits, both_details = both.forward_graphs(graphs, return_details=True)
    torch.testing.assert_close(readout_details["z"], plain["z"])
    torch.testing.assert_close(message_details["z"], both_details["z"])
    assert not torch.allclose(message_details["z"], plain["z"])
    assert not torch.allclose(readout_logits, ungated)
    assert not torch.allclose(message_logits, both_logits)


def test_no_reply_skips_gain_targets_without_breaking_classification_loss():
    data = graph([[1, 0, 0, 0]], [[], []])
    student = make_model().train()
    teacher = freeze_teacher(make_model(variant="module_a_only"))
    graphs = student.prepare(data)
    targets = compute_gain_targets(teacher, graphs, correct_only=False)
    assert targets.samples == []
    assert targets.values.numel() == 0
    assert student.gain_loss(graphs, targets).item() == 0
    loss = student.compute_loss(data, teacher=teacher, gain_targets=targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in student.classifier.parameters())


def test_checkpoint_is_independent_of_teacher_and_preserves_unlabeled_inference():
    model = make_model().eval()
    data = multigroup_graph()
    del data.y
    before = model(data)
    with tempfile.TemporaryDirectory() as directory:
        path = directory + "/group_gain.pt"
        model.save_checkpoint(path, metadata={"seed": 29, "label_mapping": {"rumor": 0}})
        restored = GroupGain.load_checkpoint(path, map_location="cpu").eval()
        torch.testing.assert_close(before, restored(data))
        assert not any("teacher" in key for key in restored.state_dict())


def test_reference_training_reencodes_full_and_masked_graphs_without_gain_gradients():
    teacher = make_model(variant="module_a_only").train()
    data = batch([multigroup_graph(0), multigroup_graph(1)])
    with patch.object(teacher, "forward_graphs", wraps=teacher.forward_graphs) as forward:
        loss = teacher.teacher_training_loss(data, mask_probability=1.0)
        assert forward.call_count == 2
        assert all(call.kwargs["use_gain"] is False for call in forward.call_args_list)
        masks = forward.call_args_list[1].kwargs["keep_masks"]
        assert all(mask[0] and mask.sum().item() == 1 for mask in masks)
    graphs = teacher.prepare(data)
    full = teacher.forward_graphs(graphs, use_gain=False)
    masked = teacher.forward_graphs(graphs, keep_masks=masks, use_gain=False)
    expected = 0.5 * (F.cross_entropy(full, data.y) + F.cross_entropy(masked, data.y))
    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("node_encoder", "layers", "classifier"):
        # A generic-only graph does not activate other relation transforms.
        gradients = [parameter.grad for parameter in getattr(teacher, name).parameters()
                     if parameter.grad is not None]
        assert gradients and all(torch.isfinite(item).all() for item in gradients)
        assert sum(item.abs().sum().item() for item in gradients) > 0
    assert all(parameter.grad is None for parameter in teacher.gain_head.parameters())
    assert all(parameter.grad is None for parameter in teacher.context_encoder.parameters())


def test_ein_wrapper_returns_logits_and_cross_entropy_with_unlabeled_inference():
    torch.manual_seed(29)
    model = EINGroupGain(4, 8, 2, args=SimpleNamespace(max_hop=3),
                        layers=2, dropout=0.0).eval()
    graphs = [multigroup_graph(0),
              graph([[1, 0, 0, 0]], [[], []], label=1, graph_id="root")]
    data = batch(graphs)
    output = model(data)
    assert isinstance(output, tuple) and len(output) == 4
    logits, first, second, third = output
    assert logits.shape == (2, 2)
    assert first.shape == second.shape == third.shape == (2, 3, 1)
    assert torch.isfinite(logits).all()
    assert first.eq(0).all() and second.eq(0).all() and third.eq(0).all()
    torch.testing.assert_close(logits, model.forward_graphs(model.prepare(data)))
    torch.testing.assert_close(model.classification_loss(logits, data.y),
                               F.cross_entropy(logits, data.y))
    for item in graphs:
        del item.y
    torch.testing.assert_close(logits, model(batch(graphs))[0])


def test_gain_prediction_matches_full_context_and_ignores_groups_outside_context():
    model = make_model().eval()
    grouped = model.prepare(multigroup_graph())
    count = grouped[0].num_groups
    contexts = [(0, candidate, tuple(other for other in range(1, count)
                                    if other != candidate))
                for candidate in range(1, count)]
    explicit = model.gain_predictions(grouped, contexts)
    _, details = model.forward_graphs(grouped, return_details=True)
    torch.testing.assert_close(explicit, details["predicted_gains"][~details["root_mask"]],
                               atol=1e-6, rtol=1e-5)
    # Group 3 is absent from this context and cannot alter either its candidate
    # encoding or the independently encoded context representation.
    partial_context = [(0, 1, (2,))]
    before = model.gain_predictions(grouped, partial_context)
    altered = deepcopy(grouped)
    outside_context = altered[0].node_to_group == 3
    assert outside_context.any()
    altered[0].x[outside_context] = torch.tensor([1000.0, -2000.0, 3000.0, -4000.0])
    torch.testing.assert_close(before, model.gain_predictions(altered, partial_context))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
                              for name, function in sorted(globals().items())
                              if name.startswith("test_"))

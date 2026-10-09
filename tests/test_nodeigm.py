from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.NodeIGM import NodeIGM


def make_model(**options):
    torch.manual_seed(17)
    values = dict(n_layers_conv=2, dropout=0.0, global_pool="mean", max_hop=3,
                  nodeigm_env_weight=0.1, nodeigm_keep_weight=0.01)
    values.update(options)
    return NodeIGM(4, 8, 2, SimpleNamespace(**values))


def graph(edges, num_nodes, label=0):
    return Data(x=torch.randn(num_nodes, 4),
                edge_index=torch.tensor(edges, dtype=torch.long).reshape(2, -1),
                y=torch.tensor([label]))


def keep_only(pair):
    def scores(x, edge):
        keep = (edge[0] == pair[0]) & (edge[1] == pair[1])
        return torch.where(keep, x.new_tensor(0.9), x.new_tensor(0.1))
    return scores


def _check_deleted_leaf_and_self_loop_cannot_contribute_to_pooling(pool):
    model = make_model(global_pool=pool).eval()
    data = graph([[0, 0, 2], [1, 2, 2]], 3)
    with patch.object(model, "_edge_scores", side_effect=keep_only((0, 1))):
        before = model(data)[0]
        evidence = model.extract_evidence_subgraph(data)
        assert evidence.pool_mask.tolist() == [True, True, False]
        assert evidence.edge_index.tolist() == [[0, 1], [1, 0]]
        assert evidence.original_node_id.tolist() == [0, 1, 2]

        # Compare to encoding the physically pruned edges and pooling only
        # their endpoints. Mean readout must divide by two, not three.
        hidden = model._encode(data.x, torch.tensor([[0], [1]]), torch.ones(1))[-1]
        expected_repr = hidden[:2].sum(0, keepdim=True)
        if pool == "mean":
            expected_repr = expected_repr / 2
        expected = F.log_softmax(model.classifier(expected_repr), dim=-1)
        torch.testing.assert_close(before, expected)

        data.x[2] = torch.tensor([1000.0, -2000.0, 3000.0, -4000.0])
        torch.testing.assert_close(before, model(data)[0])


def test_deleted_leaf_cannot_contribute_to_mean_pooling():
    _check_deleted_leaf_and_self_loop_cannot_contribute_to_pooling("mean")


def test_deleted_leaf_cannot_contribute_to_sum_pooling():
    _check_deleted_leaf_and_self_loop_cannot_contribute_to_pooling("sum")


def test_isolated_source_is_excluded_but_nonisolated_disconnected_component_remains():
    model = make_model().eval()
    data = graph([[0, 1, 2], [1, 2, 3]], 4)
    with patch.object(model, "_edge_scores", side_effect=keep_only((1, 2))):
        output = model(data)[0]
        assert model.get_diagnostics()["pool_mask"].tolist() == [False, True, True, False]
        data.x[[0, 3]] += 1000
        torch.testing.assert_close(output, model(data)[0])


def test_empty_evidence_graphs_keep_batch_alignment_and_have_zero_readout():
    model = make_model().eval()
    graphs = [graph([[0], [1]], 2), graph([[], []], 1),
              graph([[0, 1], [0, 1]], 2)]
    data = Batch.from_data_list(graphs)
    with patch.object(model, "_edge_scores", side_effect=lambda x, edge: x.new_zeros(edge.size(1))):
        output, u, s, d = model(data)
        assert output.shape == (3, 2)
        assert u.shape == s.shape == d.shape == (3, 3, 1)
        assert not model.get_diagnostics()["pool_mask"].any()
        assert torch.equal(model.get_diagnostics()["graph_representation"], torch.zeros(3, 8))
        expected = F.log_softmax(model.classifier.bias, dim=0).expand(3, -1)
        torch.testing.assert_close(output, expected)
        assert torch.isfinite(output).all()


def test_discriminator_gets_classification_gradient_even_if_all_edges_are_deleted():
    model = make_model(nodeigm_env_weight=0.0, nodeigm_keep_weight=0.0).train()
    with torch.no_grad():
        model.edge_discriminator[-2].bias.fill_(-5)
    data = Batch.from_data_list([graph([[0, 0], [1, 2]], 3),
                                graph([[0, 1], [1, 2]], 3, 1)])
    loss = model.compute_loss(data)
    assert not model.get_diagnostics()["pool_mask"].any()
    assert torch.equal(model.get_diagnostics()["graph_representation"], torch.zeros(2, 8))
    loss.backward()
    grads = [parameter.grad for parameter in model.edge_discriminator.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert sum(grad.abs().sum().item() for grad in grads) > 0


def test_training_losses_are_finite_and_optimizer_updates_selector():
    model = make_model(dropout=0.2).train()
    data = Batch.from_data_list([graph([[0, 0, 1], [1, 2, 3]], 4),
                                graph([[0, 1], [1, 2]], 3, 1),
                                graph([[], []], 1)])
    before = model.edge_discriminator[-2].weight.detach().clone()
    optimizer = model.init_optimizer()
    model.set_epoch(1)
    loss = model.compute_loss(data)
    assert torch.isfinite(loss)
    assert torch.isfinite(model.auxiliary_loss())
    assert model.physics_loss(None, None, None, None).item() == 0
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.edge_discriminator[-2].weight)


def test_shared_dropout_does_not_create_environment_loss_when_views_are_identical():
    model = make_model(dropout=0.5).train()
    data = graph([[0, 0], [1, 2]], 3)
    with patch.object(model, "_edge_scores", side_effect=lambda x, edge: x.new_full((edge.size(1),), 0.9)):
        model(data)
    assert model.get_diagnostics()["environment_loss"].item() == 0


def test_environment_restoration_budget_is_per_event():
    model = make_model()
    data = Batch.from_data_list([
        graph([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], 6),
        graph([[0, 0, 0], [1, 2, 3]], 4),
    ])
    _, edge, batch, count = model._inputs(data)
    causal = torch.zeros(edge.size(1), dtype=torch.bool)
    for graph_id in range(count):
        first = (batch[edge[0]] == graph_id).nonzero().flatten()[0]
        causal[first] = True
    restored = model._restore_edges(edge, causal, batch, count, 0.5)
    assert not (restored & causal).any()
    assert torch.bincount(batch[edge[0, restored]], minlength=count).tolist() == [2, 1]


def test_eval_batch_matches_separate_graphs_and_does_not_need_labels():
    model = make_model().eval()
    graphs = [graph([[0, 0, 1], [1, 2, 3]], 4), graph([[], []], 1),
              graph([[0], [1]], 2)]
    together = model(Batch.from_data_list(graphs))[0]
    separate = torch.cat([model(item)[0] for item in graphs])
    torch.testing.assert_close(together, separate)
    for item in graphs:
        before = model(item)[0]
        del item.y
        torch.testing.assert_close(before, model(item)[0])
        model.extract_evidence_subgraph(item)
    assert model.auxiliary_loss().item() == 0


def test_reversed_duplicate_edges_share_one_gate_and_node_relabeling_is_invariant():
    model = make_model().eval()
    data = graph([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 2, 1]], 3)
    output = model(data)[0]
    evidence = model.extract_evidence_subgraph(data)
    assert evidence.candidate_edge_index.size(1) == 2
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.argsort(permutation)
    relabeled = Data(x=data.x[permutation], edge_index=inverse[data.edge_index])
    torch.testing.assert_close(output, model(relabeled)[0])


def test_cross_event_edges_are_rejected():
    model = make_model()
    data = Batch.from_data_list([graph([[0], [1]], 2), graph([[0], [1]], 2)])
    data.edge_index[:, 0] = torch.tensor([0, 2])
    with unittest.TestCase().assertRaisesRegex(ValueError, "different event graphs"):
        model(data)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
                              for name, function in sorted(globals().items())
                              if name.startswith("test_"))

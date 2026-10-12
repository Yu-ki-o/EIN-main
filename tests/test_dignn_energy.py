"""Numerical and behavioral checks for anchored stance energy descriptors."""

from types import SimpleNamespace
import unittest

import torch
from torch_geometric.data import Batch, Data

from model.DIGNN import DIGNN, StanceEnergyField


def graph(nodes, edges, stance=None, directed=True):
    edges = torch.tensor(edges, dtype=torch.long).reshape(2, -1)
    data = Data(x=torch.randn(nodes, 4), edge_index=edges,
                root_index=torch.tensor([0]), y=torch.tensor([0]))
    if directed:
        data.directed_edge_index = edges.clone()
    if stance is not None:
        labels = torch.tensor(stance, dtype=torch.long)
        data.edge_stance = labels.clone()
        if directed:
            data.directed_edge_stance = labels.clone()
    return data


def model(**options):
    settings = dict(dropout=0.0, dignn_num_layers=2, max_hop=3)
    settings.update(options)
    return DIGNN(4, 8, 2, SimpleNamespace(**settings))


class StanceEnergyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(21)

    def test_identity_maps_converge_to_dense_dirichlet_solution(self):
        edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 3, 4, 5, 6]])
        source = torch.tensor([[2., 0.], [2., 0.], [0., 2.]] + [[0., 0.]] * 4,
                              dtype=torch.float64)
        field = StanceEnergyField(scales=(1.,), num_steps=400).double()
        result = field(source, edges, torch.tensor([0, 0, 0, 0, 1, 1]), torch.zeros(7, dtype=torch.long))
        incidence = source.new_zeros(6, 7)
        incidence[torch.arange(6), edges[0]] = 1
        incidence[torch.arange(6), edges[1]] = -1
        expected = torch.linalg.solve(torch.eye(7, dtype=source.dtype) + incidence.T @ incidence, source)
        torch.testing.assert_close(result["fields"][:, 0], expected, atol=1e-10, rtol=1e-10)
        root_energy = result["relation_energy"][0].sum()
        self.assertAlmostEqual(float(root_energy.detach()), 94 / 441, places=10)
        self.assertAlmostEqual(float(result["final_energy"].detach()), 64 / 21, places=10)

    def test_field_value_and_local_energy_retain_distinct_information(self):
        edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 3, 4, 5, 6]])
        source_a = torch.tensor([[2., 0.], [2., 0.], [0., 2.]] + [[0., 0.]] * 4)
        source_b = torch.tensor([[2., 0.], [1., 1.], [1., 1.]] + [[0., 0.]] * 4)
        solver = StanceEnergyField(scales=(1.,), num_steps=200)
        batch = torch.zeros(7, dtype=torch.long)
        first = solver(source_a, edges, torch.tensor([0, 0, 0, 0, 1, 1]), batch)
        second = solver(source_b, edges, torch.tensor([0, 0, 0, 1, 0, 1]), batch)
        torch.testing.assert_close(first["fields"][0], second["fields"][0])
        self.assertGreater(float(first["relation_energy"][0].sum().detach()),
                           float(second["relation_energy"][0].sum().detach()))

    def test_safe_steps_decrease_learned_energy_for_every_scale_and_event(self):
        edges = torch.tensor([[0, 0, 1, 4, 5], [1, 2, 3, 5, 6]])
        labels = torch.tensor([0, 1, -1, 1, 0])
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        source = torch.rand(7, 2) * 4
        solver = StanceEnergyField(num_steps=1)
        with torch.no_grad():
            solver.relation_log_scales.copy_(torch.randn_like(solver.relation_log_scales) * 2)
        previous = solver(source, edges, labels, batch)["initial_energy"]
        for steps in range(1, 10):
            solver.num_steps = steps
            result = solver(source, edges, labels, batch)
            self.assertTrue(bool((result["final_energy"] <= previous + 2e-6).all()))
            self.assertTrue(bool((result["fields"] >= 0).all()))
            previous = result["final_energy"]
        result["fields"].square().sum().backward()
        gradient = solver.relation_log_scales.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_batch_step_bounds_match_individual_event_solutions(self):
        solver = StanceEnergyField()
        first = torch.rand(5, 2)
        second = torch.rand(2, 2)
        edges_first = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]])
        edges_second = torch.tensor([[0], [1]])
        labels_first, labels_second = torch.tensor([0, 1, -1, 0]), torch.tensor([1])
        batched = solver(torch.cat((first, second)),
                         torch.cat((edges_first, edges_second + 5), dim=1),
                         torch.cat((labels_first, labels_second)),
                         torch.tensor([0] * 5 + [1] * 2))
        singles = [solver(first, edges_first, labels_first, torch.zeros(5, dtype=torch.long)),
                   solver(second, edges_second, labels_second, torch.zeros(2, dtype=torch.long))]
        for key in ("fields", "source_energy", "relation_energy", "final_energy"):
            torch.testing.assert_close(batched[key], torch.cat([value[key] for value in singles]))

    def test_invalid_solver_settings_and_sources_are_rejected(self):
        for options in ({"scales": ()}, {"scales": (float("nan"),)}, {"num_steps": 0},
                        {"num_steps": 1.5}, {"fidelity": 0}, {"map_bound": -1},
                        {"step_fraction": 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                StanceEnergyField(**options)
        solver = StanceEnergyField()
        for source in (torch.tensor([[-1., 0.]]), torch.tensor([[float("nan"), 0.]])):
            with self.assertRaises(ValueError):
                solver(source, torch.empty(2, 0, dtype=torch.long),
                       torch.empty(0, dtype=torch.long), torch.zeros(1, dtype=torch.long))


class DIGNNEnergyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def test_default_sources_stances_and_separated_view_gradients(self):
        network = model().eval()
        data = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 1])
        data.x.requires_grad_(True)
        views = network.encode_views(data)
        expected = torch.tensor([[1., 1.], [0., 1.], [0., 0.], [0., 0.]]).log1p()
        torch.testing.assert_close(views["energy_source"], expected)
        self.assertEqual(views["structure_features"].shape, (4, 16))
        self.assertEqual(network.topology_encoder.projection.in_features, 49)
        self.assertEqual(network.structure_decoder.out_features, 16)
        self.assertIsNone(torch.autograd.grad(views["topology_graph"].sum(), data.x,
                                              allow_unused=True)[0])
        changed = data.clone()
        changed.directed_edge_stance = 1 - changed.directed_edge_stance
        changed_views = network.encode_views(changed)
        torch.testing.assert_close(views["text_graph"], changed_views["text_graph"])
        self.assertFalse(torch.allclose(views["topology_graph"], changed_views["topology_graph"]))
        changed.x = -10 * changed.x
        torch.testing.assert_close(changed_views["topology_graph"], network.encode_views(changed)["topology_graph"])

    def test_predictions_match_batch_and_permuted_nodes(self):
        network = model().eval()
        graphs = [graph(5, [[0, 0, 1, 1], [1, 2, 3, 4]], [0, 1, 0, 1]),
                  graph(1, [[], []], []), graph(3, [[0], [1]], [-1])]
        batched = Batch.from_data_list(graphs)
        torch.testing.assert_close(network(batched)[0], torch.cat([network(item)[0] for item in graphs]))
        original = graphs[0]
        order = torch.tensor([4, 2, 0, 3, 1])
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(5)
        permuted = Data(x=original.x[order], edge_index=inverse[original.edge_index],
                        directed_edge_index=inverse[original.directed_edge_index],
                        directed_edge_stance=original.directed_edge_stance,
                        root_index=inverse[original.root_index])
        torch.testing.assert_close(network(original)[0], network(permuted)[0])

    def test_sorted_coalescing_preserves_labels_and_rejects_conflicts(self):
        network = model().eval()
        original = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 1])
        expected = network(original)[0]
        duplicate = original.clone()
        duplicate.directed_edge_index = torch.tensor([[1, 0, 0, 0, 1, 2], [3, 2, 1, 1, 3, 2]])
        duplicate.directed_edge_stance = torch.tensor([1, 1, 0, -1, 1, 0])
        torch.testing.assert_close(expected, network(duplicate)[0])
        duplicate.directed_edge_stance[3] = 1
        with self.assertRaisesRegex(ValueError, "conflicting"):
            network(duplicate)

    def test_old_bidirectional_cache_recovers_root_direction_and_aligned_labels(self):
        network = model().eval()
        original = graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 1])
        old = Data(x=original.x, root_index=original.root_index,
                   edge_index=torch.tensor([[2, 3, 0, 1, 1, 0], [0, 1, 2, 3, 0, 1]]),
                   edge_stance=torch.tensor([1, 1, 1, 1, 0, 0]),
                   directed_edge_index=original.directed_edge_index.flip(1))
        # No directed labels: fall back to the aligned old edge pair.
        torch.testing.assert_close(network(original)[0], network(old)[0])

    def test_missing_labels_are_unknown_and_do_not_read_node_states_or_targets(self):
        network = model().eval()
        data = graph(3, [[0, 1], [1, 2]])
        views = network.encode_views(data)
        torch.testing.assert_close(views["energy_source"], torch.zeros(3, 2))
        self.assertTrue(bool((views["edge_stance"] == -1).all()))
        self.assertEqual(float(network(data)[0].isfinite().all()), 1)
        expected = network(data)[0]
        del data.y
        data.node_state = torch.tensor([1, 1, 0])
        data.user_state = torch.randn(1, 3, 3)
        torch.testing.assert_close(expected, network(data)[0])
        torch.testing.assert_close(network.get_diagnostics()["energy_label_coverage"], torch.zeros(1))
        known = data.clone()
        known.directed_edge_stance = torch.tensor([0, 0])
        self.assertFalse(torch.allclose(views["structure_features"], network.encode_views(known)["structure_features"]))

    def test_invalid_labels_and_non_tree_propagation_are_rejected(self):
        network = model()
        data = graph(3, [[0, 1], [1, 2]], [0, 1])
        for labels in (torch.tensor([0]), torch.tensor([0., 1.]), torch.tensor([0, 2])):
            invalid = data.clone()
            invalid.directed_edge_stance = labels
            with self.assertRaises(ValueError):
                network(invalid)
        for edges in ([[0, 1], [2, 2]], [[1, 2], [2, 1]], [[1], [0]]):
            invalid = graph(3, edges, [0] * len(edges[0]))
            with self.assertRaises(ValueError):
                network(invalid)
        unrooted = graph(4, [[1, 2], [2, 1]], [0, 0], directed=False)
        with self.assertRaisesRegex(ValueError, "unrooted"):
            network(unrooted)

    def test_training_updates_relation_maps_and_all_auxiliary_heads(self):
        network = model().train()
        data = Batch.from_data_list([graph(4, [[0, 0, 1], [1, 2, 3]], [0, 1, 0]),
                                     graph(3, [[0, 1], [1, 2]], [1, -1]),
                                     graph(1, [[], []], [])])
        data.y[1] = 1
        optimizer = network.init_optimizer()
        before = network.energy_field.relation_log_scales.detach().clone()
        loss = network.compute_loss(data)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        self.assertGreater(float(network.energy_field.relation_log_scales.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, network.energy_field.relation_log_scales))
        # Reconstruction targets remain fixed observations, not learned fields.
        self.assertFalse(network.encode_views(data)["structure_features"].requires_grad)


if __name__ == "__main__":
    unittest.main()

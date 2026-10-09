"""Grouping semantics tests that need PyTorch but do not require PyG."""

import math
from types import SimpleNamespace
import unittest

import torch

from model.group_gain_grouping import (
    DEFAULT_RELATION_MAPPING,
    build_group_graph,
    grouping_cache_key,
)


def graph(features, edges, root=0, **extra):
    return SimpleNamespace(
        x=torch.tensor(features, dtype=torch.float32),
        reply_edge_index=torch.tensor(edges, dtype=torch.long).reshape(2, -1),
        root_index=torch.tensor([root]),
        graph_id="synthetic-event",
        **extra,
    )


class GroupGainGroupingTest(unittest.TestCase):
    def test_complete_link_avoids_similarity_chain(self):
        # A/B and B/C pass .95; A/C does not.  A connected component would
        # incorrectly merge all three replies.
        vectors = [[2.0, 1.0]] + [
            [math.cos(math.radians(a)), math.sin(math.radians(a))]
            for a in (0, 15, 30)
        ]
        result = build_group_graph(graph(vectors, [[0, 0, 0], [1, 2, 3]]))
        self.assertEqual(result.num_groups, 3)
        for members in result.group_members[1:]:
            x = result.x[list(members)]
            norms = x.norm(dim=1, keepdim=True)
            self.assertTrue(bool(((x / norms) @ (x / norms).t() >= .95 - 1e-6).all()))

    def test_unique_content_drives_partition_not_multiplicity(self):
        data = graph(
            [[2., 1.], [1., 0.], [.97, .24], [.87, .5]],
            [[0, 0, 0], [1, 2, 3]],
        )
        reference = build_group_graph(data)
        for copies in (1, 5, 50):
            copied = graph(
                data.x.tolist() + [data.x[2].tolist()] * copies,
                [[0] * (3 + copies), list(range(1, 4 + copies))],
            )
            result = build_group_graph(copied)
            self.assertEqual(reference.num_groups, result.num_groups)
            self.assertEqual(reference.node_to_group.tolist(), result.node_to_group[:4].tolist())
            self.assertTrue(torch.equal(reference.edge_index, result.edge_index))
            self.assertTrue(torch.equal(reference.edge_type, result.edge_type))

    def test_zero_features_are_singletons_even_exact(self):
        data = graph([[1., 0.], [0., 0.], [0., 0.]], [[0, 0], [1, 2]])
        for mode in ("context", "exact", "feature_only"):
            result = build_group_graph(data, mode=mode)
            self.assertEqual(result.num_groups, 3)
            self.assertEqual(result.diagnostics["zero_feature_nodes"], [1, 2])

    def test_same_original_parent_required_after_parents_merge(self):
        data = graph(
            [[1., 0.], [0., 1.], [0., 1.], [1., 1.], [1., 1.]],
            [[0, 0, 1, 2], [1, 2, 3, 4]],
        )
        result = build_group_graph(data)
        self.assertEqual(result.node_to_group[1], result.node_to_group[2])
        self.assertNotEqual(result.node_to_group[3], result.node_to_group[4])

    def test_explicit_local_relation_types_separate_equal_replies(self):
        data = graph(
            [[1., 0.], [0., 1.], [0., 1.]], [[0, 0], [1, 2]],
            reply_edge_type=["support", "deny"],
        )
        result = build_group_graph(data)
        self.assertEqual(result.num_groups, 3)
        self.assertEqual(set(result.edge_type.tolist()), {2, 3})

    def test_integer_relations_need_explicit_mapping(self):
        data = graph([[1., 0.], [0., 1.]], [[0], [1]], reply_edge_type=torch.tensor([2]))
        with self.assertRaisesRegex(ValueError, "explicit stable"):
            build_group_graph(data)
        result = build_group_graph(data, relation_mapping=DEFAULT_RELATION_MAPPING)
        self.assertEqual(result.edge_type.tolist(), [2])

    def test_stance_is_only_used_after_parent_target_declaration(self):
        data = graph([[1., 0.], [0., 1.], [0., 1.]], [[0, 0], [1, 2]])
        data.directed_edge_index = data.reply_edge_index
        del data.reply_edge_index
        data.directed_edge_stance = torch.tensor([0, 1])
        generic = build_group_graph(data)
        self.assertEqual(generic.num_groups, 2)
        self.assertEqual(generic.edge_type.tolist(), [0])
        self.assertFalse(generic.diagnostics["uses_parent_stance"])
        source = build_group_graph(data, stance_target="source")
        self.assertEqual(source.num_groups, 2)
        with self.assertRaisesRegex(ValueError, "explicit stable"):
            build_group_graph(data, stance_target="parent")
        with self.assertRaisesRegex(ValueError, "explicit raw-value mapping"):
            build_group_graph(data, stance_target="parent", relation_mapping=DEFAULT_RELATION_MAPPING)
        parent = build_group_graph(data, stance_target="parent", relation_mapping={0: 2, 1: 3})
        self.assertEqual(parent.num_groups, 3)
        self.assertTrue(parent.diagnostics["uses_parent_stance"])
        self.assertEqual(set(parent.edge_type.tolist()), {2, 3})

    def test_multiple_parents_cycle_and_original_edges_are_retained(self):
        data = graph(
            [[1., 0.]] + [[0., 1.]] * 5,
            [[0, 0, 1, 2, 3, 4], [1, 2, 3, 3, 4, 3]],
        )
        result = build_group_graph(data)
        self.assertEqual(result.diagnostics["multi_parent_nodes"], [3])
        self.assertEqual(result.diagnostics["cyclic_nodes"], [3, 4])
        self.assertEqual(result.diagnostics["isolated_nonroot_nodes"], [5])
        for node in (3, 4, 5):
            self.assertEqual(result.group_members[int(result.node_to_group[node])], (node,))
        expected = {
            (int(result.node_to_group[u]), int(result.node_to_group[v]), 0)
            for u, v in data.reply_edge_index.t().tolist()
        }
        observed = {
            (u, v, r)
            for (u, v), r in zip(result.edge_index.t().tolist(), result.edge_type.tolist())
        }
        self.assertEqual(observed, expected)

    def test_direction_is_not_inferred_from_undirected_edges(self):
        data = SimpleNamespace(x=torch.eye(2), edge_index=torch.tensor([[0, 1], [1, 0]]), root_index=0)
        with self.assertRaisesRegex(ValueError, "direction is ambiguous"):
            build_group_graph(data)
        data.edge_index = torch.tensor([[0], [1]])
        result = build_group_graph(data, edge_direction="parent_to_child")
        self.assertEqual(result.edge_index.tolist(), [[0], [1]])

    def test_root_is_not_assumed_to_be_node_zero(self):
        data = graph([[0., 1.], [1., 0.], [0., 1.]], [[1, 1], [0, 2]], root=1)
        del data.root_index
        result = build_group_graph(data)
        self.assertEqual(result.root_index, 1)
        self.assertEqual(result.group_members[0], (1,))
        self.assertEqual(result.node_to_group.tolist(), [1, 0, 1])

    def test_isolated_root_needs_explicit_declaration(self):
        data = graph([[1., 0.], [0., 1.]], [[], []])
        del data.root_index
        with self.assertRaisesRegex(ValueError, "unique source"):
            build_group_graph(data)
        data.root_index = 0
        result = build_group_graph(data)
        self.assertTrue(result.diagnostics["root_is_isolated"])
        self.assertEqual(result.edge_index.shape, (2, 0))

    def test_root_only_mapping_and_single_root_event(self):
        data = graph([[1., 0.], [0., 1.]], [[0], [1]])
        result = build_group_graph(data, mode="root_only")
        self.assertEqual(result.group_members, ((0,),))
        self.assertEqual(result.node_to_group.tolist(), [0, -1])
        self.assertEqual(result.diagnostics["root_only_discarded_edges"], 1)
        source = graph([[1., 0.]], [[], []])
        self.assertEqual(build_group_graph(source).num_groups, 1)

    def test_group_internal_original_relation_is_explicitly_retained(self):
        data = graph([[1., 0.], [0., 1.], [0., 1.]], [[0, 1], [1, 2]])
        result = build_group_graph(data, mode="feature_only")
        self.assertEqual(result.num_groups, 2)
        self.assertEqual(result.diagnostics["original_internal_edges"], 1)
        self.assertIn([1, 1], result.edge_index.t().tolist())

    def test_input_features_keep_autograd_and_cache_key_tracks_settings(self):
        data = graph([[1., 0.], [0., 1.]], [[0], [1]])
        data.x.requires_grad_(True)
        result = build_group_graph(data)
        self.assertIs(result.x, data.x)
        result.x.sum().backward()
        self.assertTrue(torch.equal(data.x.grad, torch.ones_like(data.x)))
        self.assertEqual(grouping_cache_key(data), grouping_cache_key(data))
        self.assertNotEqual(grouping_cache_key(data, threshold=.9), grouping_cache_key(data, threshold=.95))
        self.assertEqual(
            grouping_cache_key(data, relation_mapping={"generic": 0, 0: 2, 1: 3}),
            grouping_cache_key(data, relation_mapping={1: 3, 0: 2, "generic": 0}),
        )
        before = grouping_cache_key(data)
        data.x = data.x.detach().clone()
        data.x[1, 0] = .1
        self.assertNotEqual(before, grouping_cache_key(data))

    def test_split_metadata_is_preserved_only_when_present(self):
        data = graph([[1., 0.]], [[], []])
        self.assertNotIn("split", build_group_graph(data).diagnostics)
        data.split = "val"
        self.assertEqual(build_group_graph(data).diagnostics["split"], "val")

    def test_deep_thread_avoids_recursive_graph_traversal(self):
        n = 1600
        data = graph([[1., 0.]] * n, [list(range(n - 1)), list(range(1, n))])
        result = build_group_graph(data)
        self.assertEqual(result.num_groups, n)
        self.assertEqual(result.diagnostics["cyclic_nodes"], [])


if __name__ == "__main__":
    unittest.main()

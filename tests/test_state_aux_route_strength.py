import unittest
from types import SimpleNamespace

import torch

from model.state_aux_samediff import StateAuxSameDiffEnhancer


def make_args():
    return SimpleNamespace(
        dropout=0.0,
        state_aux_num_states=2,
        state_aux_edge_relation_routing=True,
        state_aux_preserve_route_strength=True,
        state_aux_use_node_state_aux=False,
        lambda_node_state_aux=0.0,
        lambda_edge_same_aux=0.0,
        lambda_same_diff_sep=0.0,
        same_dulreg_blend=0.0,
        same_path_blend=1.0,
        diff_path_blend=1.0,
        same_diff_gate_blend=1.0,
        state_aux_transition_attention=False,
        state_aux_reply_target_attention=False,
        state_aux_cross_view_attention=True,
        state_aux_cross_view_attention_blend=1.0,
        state_aux_cross_view_topk_ratio=0.4,
        state_aux_cross_view_min_nodes=1,
        state_aux_cross_view_vectorized=True,
        state_aux_cross_view_bucket_sizes=[2, 4, 8],
    )


class RouteStrengthTest(unittest.TestCase):
    def setUp(self):
        self.enhancer = StateAuxSameDiffEnhancer(2, make_args())
        self.h = torch.tensor([[1.0, 2.0], [0.5, -0.5]])
        self.edge_index = torch.tensor([[0], [1]])

    def test_single_parent_same_content_keeps_probability_as_strength(self):
        low_content, low_strength, _, _ = (
            self.enhancer._aggregate_same_uncertain(
                self.h,
                self.edge_index,
                torch.tensor([0.05]),
            )
        )
        high_content, high_strength, _, _ = (
            self.enhancer._aggregate_same_uncertain(
                self.h,
                self.edge_index,
                torch.tensor([0.95]),
            )
        )

        self.assertTrue(torch.allclose(low_content, high_content))
        self.assertAlmostEqual(float(low_strength[1]), 0.05, places=6)
        self.assertAlmostEqual(float(high_strength[1]), 0.95, places=6)

    def test_post_norm_blend_preserves_update_strength(self):
        candidate = torch.tensor([[0.0, 0.0], [2.5, 1.5]])
        low = self.enhancer._blend_route_strength(
            self.h,
            candidate,
            torch.tensor([[0.0], [0.05]]),
        )
        high = self.enhancer._blend_route_strength(
            self.h,
            candidate,
            torch.tensor([[0.0], [0.95]]),
        )

        low_update = torch.linalg.vector_norm(low[1] - self.h[1])
        high_update = torch.linalg.vector_norm(high[1] - self.h[1])
        self.assertAlmostEqual(
            float(high_update / low_update),
            19.0,
            places=5,
        )

    def test_vectorized_cross_view_matches_per_graph_reference(self):
        torch.manual_seed(7)
        enhancer = StateAuxSameDiffEnhancer(4, make_args()).eval()
        same = torch.randn(8, 4)
        diff = torch.randn(8, 4)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])

        expected_same, expected_diff = (
            enhancer._apply_cross_view_attention_loop(
                same,
                diff,
                batch,
            )
        )
        actual_same, actual_diff = (
            enhancer._apply_cross_view_attention_vectorized(
                same,
                diff,
                batch,
            )
        )
        self.assertTrue(
            torch.allclose(actual_same, expected_same, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(actual_diff, expected_diff, atol=1e-6)
        )


if __name__ == '__main__':
    unittest.main()

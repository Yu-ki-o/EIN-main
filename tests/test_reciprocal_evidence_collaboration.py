import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)


def make_args(enabled=False, fusion_mode="change_semantic_tree"):
    return SimpleNamespace(
        max_hop=4,
        dropout=0.0,
        global_pool="mean",
        n_layers_conv=2,
        edge_norm=True,
        relation_hidden_dim=8,
        relation_temperature=1.0,
        stance_route_temperature=0.5,
        stance_route_hard=False,
        use_uncertainty_sampling=False,
        uncertainty_sample_temperature=0.5,
        uncertainty_keep_floor=0.05,
        edge_relation_distribution="softmax",
        use_degree_importance=False,
        lambda_edge_relation_aux=0.0,
        lambda_view_mi_aux=0.0,
        use_semantic_parity_gnn=True,
        semantic_parity_aggregation="mean",
        semantic_parity_residual=True,
        semantic_parity_direction="bottom_up",
        semantic_node_weight_mode="local",
        semantic_change_encoder="mlp",
        semantic_change_hidden_dim=8,
        use_gaussian_semantic_change_bottleneck=False,
        lambda_semantic_change_bottleneck=0.0,
        lambda_semantic_tree_change_mi_aux=0.0,
        use_node_keep_in_change_pool=False,
        use_semantic_tree_transformer=False,
        semantic_tree_input_mode="support_deny_original",
        semantic_tree_query_mode="learned",
        semantic_tree_num_queries=1,
        semantic_tree_transformer_layers=1,
        semantic_tree_transformer_ffn_dim=16,
        semantic_tree_transformer_dropout=0.0,
        semantic_tree_transformer_max_depth=4,
        semantic_tree_depth_dim=8,
        semantic_tree_transformer_pool="mean",
        use_semantic_tree_change_uncertainty_bias=False,
        semantic_tree_uncertainty_source="none",
        use_trend_graph=False,
        use_vertical_path_attention=False,
        use_conflict_field_bottleneck=False,
        use_global_ds_fusion=False,
        classification_fusion_mode=fusion_mode,
        classification_fusion_hidden_dim=16,
        classification_head_mode="fusion",
        use_reciprocal_evidence_collaboration=enabled,
        repv_num_slots=3,
        repv_proposal_temperature=1.0,
        repv_proposal_bias_scale=0.5,
        repv_counterfactual_temperature=1.0,
        repv_warmup_epochs=1,
        repv_ramp_epochs=2,
        repv_residual_gate_init=-2.0,
        repv_dropout=0.0,
        lambda_repv_classification_aux=0.1,
        lambda_repv_feedback_aux=0.1,
        lambda_repv_overlap_aux=0.01,
        lambda_repv_diversity_aux=0.001,
        lr=1e-3,
        weight_decay=0.0,
    )


def make_batch():
    first = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_stance=torch.tensor([0, 1]),
        y=torch.tensor([1]),
        num_hop=torch.tensor([2]),
        user_state=torch.zeros(1, 4, 3),
    )
    second = Data(
        x=torch.randn(2, 5),
        edge_index=torch.tensor([[0], [1]]),
        edge_stance=torch.tensor([0]),
        y=torch.tensor([0]),
        num_hop=torch.tensor([1]),
        user_state=torch.zeros(1, 4, 3),
    )
    return Batch.from_data_list([first, second])


def make_model(args):
    return BiGCN_UncertaintySemanticChange(
        in_feats=5,
        hid_feats=8,
        out_feats=8,
        num_classes=2,
        args=args,
        device=torch.device("cpu"),
    )


class ReciprocalEvidenceCollaborationTest(unittest.TestCase):
    def test_missing_switch_and_explicit_false_are_identical(self):
        args_missing = make_args(enabled=False)
        delattr(args_missing, "use_reciprocal_evidence_collaboration")
        args_false = make_args(enabled=False)

        torch.manual_seed(19)
        model_missing = make_model(args_missing).eval()
        torch.manual_seed(19)
        model_false = make_model(args_false).eval()

        self.assertIsNone(model_missing.reciprocal_evidence_collaboration)
        self.assertIsNone(model_false.reciprocal_evidence_collaboration)
        self.assertEqual(
            model_missing.fusion[0].in_features,
            model_false.fusion[0].in_features,
        )
        for (name_a, value_a), (name_b, value_b) in zip(
            model_missing.state_dict().items(),
            model_false.state_dict().items(),
        ):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(value_a, value_b))

        torch.manual_seed(23)
        data = make_batch()
        with torch.no_grad():
            output_missing = model_missing(data)[0]
            output_false = model_false(data)[0]
        self.assertTrue(torch.equal(output_missing, output_false))

    def test_enabled_module_preserves_branch_and_fusion_dimensions(self):
        model = make_model(make_args(enabled=True)).train()
        model.set_epoch(0)
        data = make_batch()
        output = model(data)[0]

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            model.classification_branch_names,
            ("change", "semantic_tree"),
        )
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertEqual(
            tuple(model._last_repv_prototypes.shape),
            (2, 3, 8),
        )
        self.assertTrue(
            torch.allclose(
                model._last_repv_proposal_attention.sum(dim=-1),
                torch.ones(2, 3),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_repv_tree_attention.sum(dim=-1),
                torch.ones(2, 3),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                model._last_repv_slot_weights.sum(dim=-1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertEqual(float(model._last_repv_feedback_scale), 0.0)
        self.assertIsNone(model._last_repv_counterfactual_necessity)

        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()
        self.assertIsNotNone(
            model.reciprocal_evidence_collaboration.slot_queries.grad
        )

    def test_counterfactual_feedback_starts_after_warmup(self):
        model = make_model(make_args(enabled=True)).train()
        model.set_epoch(2)
        data = make_batch()
        output = model(data)[0]

        self.assertEqual(float(model._last_repv_feedback_scale), 1.0)
        self.assertEqual(
            tuple(model._last_repv_counterfactual_necessity.shape),
            (2, 3),
        )
        self.assertEqual(
            tuple(model._last_repv_feedback_target.shape),
            (2, 3),
        )
        self.assertTrue(torch.isfinite(model._last_repv_feedback_loss))
        self.assertTrue(torch.isfinite(model._last_repv_overlap_loss))
        self.assertTrue(torch.isfinite(model.auxiliary_loss()))

        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()
        self.assertIsNotNone(
            model.reciprocal_evidence_collaboration
            .collaboration_classifier.weight.grad
        )

    def test_eval_does_not_compute_counterfactual_paths(self):
        model = make_model(make_args(enabled=True)).eval()
        model.set_epoch(20)
        with torch.no_grad():
            output = model(make_batch())[0]
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(model._last_repv_counterfactual_necessity)
        self.assertIsNone(model._last_repv_feedback_target)
        self.assertEqual(float(model._last_repv_feedback_loss), 0.0)

    def test_invalid_fusion_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both 'change' and"):
            make_model(make_args(enabled=True, fusion_mode="change"))

    def test_branch_sum_head_remains_compatible(self):
        args = make_args(enabled=True)
        args.classification_head_mode = "branch_sum"
        model = make_model(args).eval()
        with torch.no_grad():
            output = model(make_batch())[0]
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(
            set(model.branch_classifiers),
            {"change", "semantic_tree"},
        )


if __name__ == "__main__":
    unittest.main()

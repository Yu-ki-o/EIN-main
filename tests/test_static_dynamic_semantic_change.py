import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.BiGCN_StaticDynamicSemanticChange import (
    BiGCN_StaticDynamicSemanticChange,
)
from model.ResGCN_StaticDynamicSemanticChange import (
    ResGCN_StaticDynamicSemanticChange,
    StaticDynamicChangeEncoder,
)


def encoder_args():
    return SimpleNamespace(
        dropout=0.0,
        max_hop=4,
        n_layers_conv=2,
        st_change_attention_temperature=1.0,
        st_change_min_std=1e-3,
        st_change_max_wasserstein_distance=20.0,
        st_change_use_ib_sampling=False,
    )


def model_args():
    return SimpleNamespace(
        max_hop=4,
        dropout=0.0,
        global_pool="mean",
        n_layers_conv=2,
        edge_norm=True,
        relation_hidden_dim=8,
        relation_temperature=1.0,
        stance_route_temperature=0.5,
        stance_route_hard=True,
        use_uncertainty_sampling=False,
        uncertainty_sample_temperature=0.5,
        uncertainty_keep_floor=0.05,
        use_ds_mass_routing=False,
        ds_unknown_prior=2.0,
        lambda_ds_unknown_edge_aux=0.0,
        use_global_ds_fusion=False,
        use_degree_importance=True,
        degree_importance_strength=1.0,
        lambda_edge_relation_aux=0.1,
        lambda_edge_relation_warmup=0.5,
        lambda_view_mi_aux=0.0,
        use_semantic_parity_gnn=False,
        semantic_node_weight_mode="local",
        semantic_change_encoder="mlp",
        semantic_change_hidden_dim=8,
        use_node_keep_in_change_pool=False,
        use_vertical_path_attention=False,
        use_semantic_tree_transformer=False,
        use_trend_graph=False,
        uncertainty_trend_hidden_dim=8,
        use_conflict_field_bottleneck=False,
        classification_fusion_mode="change",
        classification_fusion_hidden_dim=16,
        classification_class_weights=[1.0, 1.0],
        classification_head_mode="fusion",
        st_change_attention_temperature=1.0,
        st_change_min_std=1e-3,
        st_change_max_wasserstein_distance=20.0,
        st_change_use_ib_sampling=False,
        lambda_static_change_cls=0.15,
        lambda_dynamic_change_cls=0.15,
        lambda_temporal_prediction=0.05,
        lambda_information_bottleneck=1e-3,
        lambda_variant_intervention=0.05,
        lambda_static_dynamic_decorr=0.01,
        lambda_invariant_mask_prior=0.01,
        invariant_mask_prior=0.5,
        spatiotemporal_warmup_epochs=2,
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


class StaticDynamicChangeEncoderTest(unittest.TestCase):
    def test_identical_view_trajectories_produce_zero_change(self):
        encoder = StaticDynamicChangeEncoder(4, encoder_args()).eval()
        history = [torch.randn(5, 4) for _ in range(3)]
        original = [torch.randn(5, 4) for _ in range(3)]
        edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
        depth = torch.tensor([0, 1, 2, 0, 1])
        encoder.set_trajectories(history, history, original, depth)

        output = encoder(
            history[-1],
            history[-1],
            edge_index=edge_index,
        )

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(
            torch.equal(
                encoder.last_static_nodes,
                torch.zeros_like(encoder.last_static_nodes),
            )
        )
        self.assertTrue(
            torch.equal(
                encoder.last_dynamic_nodes,
                torch.zeros_like(encoder.last_dynamic_nodes),
            )
        )

    def test_temporal_attention_is_causal_and_differentiable(self):
        encoder = StaticDynamicChangeEncoder(4, encoder_args()).train()
        support = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
        deny = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
        original = [torch.randn(5, 4, requires_grad=True) for _ in range(3)]
        edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])
        depth = torch.tensor([0, 1, 2, 0, 1])
        encoder.set_trajectories(support, deny, original, depth)

        output = encoder(
            support[-1],
            deny[-1],
            edge_index=edge_index,
        )
        loss = (
            output.pow(2).mean()
            + encoder.last_kl_loss
            + encoder.last_temporal_prediction_loss
        )
        loss.backward()

        upper = torch.triu(
            encoder.last_temporal_near_attention,
            diagonal=1,
        )
        self.assertTrue(torch.equal(upper, torch.zeros_like(upper)))
        self.assertIsNotNone(support[-1].grad)
        self.assertIsNotNone(deny[-1].grad)


class ResGCNStaticDynamicSemanticChangeTest(unittest.TestCase):
    def test_model_forward_and_all_auxiliary_losses(self):
        model = ResGCN_StaticDynamicSemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=model_args(),
            device=torch.device("cpu"),
        )
        model.set_epoch(1)
        data = make_batch()

        output, uncertainty, support, deny = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(uncertainty.shape), (2, 4, 1))
        self.assertEqual(tuple(support.shape), (2, 4, 1))
        self.assertEqual(tuple(deny.shape), (2, 4, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model._last_static_change_loss)
        self.assertIsNotNone(model._last_dynamic_change_loss)
        self.assertIsNotNone(model._last_information_bottleneck_loss)
        self.assertIsNotNone(model._last_variant_intervention_loss)


class BiGCNStaticDynamicSemanticChangeTest(unittest.TestCase):
    def test_model_forward_uses_three_pseudo_time_states(self):
        model = BiGCN_StaticDynamicSemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=model_args(),
            device=torch.device("cpu"),
        )
        model.set_epoch(1)
        data = make_batch()

        output, uncertainty, support, deny = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(uncertainty.shape), (2, 4, 1))
        self.assertEqual(
            model.static_dynamic_encoder.last_temporal_near_attention.size(1),
            2,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax

from model.EIN_StanceGuidedGAT import (
    RelationTeacherGATLayer,
    StanceGuidedGAT,
)


def make_args(backbone="bigcn"):
    return SimpleNamespace(
        stance_gat_backbone=backbone,
        max_hop=4,
        dropout=0.0,
        global_pool="mean",
        n_layers_conv=2,
        edge_norm=True,
        gat_heads=2,
        gat_num_layers=2,
        gat_attention_dropout=0.0,
        gat_negative_slope=0.2,
        stance_relation_attention_bias=1.0,
        stance_relation_hidden_dim=8,
        stance_gat_fusion_hidden_dim=16,
        stance_relation_class_weights=[1.0, 1.0],
        stance_relation_temperature=1.0,
        stance_self_support_prior=1.0,
        lambda_edge_relation=0.1,
        lambda_attention_kl=0.05,
        attention_kl_warmup_epochs=0,
        attention_kl_ramp_epochs=0,
        attention_kl_min_labeled_edges=1,
        classification_class_weights=None,
        lr=1e-3,
        weight_decay=0.0,
    )


def make_batch():
    first_edges = torch.tensor([[0, 0, 1], [1, 2, 3]])
    first_stance = torch.tensor([0, 1, 0])
    first = Data(
        x=torch.randn(4, 6),
        edge_index=first_edges,
        directed_edge_index=first_edges.clone(),
        edge_stance=first_stance,
        directed_edge_stance=first_stance.clone(),
        y=torch.tensor([1]),
    )
    second_edges = torch.tensor([[0, 0], [1, 2]])
    second_stance = torch.tensor([1, 0])
    second = Data(
        x=torch.randn(3, 6),
        edge_index=second_edges,
        directed_edge_index=second_edges.clone(),
        edge_stance=second_stance,
        directed_edge_stance=second_stance.clone(),
        y=torch.tensor([0]),
    )
    return Batch.from_data_list([first, second])


class RelationTeacherGATLayerTest(unittest.TestCase):
    def test_support_probability_downweights_opposing_edge(self):
        layer = RelationTeacherGATLayer(
            hidden_dim=4,
            heads=2,
            dropout=0.0,
            relation_bias=1.0,
        ).eval()
        hidden = torch.zeros(3, 4)
        edge_index = torch.tensor([[1, 2, 0], [0, 0, 0]])
        support_probability = torch.tensor([0.9, 0.1, 1.0])

        _, raw_attention, biased_attention = layer(
            hidden,
            edge_index,
            support_probability,
        )

        self.assertTrue(
            torch.allclose(
                raw_attention,
                torch.full_like(raw_attention, 1.0 / 3.0),
            )
        )
        self.assertTrue((biased_attention[0] > biased_attention[1]).all())
        self.assertTrue((biased_attention[2] > biased_attention[0]).all())


class StanceGuidedGATTest(unittest.TestCase):
    def test_both_backbones_produce_finite_outputs_and_auxiliary_loss(self):
        batch = make_batch()
        for backbone in ("bigcn", "resgcn"):
            with self.subTest(backbone=backbone):
                model = StanceGuidedGAT(
                    in_feats=6,
                    hidden_dim=8,
                    num_classes=2,
                    args=make_args(backbone),
                ).train()
                output, U, S, D = model(batch)

                self.assertEqual(tuple(output.shape), (2, 2))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(model.auxiliary_loss()))
                self.assertIsNone(U)
                self.assertIsNone(S)
                self.assertIsNone(D)

                loss = model.classification_loss(output, batch.y)
                loss = loss + model.auxiliary_loss()
                loss.backward()
                self.assertIsNotNone(
                    model.edge_relation_classifier[-1].weight.grad
                )
                self.assertIsNotNone(model.gat_layers[-1].linear.weight.grad)

    def test_kl_target_is_detached_from_relation_teacher(self):
        model = StanceGuidedGAT(
            in_feats=6,
            hidden_dim=8,
            num_classes=2,
            args=make_args(),
        )
        edge_index = torch.tensor(
            [[1, 2, 0, 1, 2], [0, 0, 0, 1, 2]]
        )
        support = torch.tensor(
            [0.8, 0.2, 1.0, 1.0, 1.0],
            requires_grad=True,
        )
        teacher = softmax(
            torch.log(support.detach()),
            edge_index[1],
            num_nodes=3,
        )
        raw_attention = teacher.unsqueeze(-1).repeat(1, 2)
        raw_attention.requires_grad_(True)

        loss = model._attention_kl(
            raw_attention,
            support,
            edge_index,
            torch.tensor([0, 1]),
            num_reply_edges=2,
            num_nodes=3,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertIsNotNone(raw_attention.grad)
        self.assertIsNone(support.grad)

    def test_predictions_do_not_read_llm_labels_at_inference(self):
        torch.manual_seed(11)
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed.directed_edge_stance = 1 - changed.directed_edge_stance
        changed.edge_stance = 1 - changed.edge_stance
        model = StanceGuidedGAT(
            in_feats=6,
            hidden_dim=8,
            num_classes=2,
            args=make_args(),
        ).eval()

        with torch.no_grad():
            original_output = model(batch)[0]
            changed_output = model(changed)[0]

        self.assertTrue(torch.equal(original_output, changed_output))

    def test_three_dataset_configs_select_supported_backbone(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "EIN"
        for dataset in ("DRWeibo", "Weibo", "Pheme"):
            path = config_dir / "{}_StanceGuidedGAT_word2vec.yaml".format(
                dataset
            )
            with self.subTest(dataset=dataset):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["base_model"], "StanceGuidedGAT")
                self.assertIn(
                    config["stance_gat_backbone"],
                    {"bigcn", "resgcn"},
                )


if __name__ == "__main__":
    unittest.main()

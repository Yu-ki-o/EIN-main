import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.utils import softmax

from model.EIN_StanceGuidedGAT import (
    DualChannelRelationTeacherGATLayer,
    RelationTeacherGATLayer,
    StanceGuidedGAT,
)


def make_args(backbone="bigcn"):
    return SimpleNamespace(
        stance_gat_backbone=backbone,
        stance_gat_dual_channel=False,
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
        stance_relation_gate_power=1.0,
        stance_relation_hidden_dim=8,
        stance_gat_fusion_hidden_dim=16,
        stance_relation_class_weights=[1.0, 1.0],
        stance_relation_temperature=1.0,
        stance_self_support_prior=1.0,
        lambda_edge_relation=0.1,
        lambda_attention_kl=0.05,
        attention_kl_warmup_epochs=0,
        attention_kl_ramp_epochs=0,
        attention_kl_min_labeled_edges=2,
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

    def test_dual_channel_support_self_and_deny_children_only(self):
        layer = DualChannelRelationTeacherGATLayer(
            hidden_dim=4,
            heads=2,
            dropout=0.0,
            relation_gate_power=1.0,
        ).eval()
        # Zero features make the two content-attention logits identical, so
        # the resulting deny weights isolate the edge-probability gate.
        hidden = torch.zeros(4, 4)
        reply_edge_index = torch.tensor([[1, 2], [0, 0]])
        relation_probability = torch.tensor(
            [[0.9, 0.1], [0.2, 0.8]]
        )

        outputs = layer(
            hidden,
            reply_edge_index,
            relation_probability,
            self_support_prior=1.0,
        )

        # Reply edges come first; the final four support entries are self-loops.
        self.assertTrue(
            torch.allclose(
                outputs["support_biased_attention"][-1],
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertEqual(
            tuple(outputs["deny_biased_attention"].shape),
            (2, 2),
        )
        self.assertTrue(
            (outputs["deny_biased_attention"][1]
             > outputs["deny_biased_attention"][0]).all()
        )
        self.assertTrue(
            torch.allclose(
                outputs["deny_relation_gate"],
                relation_probability[:, 1],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["support_relation_gate"][:2],
                relation_probability[:, 0],
                atol=1e-6,
            )
        )
        # The relation gates are absolute: their multiplication is not
        # followed by another neighbourhood normalisation.
        self.assertTrue(
            torch.allclose(
                outputs["deny_biased_attention"].sum(dim=0),
                torch.full((2,), 0.45),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["support_biased_attention"][[0, 1, 2]].sum(dim=0),
                torch.full((2,), 0.7),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["deny_nodes"][3],
                torch.zeros(4),
                atol=1e-6,
            )
        )


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

    def test_dual_channel_model_produces_support_and_deny_views(self):
        args = make_args()
        args.stance_gat_dual_channel = True
        model = StanceGuidedGAT(
            in_feats=6,
            hidden_dim=8,
            num_classes=2,
            args=args,
        ).train()
        batch = make_batch()

        output = model(batch)[0]
        loss = model.classification_loss(output, batch.y)
        loss = loss + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(model._last_support_nodes.shape), (7, 8))
        self.assertEqual(tuple(model._last_deny_nodes.shape), (7, 8))
        self.assertIsNotNone(
            model.gat_layers[-1].support_attention_linear.weight.grad
        )
        self.assertIsNotNone(
            model.gat_layers[-1].deny_attention_linear.weight.grad
        )
        self.assertIsNotNone(
            model.gat_layers[-1].value_linear.weight.grad
        )
        self.assertIsNotNone(
            model.edge_relation_classifier[-1].weight.grad
        )

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

    def test_dual_kl_uses_conditional_child_proportions(self):
        args = make_args()
        args.stance_gat_dual_channel = True
        model = StanceGuidedGAT(
            in_feats=6,
            hidden_dim=8,
            num_classes=2,
            args=args,
        )
        # The Support children carry only half of the total attention because
        # the other half may belong to the self-loop.  Conditionally, however,
        # their 0.4:0.1 allocation is exactly the 0.8:0.2 teacher ratio.
        child_attention = torch.tensor(
            [[0.4, 0.4], [0.1, 0.1]], requires_grad=True
        )
        child_probability = torch.tensor(
            [0.8, 0.2], requires_grad=True
        )
        loss = model._child_attention_kl(
            child_attention,
            child_probability,
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            num_nodes=3,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertIsNotNone(child_attention.grad)
        self.assertIsNone(child_probability.grad)

    def test_predictions_do_not_read_llm_labels_at_inference(self):
        batch = make_batch()
        changed = copy.deepcopy(batch)
        changed.directed_edge_stance = 1 - changed.directed_edge_stance
        changed.edge_stance = 1 - changed.edge_stance
        for dual_channel in (False, True):
            with self.subTest(dual_channel=dual_channel):
                torch.manual_seed(11)
                args = make_args()
                args.stance_gat_dual_channel = dual_channel
                model = StanceGuidedGAT(
                    in_feats=6,
                    hidden_dim=8,
                    num_classes=2,
                    args=args,
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
                self.assertTrue(config["stance_gat_dual_channel"])


if __name__ == "__main__":
    unittest.main()

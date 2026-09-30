import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from model.SEEGraphMAE import SEEGraphMAE
from trainer.SEEGraphMAE_trainer import SEEGraphMAETrainer


def make_args():
    return SimpleNamespace(
        lr=5e-4,
        weight_decay=1e-4,
        dropout=0.0,
        see_mask_ratio=0.25,
        see_alpha_rec=0.1,
        see_alpha_uni=0.5,
        see_uniformity_t=2.0,
        see_mlp_layers=2,
        see_decoder_layers=2,
        see_global_pool="mean",
        classification_class_weights=[1.0, 1.0],
        see_subgraph_mask_steps=2,
        see_subgraph_mask_lr=0.1,
        see_subgraph_sparsity=0.1,
        see_subgraph_mask_init=2.0,
        see_ttt_lr=5e-4,
        see_ttt_weight_decay=0.0,
        see_ttt_reset_each_batch=True,
        see_ttt_epochs=1,
        see_alpha_sub=0.5,
        see_use_subgraph_regularizer=True,
    )


def make_graph(label, num_nodes):
    edge_index = torch.stack(
        (
            torch.arange(0, num_nodes - 1),
            torch.arange(1, num_nodes),
        ),
        dim=0,
    )
    return Data(
        x=torch.randn(num_nodes, 5),
        edge_index=edge_index,
        directed_edge_index=edge_index.clone(),
        y=torch.tensor([label]),
    )


def make_batch():
    return Batch.from_data_list(
        [
            make_graph(0, 4),
            make_graph(1, 5),
        ]
    )


class MiniSEEGraphMAETrainer(SEEGraphMAETrainer):
    def __init__(self):
        pass


class SEEGraphMAETest(unittest.TestCase):
    def test_load_evaluation_checkpoint_restores_state_dict(self):
        trainer = MiniSEEGraphMAETrainer()
        trainer.model = torch.nn.Linear(3, 2)
        trainer.device = torch.device("cpu")
        trainer.logger = logging.getLogger("see-graphmae-eval-test")
        expected = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best_model.pth.m"
            torch.save(expected, checkpoint_path)
            trainer.args = SimpleNamespace(checkpoint_path=str(checkpoint_path))
            with torch.no_grad():
                for parameter in trainer.model.parameters():
                    parameter.zero_()

            trainer.load_evaluation_checkpoint()

        actual = trainer.model.state_dict()
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]))

    def test_forward_and_self_supervised_loss_are_differentiable(self):
        model = SEEGraphMAE(
            in_feats=5,
            hid_feats=8,
            num_classes=2,
            args=make_args(),
            device=torch.device("cpu"),
        )
        data = make_batch()

        output = model(data)
        ssl_loss, metrics = model.self_supervised_loss(data)
        loss = model.classification_loss(output, data.y) + ssl_loss
        loss.backward()

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("rec_top_down", metrics)
        self.assertIsNotNone(model.mask_token.grad)

    def test_attention_mask_and_ttt_path(self):
        args = make_args()
        model = SEEGraphMAE(
            in_feats=5,
            hid_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        )
        data = make_batch()
        trainer = MiniSEEGraphMAETrainer()
        trainer.model = model
        trainer.args = args
        trainer.device = torch.device("cpu")

        mask = trainer._optimize_attention_mask(model, data)
        self.assertEqual(tuple(mask.shape), (data.x.size(0),))
        self.assertTrue(((mask >= 0.0) & (mask <= 1.0)).all())

        trained_state = trainer._clone_state_dict()
        reference_model = trainer._frozen_reference_model()
        trainer._adapt_test_batch(data, trained_state, reference_model)
        output = model(data)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

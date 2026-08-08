import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from model.BiGCN_UncertaintySemanticChange import (
    BiGCN_UncertaintySemanticChange,
)
from model.stance_motif_codebook import StanceMotifCodebookBranch


def make_args(mode):
    return SimpleNamespace(
        max_hop=4,
        dropout=0.0,
        global_pool="mean",
        n_layers_conv=2,
        relation_hidden_dim=8,
        use_trend_graph=False,
        use_semantic_tree_transformer="semantic_tree" in mode,
        semantic_tree_transformer_heads=2,
        semantic_tree_transformer_layers=1,
        semantic_tree_depth_dim=4,
        classification_fusion_mode=mode,
        classification_fusion_hidden_dim=16,
        codebook_prototypes_per_type=3,
        codebook_data_initialize=True,
        lr=1e-3,
        weight_decay=0.0,
    )


def make_batch():
    chain = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        # S-D-S yields the overlapping motifs SD followed by DS.
        edge_stance=torch.tensor([0, 1, 0]),
        y=torch.tensor([1]),
        num_hop=torch.tensor([3]),
        user_state=torch.zeros(1, 4, 3),
    )
    shallow = Data(
        x=torch.randn(2, 5),
        edge_index=torch.tensor([[0], [1]]),
        edge_stance=torch.tensor([1]),
        y=torch.tensor([0]),
        num_hop=torch.tensor([1]),
        user_state=torch.zeros(1, 4, 3),
    )
    return Batch.from_data_list([chain, shallow])


class StanceMotifCodebookBranchTest(unittest.TestCase):
    def test_extracts_ordered_overlapping_hard_label_motifs(self):
        branch = StanceMotifCodebookBranch(
            hidden_dim=8,
            max_hop=4,
            args=make_args("codebook"),
        )
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        motifs = branch._extract_motifs(
            edge_index=edge_index,
            edge_stance=torch.tensor([0, 1, 0]),
            depth=torch.tensor([0, 1, 2, 3]),
            batch=torch.zeros(4, dtype=torch.long),
        )

        self.assertEqual(motifs["type"].tolist(), [1, 2])
        self.assertEqual(motifs["parent_token"].tolist(), [-1, 0])
        self.assertEqual(motifs["child"].tolist(), [2, 3])
        self.assertEqual(
            [branch.motif_names[index] for index in motifs["type"]],
            ["SD", "DS"],
        )

    def test_codebook_mode_backpropagates_through_selected_prototypes(self):
        args = make_args("codebook")
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).train()
        data = make_batch()

        output, _, _, _ = model(data)
        loss = F.nll_loss(output, data.y) + model.auxiliary_loss()
        loss.backward()

        self.assertEqual(model.classification_branch_names, ("codebook",))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(model._last_codebook_graph.shape), (2, 8))
        self.assertEqual(model._last_codebook_motif_types.tolist(), [1, 2])
        self.assertEqual(model._last_codebook_motif_depth.tolist(), [2, 3])
        self.assertEqual(
            tuple(model._last_codebook_prototype_indices.shape),
            (2,),
        )
        self.assertIsNotNone(model.stance_motif_codebook.codebook.grad)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(model._last_codebook_aux_loss), 0.0)

    def test_codebook_semantic_tree_fuses_two_graph_branches(self):
        args = make_args("codebook_semantic_tree")
        model = BiGCN_UncertaintySemanticChange(
            in_feats=5,
            hid_feats=8,
            out_feats=8,
            num_classes=2,
            args=args,
            device=torch.device("cpu"),
        ).eval()

        output, _, _, _ = model(make_batch())

        self.assertEqual(
            model.classification_branch_names,
            ("codebook", "semantic_tree"),
        )
        self.assertEqual(model.fusion[0].in_features, 16)
        self.assertEqual(tuple(model._last_codebook_graph.shape), (2, 8))
        self.assertEqual(
            tuple(model._last_semantic_tree_graph.shape),
            (2, 8),
        )
        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()

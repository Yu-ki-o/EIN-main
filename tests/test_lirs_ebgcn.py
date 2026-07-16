from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.data import Batch, Data

from model.LIRS_EBGCN import LIRSEBGCN
from trainer.LIRS_EBGCN_trainer import LIRSEBGCNTrainer


def _args(backbone):
    return SimpleNamespace(
        lr=5e-4,
        weight_decay=1e-4,
        dropout=0.1,
        hidden_dim=8,
        ebgcn_hidden_dim=8,
        ebgcn_output_dim=8,
        ebgcn_edge_num=2,
        ebgcn_edge_infer_td=True,
        ebgcn_edge_infer_bu=True,
        n_layers_conv=2,
        n_layers_fc=2,
        skip_connection=True,
        edge_norm=True,
        global_pool='sum',
        lirs_ebgcn_backbone=backbone,
        lirs_removal_strength=0.35,
        lirs_edge_spurious_suppression=0.25,
        lirs_spurious_ratio=0.3,
        lirs_spurious_dropout=0.0,
        lirs_detach_spurious_projection=True,
        lirs_num_clusters=2,
        lirs_prototype_momentum=0.9,
        lirs_cluster_grl_scale=1.0,
    )


def _batch():
    graphs = [
        Data(
            x=torch.randn(4, 6),
            edge_index=torch.tensor(
                [[0, 0, 1], [1, 2, 3]], dtype=torch.long
            ),
            directed_edge_index=torch.tensor(
                [[0, 0, 1], [1, 2, 3]], dtype=torch.long
            ),
            y=torch.tensor([0]),
        ),
        Data(
            x=torch.randn(3, 6),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            directed_edge_index=torch.tensor(
                [[0, 1], [1, 2]], dtype=torch.long
            ),
            y=torch.tensor([1]),
        ),
        Data(
            x=torch.randn(2, 6),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            directed_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            y=torch.tensor([0]),
        ),
        Data(
            x=torch.randn(3, 6),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            directed_edge_index=torch.tensor(
                [[0, 0], [1, 2]], dtype=torch.long
            ),
            y=torch.tensor([1]),
        ),
    ]
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize('backbone', ['bigcn', 'resgcn'])
def test_lirs_ebgcn_forward_and_backward(backbone):
    model = LIRSEBGCN(
        6, 8, 2, _args(backbone), torch.device('cpu')
    )
    model.train()
    out, auxiliary = model(_batch())
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()
    expected = {
        'edge_loss',
        'shortcut_loss',
        'independence_loss',
        'conditional_independence_loss',
        'cluster_loss',
        'infomax_loss',
        'gate_balance_loss',
        'mean_spurious_gate',
        'invariant_graph',
        'spurious_graph',
    }
    assert set(auxiliary) == expected
    scalar_keys = expected - {
        'invariant_graph',
        'spurious_graph',
    }
    for key in scalar_keys:
        assert auxiliary[key].ndim == 0
        assert torch.isfinite(auxiliary[key])
    assert 0.0 <= auxiliary['mean_spurious_gate'].item() <= 1.0

    loss = F.nll_loss(out, _batch().y.view(-1))
    loss = loss + sum(
        auxiliary[key]
        for key in scalar_keys
        if key != 'mean_spurious_gate'
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_lirs_ebgcn_eval_does_not_update_prototypes():
    model = LIRSEBGCN(
        6, 8, 2, _args('bigcn'), torch.device('cpu')
    )
    model.train()
    model(_batch())
    before = model.prototype_bank.prototype_counts.clone()
    model.eval()
    with torch.no_grad():
        model(_batch())
    assert torch.equal(before, model.prototype_bank.prototype_counts)


def test_lirs_ebgcn_rejects_unknown_backbone():
    with pytest.raises(ValueError, match='lirs_ebgcn_backbone'):
        LIRSEBGCN(
            6, 8, 2, _args('unknown'), torch.device('cpu')
        )


def test_lirs_ebgcn_metrics_match_project_hard_label_protocol():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = LIRSEBGCNTrainer._metrics(y_true, y_pred, val_loss=0.25)
    assert metrics['val_acc'] == accuracy_score(y_true, y_pred)
    assert metrics['val_auc'] == roc_auc_score(y_true, y_pred)
    assert metrics['val_f1'] == f1_score(y_true, y_pred, zero_division=0)
    assert metrics['val_loss'] == 0.25

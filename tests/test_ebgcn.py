from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from model.EBGCN import EBGCN, EBGCNResGCN
from model.EBGCN_ResGCN_StateAuxSameDiff import (
    EBGCNResGCNStateAuxSameDiff,
)
from model.EBGCN_BiGCN_StateAuxSameDiff import (
    EBGCNBiGCNStateAuxSameDiff,
)


def _args():
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
    )


def _batch():
    graphs = [
        Data(
            x=torch.randn(3, 6),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            directed_edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            node_state=torch.tensor([0, 0, 1], dtype=torch.long),
            edge_stance=torch.tensor([0, 1], dtype=torch.long),
            y=torch.tensor([0]),
        ),
        Data(
            x=torch.randn(2, 6),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            directed_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            node_state=torch.tensor([1, 1], dtype=torch.long),
            edge_stance=torch.tensor([0], dtype=torch.long),
            y=torch.tensor([1]),
        ),
    ]
    return Batch.from_data_list(graphs)


def _assert_forward(model):
    model.eval()
    out, td_edge_loss, bu_edge_loss = model(_batch())
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()
    assert td_edge_loss.ndim == 0 and torch.isfinite(td_edge_loss)
    assert bu_edge_loss.ndim == 0 and torch.isfinite(bu_edge_loss)


def test_ebgcn_forward():
    args = _args()
    _assert_forward(EBGCN(6, 8, 2, args, torch.device('cpu')))


def test_ebgcn_resgcn_forward():
    args = _args()
    _assert_forward(EBGCNResGCN(6, 8, 2, args, torch.device('cpu')))


def test_ebgcn_resgcn_dual_subgraph_forward():
    args = _args()
    model = EBGCNResGCNStateAuxSameDiff(6, 8, 2, args, torch.device('cpu'))
    _assert_forward(model)
    assert torch.isfinite(model.auxiliary_loss())


def test_ebgcn_bigcn_dual_subgraph_forward():
    args = _args()
    model = EBGCNBiGCNStateAuxSameDiff(6, 8, 2, args, torch.device('cpu'))
    _assert_forward(model)
    assert torch.isfinite(model.auxiliary_loss())

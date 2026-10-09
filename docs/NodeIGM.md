# NodeIGM rumor graph classifier

`model/NodeIGM.py` adapts the NodeIGM edge discriminator and environment
restoration mechanism to event-level rumor classification. It uses an
undirected GCN encoder and one symmetric score per propagation relation;
reverse edges and duplicate edges share a decision. It prefers
`directed_edge_index` when the dataset provides it.

Training fixes the selected evidence edges and restores environmental edges
at ratios `0`, `0.1`, and `1`, sampled separately within each event. The
auxiliary objective penalizes the mean and variance of pairwise graph
representation differences across layers and classifier logits. A per-event
soft keep-ratio penalty discourages selecting every edge. Hard forward edge
decisions use straight-through gradients; node participation has a
straight-through surrogate too, so an empty hard evidence view can still
train the discriminator. Dropout masks are shared between environment views.

## Isolated nodes

A node contributes to sum/mean readout only when it is incident to at least
one retained non-self propagation edge. Encoder self-loops do not count.
This also applies to the source node. If every edge is removed, the graph's
readout is zero and the output is determined by the classifier bias. Batch
positions are retained even for edgeless evidence graphs. Components with
edges participate even if they are disconnected from the source.

Nodes remain in the feature tensor. `extract_evidence_subgraph(data)` returns
the retained bidirectional edges, the original node IDs, and `pool_mask`.
This is an edge-level explanation; the scoring encoder still observes the
original graph, and this implementation does not enforce source-connected
paths or certify causal structure.

## Usage

```python
from types import SimpleNamespace
from model.NodeIGM import NodeIGM

args = SimpleNamespace(n_layers_conv=2, dropout=0.1, global_pool="mean",
                       nodeigm_env_weight=0.1, nodeigm_keep_ratio=0.7)
model = NodeIGM(in_feats=300, hid_feats=64, num_classes=2, args=args).to(device)
optimizer = model.init_optimizer()

model.train()
optimizer.zero_grad(set_to_none=True)
loss = model.compute_loss(batch.to(device))
loss.backward()
optimizer.step()

model.eval()
with torch.no_grad():
    log_probs, _, _, _ = model(batch)
    evidence = model.extract_evidence_subgraph(batch)
```

The four-output `forward`, `classification_loss`, zero `physics_loss`,
`auxiliary_loss`, and `set_epoch` methods fit `EINTrainer`'s model contract.
`main.py` dispatches `base_model: NodeIGM` through `EIN_NodeIGM_supervisor`.
The three dataset configs reuse the plain-GCN Word2Vec caches and ID splits,
with the same backbone hyperparameters and validation-loss checkpoint selection:

```bash
python main.py --config_filename configs/EIN/Pheme_NodeIGM_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_NodeIGM_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/DRWeibo_NodeIGM_word2vec.yaml --seed 0
```

Omit `--seed` to run seeds 0–4; append `--device cpu` for CPU execution.
Results go to
`experiments/EIN/<dataset>/nodeigm_gcn_evidence_undirected_valloss_word2vec/`.
The configs explicitly enable a 10-epoch environment-weight warmup. Their
NodeIGM settings are initial experiment settings, not tuned results.

Additional arguments: `nodeigm_edge_threshold=0.5`,
`nodeigm_mix_ratios=[0, 0.1, 1]`, `nodeigm_variance_weight=1`,
`nodeigm_keep_weight=0.01`, and `nodeigm_warmup_epochs=0`. A positive
`nodeigm_degree_threshold` enables the original hub-preservation rule; its
default is `0` (disabled), since source popularity is not automatically
evidence. `classification_class_weights` uses the existing project convention.

Inference retains the selector and classifies the evidence view. It needs no
labels and does not construct the training environments. Compared with the
original paper, this changes node-level consistency to graph-level consistency,
adds differentiable selection and masked pooling, and retains evidence
selection at inference. Standard classification supervision stays active;
optional warmup increases the environment weight instead of reproducing the
paper's loss schedule.

Validation: `python -m unittest discover -s tests -p 'test_nodeigm*.py' -v`.

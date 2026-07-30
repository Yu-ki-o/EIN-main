# EIN

This repository is the implementation of The Web Conference 2025 (WWW'25) paper: Epidemiology-informed Network for Robust Rumor Detection

![SNS](EIN.jpg)

run main.py to train and test the model.

## P2T3

The repository includes an EIN-compatible implementation of **P2T3:
Pre-Trained Propagation Tree Transformer**. It converts each propagation tree
to source, deep-conversation-chain, and shallow-conversation tokens, then adds
the released orthogonal chain identifiers, sinusoidal depth embeddings, and
conversation-type embeddings before Transformer encoding.

Ready-to-run Word2Vec configurations are available for the project datasets:

```bash
python scripts/pretrain_p2t3.py --config_filename configs/EIN/DRWeibo_P2T3_word2vec.yaml --device cuda:0
python main.py --config_filename configs/EIN/DRWeibo_P2T3_word2vec.yaml
python main.py --config_filename configs/EIN/Weibo_P2T3_word2vec.yaml
python main.py --config_filename configs/EIN/Twitter_P2T3_word2vec.yaml
python main.py --config_filename configs/EIN/Pheme_P2T3_word2vec.yaml
```

Put the downloaded unlabeled JSON files directly under
`dataset/UWeibo/dataset/raw` for Chinese experiments or
`dataset/UTwitter/dataset/raw` for English experiments. The native pre-training
command builds one shared Word2Vec encoder, caches conversation-chain tensors
under that dataset's `processed` directory, optimizes the released JSD
local-global MI objective, and saves the stable checkpoint configured by
`p2t3_pretrained_path`. Running the command again resumes that checkpoint.
The classifier is intentionally reinitialized during EIN fine-tuning.
`p2t3_unsup_weight` can additionally apply the MI objective during supervised
fine-tuning; it defaults to `0.0`, matching the released fine-tuning script.

## Requirements:
- python==3.12
- pytorch==2.3.1
- torch_geometric==2.5.3
- tqdm==4.66.4
- sklearn==1.5.0
- scipy==1.14.0
- numpy==1.26.4
- pandas==2.2.2
- jieba==0.42.1
- nltk==3.8.1
- gensim==4.3.2
- transformers==4.42.3
- yaml==0.2.5

## TCSR Prototype

This repository also includes a standalone prototype for **TCSR: Thresholded
Collective Stance Revision for Rumor Detection**.

Files:

- `model/model_tcsr.py`: modular TCSR model and `compute_tcsr_loss`
- `train_tcsr.py`: minimal multi-seed training script
- `utils_metrics.py`: acc/auc/f1 helpers matching the existing trainer

Each PyG `Data` object should contain:

- `data.x`: node text features, shape `[num_nodes, input_dim]`
- `data.edge_index`: propagation edges, shape `[2, num_edges]`
- `data.y`: graph label, shape `[1]` or scalar

Optional fields:

- `data.root_index`: root node index. Defaults to the first node of each graph.
- `data.depth`: node depth. If absent, TCSR computes it from `edge_index`.
- `data.stance_probs`: soft stance distribution `[num_nodes, 3]` in
  support/challenge/uncertain order.
- `data.stance_labels`: optional node stance labels for auxiliary supervision.

Backbone selection is controlled by `conv_type` in the TCSR config:

- `gcn`: lightweight PyG GCN encoder
- `gat`: lightweight PyG GAT encoder
- `bigcn`: BiGCN-style top-down + bottom-up propagation encoder
- `resgcn`: ResGCN-style residual graph encoder

GPU note: pass `--device cuda` to train on GPU. The training script moves each
PyG batch to the selected device, and the model keeps depth expansion,
aggregation, thresholding, and diagnostic scoring on that same device whenever
the input batch is on GPU.

Example with existing processed split directories:

```bash
python train_tcsr.py --dataset_dir data/Pheme --device cuda
```

Example with an EIN-style config file that builds dataset paths automatically:

```bash
python main.py --config_filename configs/EIN/DRWeibo_TCSR_word2vec.yaml
python main.py --config_filename configs/EIN/Weibo_TCSR_word2vec.yaml
python main.py --config_filename configs/EIN/Pheme_TCSR_word2vec.yaml
```

The same config files can also be run through `train_tcsr.py`, but `main.py`
is the preferred project-level entry point because it matches the existing
five-seed experiment and summary flow.

Example with explicit PyG `.pt` split files:

```bash
python train_tcsr.py \
  --train_path path/to/train.pt \
  --val_path path/to/val.pt \
  --test_path path/to/test.pt \
  --device cuda
```

Example with one `.pt` file and five seed re-splits:

```bash
python train_tcsr.py --data_path path/to/all_graphs.pt --seeds 0,1,2,3,4
```

Ablation flags are available with paired CLI switches:

```bash
python train_tcsr.py --dataset_dir data/Pheme --no-use_threshold --no-use_isolation
```

## LIRS-EBGCN

`LIRS_EBGCN` is an end-to-end, spuriosity-aware extension of EBGCN. It learns
a shortcut-biased node view, removes its projected component before Bayesian
edge inference, and regularizes the resulting graph representation with
biased infomax, HSIC, and online class-conditional spurious prototypes.

The same model supports both propagation backbones through one configuration
field:

```yaml
base_model: LIRS_EBGCN
lirs_ebgcn_backbone: bigcn  # choices: bigcn, resgcn
```

Ready-to-run Word2Vec configurations are provided for all project datasets:

```bash
python main.py --config_filename configs/EIN/Pheme_LIRS_EBGCN_word2vec.yaml
python main.py --config_filename configs/EIN/Weibo_LIRS_EBGCN_word2vec.yaml
python main.py --config_filename configs/EIN/DRWeibo_LIRS_EBGCN_word2vec.yaml
python main.py --config_filename configs/EIN/Twitter_LIRS_EBGCN_word2vec.yaml
```

To run the ResGCN variant, change only `lirs_ebgcn_backbone` to `resgcn` in
the selected file. The dataset cache and loader are selected automatically.

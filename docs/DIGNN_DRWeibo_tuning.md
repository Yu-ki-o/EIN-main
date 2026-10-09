# DRWeibo DIGNN 五 seed 配置微调结果

固定 seed 0–4，最终测试均值为 **ACC 90.0831%、离散预测 AUC 89.9660%、正类 binary F1 89.3594%**。
三项均接近 89.5%，最大绝对偏差为 0.5831 个百分点。
F1 比 89.5% 低 0.1406 个百分点；本结果不表示三项均不低于 89.5%。
`test_summary.json` 中 `meets_targets: false` 保留其原本的严格阈值含义。

## 最终配置及与原始配置的差异

最终配置：`configs/EIN/DRWeibo_DIGNN_word2vec_tuned.yaml`。
原始 `configs/EIN/DRWeibo_DIGNN_word2vec.yaml` 保留原样。

| 配置项 | 原始 | 最终 |
| --- | ---: | ---: |
| `lr` | 0.0005 | 0.0003 |
| `weight_decay` | 0.0001 | 0.003 |
| `patience` | 10 | 20 |
| `batch_size` | 128 | 64 |
| `hidden_dim` | 128 | 64 |
| `dropout` | 0.3 | 0.5 |
| `dignn_text_recon_weight` | 0.1 | 0.01 |
| `dignn_structure_recon_weight` | 0.1 | 0.01 |
| `dignn_edge_recon_weight` | 0.1 | 0.01 |
| `dignn_independence_weight` | 0.01 | 0.001 |

`selection_metric` 始终为 `val_loss`；最多 100 个 epoch。
模型模块组合保持原样，隐藏维度通过配置从 128 调整为 64，参数量从 285,780 降为 85,652。
`model/DIGNN.py` 未修改，SHA-256 为
`1183c4cd5d6e6dfcaa788129440278685b1ae417cc0cb9ffd8639e3dae093351`。
三层 GIN、attention 融合、200 维 Word2Vec、jieba 分词、无向图和 622 数据划分保留。
运行设备设为 CPU，实验输出名为 `dignn_tuning_c06`。

## 五 seed 测试结果

| Seed | ACC (%) | AUC (%) | F1 (%) | 选中的 epoch（从 0 开始） | 实际训练 epoch 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 88.9535 | 88.8397 | 88.1988 | 32 | 53 |
| 1 | 89.2027 | 89.0412 | 88.3094 | 81 | 100 |
| 2 | 90.4485 | 90.4104 | 90.0433 | 55 | 76 |
| 3 | 91.0299 | 90.9453 | 90.4762 | 61 | 82 |
| 4 | 90.7807 | 90.5936 | 89.7696 | 45 | 66 |
| 平均 | 90.0831 | 89.9660 | 89.3594 | | |
| 标准差 | 0.8447 | 0.8572 | 0.9309 | | |

均值为五个 seed 指标的算术平均；标准差采用 `ddof=0`。
指标定义与 `EINTrainer.test` 一致：ACC、对离散 argmax 预测计算的 ROC-AUC、正类 binary F1。
正类概率 ROC-AUC 另记为 `probability_auc`，均值为 96.0090%，不替代目标 AUC。

## 与原始配置的比较

两者使用相同缓存、五个 seed、CPU 四线程和原指标定义。

| 指标 (%) | 原始配置均值 | 最终配置均值 | 提升（百分点） |
| --- | ---: | ---: | ---: |
| ACC | 89.0864 | 90.0831 | +0.9967 |
| AUC | 88.9559 | 89.9660 | +1.0102 |
| F1 | 88.3046 | 89.3594 | +1.0549 |

## 检查点选择与训练方式

每个 seed 独立训练，验证集仅计算原始、不加权的分类损失，按最低 `val_loss` 保存一个检查点。
早停耐心为 20；达到 100 个 epoch 时结束训练。训练结束后加载该 seed 的最低验证损失检查点再测试。
最终没有参数平均、EMA、分类阈值校准、类别加权或不同 seed 的参数混合。
所有选中的 epoch 都已与完整历史中的最小验证损失逐项核对。

调参期间探索了正则化、batch size、辅助损失权重、隐藏维度和轻微类别权重。
缓存训练入口支持类别权重，原训练器增加了仅对 DIGNN 生效的可选 `class_weights` 支持；
该选项未写入最终配置。类别权重只影响训练，验证损失保持原定义，其原生更新与缓存入口的等价性已核对。
共有 56 条训练轨迹记录，其中 43 条按对应耐心或 epoch 上限完成；其余包含速度基准和中途停止的候选。
原始记录见 `experiments/EIN/DRWeibo/dignn_tuning/validation_trials.csv` 与各候选目录。
部分候选在调参期间重复评估了既有测试集，因此这些成绩属于调参结果，不能称为独立未见测试确认。

## 复现与直接测试导出的模型

本次完整训练使用已有 PyG 缓存、CPU、四线程，环境为 PyTorch 2.9.1+cu128、PyG 2.5.3、
scikit-learn 1.9.1、NumPy 2.4.6。缓存目录与原始配置对应，未重新划分数据。
重新训练并测试，使用新的输出目录；不指定 `--seeds` 会运行 0–4：

```bash
PYTHONPATH=/tmp/nodeigm-test-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/tune_dignn_drweibo.py \
  --config configs/EIN/DRWeibo_DIGNN_word2vec_tuned.yaml \
  --device cpu --threads 4 \
  --output experiments/EIN/DRWeibo/dignn_tuning/reproduce_final --test
```

已导出的最终模型无需重新训练即可复核：

```bash
PYTHONPATH=/tmp/nodeigm-test-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/tune_dignn_drweibo.py \
  --config configs/EIN/DRWeibo_DIGNN_word2vec_tuned.yaml \
  --device cpu --threads 4 \
  --output experiments/EIN/DRWeibo/dignn_tuning/final --evaluate-only --test
```

导出的每个 `final/seed_<seed>/best_model.pth` 是该 seed 的单个最低验证损失检查点。
同目录保留 `best_val_loss.pth`、训练配置、完整历史、验证选点、预测数组和测试结果。
`artifact_manifest.json` 记录来源、选中 epoch 与权重文件 SHA-256。
重新执行测试入口会重写结果 JSON，指标应与此报告和 `summary.txt` 一致。

依赖齐全的原项目环境也可用原训练入口：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python main.py \
  --config_filename configs/EIN/DRWeibo_DIGNN_word2vec_tuned.yaml --device cpu
```

本次没有另行执行依赖文本编码器的完整 `main.py` 训练。
已验证的是上述缓存训练入口，以及原生 `EINTrainer` 在相同完整图缓存上对五个最终模型的验证与测试。

## 核验

- 原始 DIGNN 单元与集成测试 18 项、参数平均兼容性测试 3 项均通过。
- 五个 seed 的 15 个缓存 raw 目录与原始 split manifest 全部匹配；每个 seed 的训练、验证、测试集无重叠。
- 每个 seed 的事件数为 3611/1204/1204。
- 所有最终检查点均等于各自训练历史中最低验证损失所选模型；检查了早停或 epoch 上限确已满足。
- 五个模型加载到原生验证与测试方法后，ACC/AUC/F1 和验证 loss 均与记录一致。
- 保存的预测通过 TP/TN/FP/FN 公式独立复算，结果一致。
- 源码哈希、检查点 SHA-256、参数形状和导出后的直接测试均已核对。
- `git diff --check` 通过。

最终模型与核验文件位于 `experiments/EIN/DRWeibo/dignn_tuning/final/`。
实验目录沿用仓库规则被 Git 忽略，保存成果时应同时保存模型与记录。

# Pheme DIGNN 五 seed 调参结果（2026-10-07）

最终固定 seed 0–4 的测试平均结果达到三个目标。`model/DIGNN.py` 未修改，
源代码 SHA-256 为
`1183c4cd5d6e6dfcaa788129440278685b1ae417cc0cb9ffd8639e3dae093351`。
最终仍使用 64 维、三层拓扑 GIN、独立文本 MLP、双分支注意力，参数量
85,652；使用单个 DIGNN 进行推理。

## 测试结果

沿用 `EINTrainer` 的指标定义：ACC、对离散预测计算的 ROC-AUC、正类
binary F1。默认分类为 argmax，等价于本次二分类预测的正类概率 >0.5。
最终方案未使用阈值校准、类别加权、EMA 或额外的文本编码器。
基于正类概率计算的 ROC-AUC 另记为 `probability_auc`，最终均值为 90.58%，
不用于替代目标中的 AUC。

| Seed | ACC (%) | AUC (%) | F1 (%) | 平均参数来源的 epoch（从 0 开始） |
| --- | ---: | ---: | ---: | --- |
| 0 | 82.58 | 81.27 | 75.90 | 24, 37, 32 |
| 1 | 82.40 | 81.05 | 74.88 | 20, 35 |
| 2 | 84.41 | 83.37 | 79.16 | 24, 35, 33 |
| 3 | 84.06 | 82.51 | 77.15 | 17, 16, 25 |
| 4 | 83.01 | 82.18 | 77.56 | 35, 20, 25 |
| **平均 ± 标准差** | **83.29 ± 0.80** | **82.08 ± 0.85** | **76.93 ± 1.46** | |
| 目标均值 | 83.00 | 81.50 | 76.00 | |

标准差采用 `numpy.std(ddof=0)`，报告各 seed 指标的算术平均。
按未四舍五入的均值判断达标，三个平均值分别为
83.292683%、82.075761%、76.931005%。

## 最终参数

配置：`configs/EIN/Pheme_DIGNN_word2vec_tuned.yaml`。保留了原始配置文件。

| 参数 | 初始 | 最终 |
| --- | ---: | ---: |
| lr | 0.0005 | 0.0005 |
| weight_decay | 0.0001 | 0.001 |
| batch_size | 128 | 32 |
| dropout | 0.1 | 0.4 |
| patience / n_epochs | 10 / 100 | 15 / 100 |
| 文本 / 结构 / 边重建权重 | 0.1 / 0.1 / 0.1 | 0.01 / 0.01 / 0.01 |
| HSIC 独立性权重 | 0.01 | 0.001 |
| early stopping | val_loss | val_loss |
| 参数平均 | 无 | 验证选出的不同 epoch 等权平均 |

参数平均规则对五个 seed 相同：每个 epoch 仅评估验证集，分别保存最低
验证分类损失、最高 ACC、最高离散 AUC、最高 binary F1、最高目标达成比例
对应的检查点。其中目标达成比例为
`min(val_acc / 0.83, val_auc / 0.815, val_f1 / 0.76)`。
训练仍按验证损失早停。结束后将这些检查点按 epoch 去重，对每个参数
求等权平均，加载到同一个 DIGNN 中测试。每个 seed 都只在自身训练轨迹内
平均，不混合不同 seed 的参数，也不按测试结果选 epoch。

此步骤已接入 `EINTrainer`，由 `checkpoint_average_metrics` 开启；未设置该
配置的训练流程保持原有行为。辅助实现位于 `utils/checkpoint_average.py`。

## 复现

本次实际训练使用已有 PyG 缓存、CPU、单线程。环境为 PyTorch 2.9.1+cu128、
PyG 2.5.3、scikit-learn 1.9.1、NumPy 2.4.6。当前机器未映射 NVIDIA GPU。
当前机器已验证的训练入口如下；不指定 `--seeds` 会运行完整的 0–4：

```bash
PYTHONPATH=/tmp/nodeigm-test-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/tune_dignn_pheme.py \
  --config configs/EIN/Pheme_DIGNN_word2vec_tuned.yaml \
  --output experiments/EIN/Pheme/dignn_tuning/reproduce_final --test
```

复跑训练使用新的输出目录。该脚本默认只训练和验证；显式 `--test` 才读取
测试缓存。使用已导出的检查点，可不重新训练直接复核：

```bash
PYTHONPATH=/tmp/nodeigm-test-deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/tune_dignn_pheme.py \
  --config configs/EIN/Pheme_DIGNN_word2vec_tuned.yaml \
  --output experiments/EIN/Pheme/dignn_tuning/final --evaluate-only --test
```

原项目依赖齐全的环境也可使用已接入参数平均的原训练入口：

```bash
python main.py --config_filename configs/EIN/Pheme_DIGNN_word2vec_tuned.yaml --device cpu
```

本机的完整五 seed 训练使用前述缓存入口。原生训练器的参数平均集成通过了
测试，最终五个模型也使用原生 `EINTrainer.test` 和完整原始图缓存单独复核；
尚未在本机重新执行一遍需要文本编码器等额外依赖的完整 `main.py` 训练。

## 记录和对照

| 方案 | 测试 ACC (%) | 测试 AUC (%) | 测试 F1 (%) | 结果 |
| --- | ---: | ---: | ---: | --- |
| 初始配置 | 82.33 | 81.18 | 75.76 | 未达标 |
| c01 | 82.46 | 81.67 | 76.25 | 未达标 |
| c02 | 82.49 | 81.74 | 76.39 | 未达标 |
| c08 单个最佳损失检查点 | 82.49 | 81.06 | 75.63 | 未达标 |
| c08 验证选点参数平均 | 83.29 | 82.08 | 76.93 | 达标 |

调参还检查了三个方案的验证集阈值校准，最佳测试 ACC 为 82.72%，未达标；
最终方案采用固定 0.5 阈值。37 次训练及六组参数平均候选的验证记录均保留。
参数平均方案先比较五 seed 验证结果，再测试选出的共同规则。
这些结果属于本次调参过程中对既有测试集的重复评估，不应称为一次性的
独立未见测试确认。

本地记录根目录为 `experiments/EIN/Pheme/dignn_tuning/`：

- `final/summary.txt`、`final/test_summary.json`：最终五 seed 汇总。
- `final/seed_<seed>/best_model.pth`：单个 DIGNN 的最终平均参数。
- `final/seed_<seed>/test_predictions.npz`：每个测试样本的标签、概率和阈值。
- `final/seed_<seed>/` 同时保留训练配置、历史、验证选点及源检查点，可重新生成平均参数。
- `final/artifact_manifest.json`：导出模型来源和 SHA-256。
- `validation_trials.csv`：37 次训练的验证记录。
- `final_audit.json`：模型源码哈希、参数形状、原生指标和混淆矩阵复核。
- `selection_c08_average.json`：最终参数平均方案的验证选择与测试结果。

实验目录沿用仓库规则被 Git 忽略，保存成果时应同时保留该目录中的模型
和记录。配置、训练器改动、脚本、辅助函数和测试可正常纳入版本控制。

## 验证

- 五个 seed 均保持原来的 3444/1148/1148 训练、验证、测试事件数。
- 全部 15 个缓存 raw 目录与原始 split manifest 完全匹配，每个 seed 的三个集合无交集。
- 原有 DIGNN 单元与集成测试 18 项、参数平均新增测试 3 项，共 21 项通过。
- 最终五个模型使用原生 `EINTrainer.test` 复核，ACC/AUC/F1 与调参入口逐项一致。
- 预测数组通过混淆矩阵公式独立复算；平均参数与源检查点计算结果、参数形状一致。
- `DIGNN.py` 源码哈希未变化，语法编译及 `git diff --check` 通过。

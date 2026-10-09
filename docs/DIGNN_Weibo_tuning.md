# Weibo DIGNN GPU 五 seed 微调结果

最终 seed 0–4 的 GPU 测试均值为 **ACC 95.6484%、离散预测 AUC 95.6442%、正类 binary F1 95.7223%**。
三项均接近 96%，最大绝对偏差为 0.3558 个百分点。
三项均值仍低于 96%；`test_summary.json` 的 `meets_targets: false` 保留“三项都不低于 96%”的严格含义。

## 配置与原始配置差异

已更新主配置 `configs/EIN/Weibo_DIGNN_word2vec.yaml`，并保存同内容副本
`configs/EIN/Weibo_DIGNN_word2vec_tuned.yaml`。
原始文件的字节备份为 `experiments/EIN/Weibo/dignn_tuning/original_config.yaml`，哈希与起始记录一致。

| 配置项 | 原始 | 最终 |
| --- | --- | --- |
| `device` | cuda | cuda:1 |
| `lr` | 0.0005 | 0.0003 |
| `weight_decay` | 1e-05 | 0.003 |
| `patience` | 10 | 20 |
| `batch_size` | 128 | 64 |
| `hidden_dim` | 128 | 64 |
| `dropout` | 0.3 | 0.5 |
| `dignn_text_recon_weight` | 0.1 | 0.01 |
| `dignn_structure_recon_weight` | 0.1 | 0.01 |
| `dignn_edge_recon_weight` | 0.1 | 0.01 |
| `dignn_independence_weight` | 0.01 | 0.001 |
| `result_name` | dignn_attention_hsic_undirected_valloss_word2vec | dignn_weibo_gpu_c02 |

`selection_metric: val_loss`、最多 100 epoch、三层 GIN、attention 融合、200 维 Word2Vec、jieba、
无向图、hop 72、622 划分和 seed 0–4 保留。
隐藏维度通过现有配置从 128 调为 64，模型参数量为 85,652（原始 285,780），模块组合保持原样。
`model/DIGNN.py` 未修改，SHA-256 为 `1183c4cd5d6e6dfcaa788129440278685b1ae417cc0cb9ffd8639e3dae093351`。
最终三个重构辅助损失权重均为 0.01，独立性权重为 0.001；最终仍训练全部既有辅助分支。

## 五 seed GPU 测试

**最终五个 seed 的训练、逐 epoch 验证、测试以及原生指标复核均在 `cuda:1` 上进行**，
GPU 为 NVIDIA A800 80GB PCIe。数据缓存读取和指标汇总使用 CPU；没有在 CPU 上进行本组模型训练或推理。

| Seed | ACC (%) | AUC (%) | F1 (%) | 选中 epoch（从 0 开始） | 训练 epoch 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 95.7128 | 95.7030 | 95.7983 | 22 | 43 |
| 1 | 95.2840 | 95.2822 | 95.3684 | 51 | 72 |
| 2 | 96.1415 | 96.1286 | 96.0265 | 19 | 40 |
| 3 | 95.6056 | 95.6141 | 95.6522 | 19 | 40 |
| 4 | 95.4984 | 95.4931 | 95.7661 | 14 | 35 |
| 平均 | 95.6484 | 95.6442 | 95.7223 | | |
| 标准差 | 0.2844 | 0.2803 | 0.2147 | | |

均值为五个 seed 指标的算术平均，标准差使用 `ddof=0`。
指标定义与项目 `EINTrainer` 一致：ACC、对离散 argmax 预测计算的 ROC-AUC、正类 binary F1。
概率 ROC-AUC 另记为 `probability_auc`，本组平均 99.0602%，不替代目标 AUC。
每个 seed 仅加载自身完整训练历史中最低验证分类损失的单个 checkpoint。
最终不使用参数平均、EMA、类别加权或分类阈值校准。

## 与原始 GPU 配置比较

两组沿用相同缓存、seed 和指标定义，均在 GPU 训练及测试。
原配置的 seed 0 在 cuda:0 训练，其余在 cuda:1 训练；原配置的五个测试均在 cuda:1 上完成。

| 指标 (%) | 原始 GPU 均值 | 最终 GPU 均值 | 提升（百分点） |
| --- | ---: | ---: | ---: |
| ACC | 95.2412 | 95.6484 | +0.4073 |
| AUC | 95.2255 | 95.6442 | +0.4187 |
| F1 | 95.3242 | 95.7223 | +0.3981 |

## 调参记录

选择 c02：其完整五 seed 测试均值与 96% 的最大偏差最小，且五个 seed 使用同一套超参数。
没有在不同 seed 之间混用候选配置、模型参数或检查点选择标准。
完整五 seed 候选结果如下，c03 仅有 seed 0 筛选结果，未纳入五 seed 排名。

| 候选 | ACC (%) | AUC (%) | F1 (%) | 与 96% 最大距离（百分点） |
| --- | ---: | ---: | ---: | ---: |
| baseline | 95.2412 | 95.2255 | 95.3242 | 0.7745 |
| c01 | 95.5413 | 95.5251 | 95.6169 | 0.4749 |
| c02 | 95.6484 | 95.6442 | 95.7223 | 0.3558 |
| c04 | 95.4341 | 95.4123 | 95.4950 | 0.5877 |
| c05 | 95.1768 | 95.1807 | 95.2369 | 0.8232 |

共有 26 条独立训练轨迹，其中 26 条按对应 patience 或 epoch 上限完整结束。
c04/c05 探索了将边重构损失权重设为 0；最终 c02 的边重构损失仍为 0.01。
配置位于 `configs/sweeps/weibo_dignn/`，完整历史和汇总位于实验目录及 `validation_trials.csv`。
调参过程中读取过原始及候选的测试结果；这些成绩属于使用测试反馈的调参结果，不能称为独立未见测试集确认。

## 原始数据重复事件与敏感性检查

沿用原有划分。原始 `3495745049431350.json` 与 `3495745049431351.json` 字节完全相同，
都对应事件 ID `3495745049431350`。源文件名在各划分间无重叠，但事件 ID 存在以下重复：
seed 1/2/3 的训练与测试各含这一个事件；seed 4 的训练与验证也各含它。
seed 0 的验证缓存写入时将同 ID 合并，因此有 932 条；其余验证缓存有 933 条。
五个训练缓存均为 2798 条、测试缓存均为 933 条。

保留原始 manifest，15 个 raw 目录的事件 ID 与原始 manifest 对应，缓存与 manifest 哈希保存在 `split_audit.json`。
另使用已产生的 GPU 预测，剔除 seed 1/2/3 测试集中与训练集重复的那一条事件后，均值为
ACC 95.6457%、AUC 95.6420%、F1 95.7167%。
该检查不重新训练，也不消除 seed 4 的训练/验证重复；结果不能描述为完全去重后独立重跑。
缓存顺序已用全部标签及节点数验证。

## 复现与模型导出

环境：PyTorch 2.9.1+cu128、PyG 2.5.3、NumPy 2.4.6、scikit-learn 1.9.1、FP32、确定性算法，TF32 未启用。
GPU 缓存入口拒绝 CPU、非 Weibo、非 val_loss、参数平均、EMA、阈值校准和 CPU 训练的检查点。
训练记录逐 epoch 保存训练/验证设备，测试记录保存真实训练设备与推理设备。

重新完整训练五个 seed，并测试；输出目录必须是新的：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=/tmp/nodeigm-test-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/tune_dignn_weibo.py --config configs/EIN/Weibo_DIGNN_word2vec_tuned.yaml --device cuda:1 --threads 4 --output experiments/EIN/Weibo/dignn_tuning/reproduce_final --test
```

直接加载已导出的模型进行 GPU 测试：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=/tmp/nodeigm-test-deps OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/tune_dignn_weibo.py --config configs/EIN/Weibo_DIGNN_word2vec_tuned.yaml --device cuda:1 --threads 4 --output experiments/EIN/Weibo/dignn_tuning/final --evaluate-only --test
```

成果目录：`experiments/EIN/Weibo/dignn_tuning/final/`。
每个 `seed_<seed>/best_model.pth` 是对应 `best_val_loss.pth` 的字节副本，
目录保留训练配置、全部 epoch 历史、验证选点、测试数组和测试 JSON。
根目录保留 `final_config.yaml`、`test_summary.json`、`summary.txt`、`artifact_manifest.json`、
`final_audit.json`、`split_audit.json`、`duplicate_event_sensitivity.json`。
实验成果沿用项目 Git 忽略规则；保存研究成果时应同时保存这些模型与记录。

本次使用上述完整缓存训练入口，未另行启动依赖其他文本编码器的完整 `main.py`。
实际原生 `EINTrainer` 的验证和测试方法已在 GPU 上对五个完整图缓存重跑，结果一致；
原生训练更新由现有模型集成测试覆盖。依赖齐全的原环境可用主配置和 `main.py` 训练。

## 核验

五个选中 epoch 均为全部训练历史中的最小 val_loss，所有训练都已正常早停或达 epoch 上限。
原生 GPU 指标与保存结果、独立 TP/TN/FP/FN 公式均一致。
检查点与源码哈希、完整配置、已导出模型的重新 GPU 测试、缓存顺序均已复核。
21 项模型/集成/参数平均兼容性测试通过，6 项 GPU/val_loss 协议拒绝检查通过；语法检查与 git diff --check 通过。
现有集成测试原先错误要求调参后的 DRWeibo 模型/优化器参数与 GCN 相同，
已保留数据和缓存兼容性断言，取消训练超参数必须与 GCN 相等的限制。

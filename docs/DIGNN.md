# DIGNN：传播结构与文本解耦的谣言检测模型

参考论文：Zhixun Li 等，*The Devil is in the Conflict: Disentangled
Information Graph Neural Networks for Fraud Detection*，ICDM 2022，
DOI: 10.1109/ICDM54844.2022.00131。实现位于 `model/DIGNN.py`。

论文把图的拓扑与属性分别编码，利用注意力决定每个样本对两个视图的
依赖程度，同时保留各视图的信息并约束跨视图依赖。本实现将该思想迁移到
当前项目的事件级谣言检测，并不是原论文节点级欺诈检测实验的严格复现。

## 从论文到传播树

| 论文设计 | 当前实现 |
| --- | --- |
| `MLP(A)` 编码固定大小的邻接矩阵行，式 (1) | 结构特征 + 结构专用 GIN，适配不同节点数量和未见过的事件 |
| `MLP(X)` 属性编码，式 (1) | 独立文本 MLP，不沿传播边混合文本 |
| 样本级加性注意力，式 (2)–(4) | 两个事件表示通过共享 `Linear → Tanh → Linear → softmax` 融合 |
| 节点级交叉熵，式 (5) | 事件级 NLL，输出图级谣言类别 |
| 保留各视图输入的信息，式 (7) | 文本 MSE、结构特征 MSE、有向边的采样 BCE 重建 |
| 变分互信息排斥项，式 (8) | 事件级归一化 RBF HSIC 独立性正则，属于替代目标 |

拓扑分支输入的六个结构特征为：`log(1 + 入度)`、`log(1 + 出度)`、
源帖标记、零入度标记、零出度标记、`log(1 + 事件节点数)`。
该分支只读取传播边、事件归属和源帖位置，不读取文本、谣言标签或立场。
GIN 在无向化的传播连接上聚合这些结构特征，使用残差和 LayerNorm；
原有方向通过入/出度特征和非对称边重建解码器保留。

文本分支仅对 `data.x` 做逐节点 MLP。每个分支独立将源帖表示与所有节点
表示的均值拼接，然后映射到相同维度的事件表示。跨视图注意力在事件层面
计算，而不是先把文本沿图传播后再拆分。

重建不会强制两个表示相同。其作用是抑制解耦过程中丢失结构或文本信息，
例如谣言文本被支持、质疑等不同语义的回复混合时造成的信息冲突。

## 损失与训练接口

```text
L = NLL(y_hat, y)
  + dignn_text_recon_weight      * L_text
  + dignn_structure_recon_weight * L_structure
  + dignn_edge_recon_weight      * L_edge
  + dignn_independence_weight    * L_HSIC
```

`L_text` 重建原始节点文本向量，`L_structure` 重建六维结构特征。
两个 MSE 先在每个事件内部平均，再在事件之间平均，防止大事件主导重建。
`L_edge` 在每个事件内部采样非边，排除已有边和自环；正/负项分别平均，
再在事件之间平均。它不构造整个批次或每个事件的稠密邻接矩阵。
无传播边的事件该项为零，完全连接的事件只使用可用的正边项。

`L_HSIC` 对批次中的配对事件表示使用 RBF 核，带宽是停止梯度的非零
距离中位数，核中心化后计算归一化 HSIC。其值在 `[0, 1]`。
HSIC 是依赖度正则，不是互信息估计值，也不实现原论文式 (8) 的变分上界。
批次少于三个事件时此项为零，因为两个事件的归一化中心核恒相似，无法
提供有用的解耦信号。常量视图的依赖项也为零，重建与分类承担防止坍塌的作用。
解耦过强可能移除与谣言类别相关的共享信息，因此默认权重较小。

兼容原有 `EINTrainer`：

```python
from model.DIGNN import DIGNN

model = DIGNN(in_feats=200, hidden_dim=128, num_classes=2, args=args)
log_probs, U, S, D = model(batch)
loss = model.classification_loss(log_probs, batch.y) + model.auxiliary_loss()
loss.backward()
# 或直接使用 model.compute_loss(batch)
```

`U/S/D` 为兼容接口的零占位，`physics_loss()` 返回零。
辅助项只在训练模式下计算；验证与测试使用分类损失选取检查点。
`get_diagnostics()` 返回停止梯度的 `view_attention [B, 2]` 和原始辅助损失，
注意力列顺序为拓扑、文本。共享训练器记录辅助损失总和。
`encode_views(data)` 可以取得各分支的节点和事件表示以进行分析。

## 输入与缓存

必要输入是 `x [N, in_feats]` 和 `edge_index [2, E]`。
存在 `directed_edge_index` 时优先使用它；旧缓存缺少该字段时使用
`edge_index`。若只有无向边，则不恢复已经丢失的方向信息。
重复边会合并，自环不作为传播边参与结构描述和重建。

支持 PyG `Data` 和 `Batch`。批次节点按事件连续排列，源帖默认是每个事件的
第一个节点，与当前缓存一致。也支持 `root_index` 或 `rootindex [B]`，
批处理后的值为整个批次中的全局节点索引。
允许单节点事件、无边图和含孤立节点的图；不接受零节点事件或跨事件边。

推理不需要 `y`、`user_state`、`node_state`、`edge_stance` 或时间戳。
原训练器仍读取 `user_state` 作为共享接口的参数，但模型不使用它。
三个配置复用原有 ResGCN/GCN Word2Vec 缓存、固定划分和文本编码器。
不修改缓存内容，不需要新增立场标注或预训练阶段。

## 运行与消融

```bash
python main.py --config_filename configs/EIN/Pheme_DIGNN_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_DIGNN_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/DRWeibo_DIGNN_word2vec.yaml --seed 0
```

去掉 `--seed` 运行 0–4 五个种子，添加 `--device cpu` 使用 CPU。
输出位于 `experiments/EIN/<dataset>/dignn_attention_hsic_undirected_valloss_word2vec/`。
这些是初始参数，未经完整数据集调参，不能据此声称优于现有模型。

| 配置 | 默认 | 用途 |
| --- | --- | --- |
| `dignn_num_layers` | 回退到 `n_layers_conv`，否则 3 | 结构 GIN 层数 |
| `dignn_fusion` | `attention` | `attention`、`mean`、`topology`、`text` |
| `dignn_text_recon_weight` | 0.1 | 文本重建 |
| `dignn_structure_recon_weight` | 0.1 | 结构特征重建 |
| `dignn_edge_recon_weight` | 0.1 | 边重建 |
| `dignn_independence_weight` | 0.01 | HSIC 解耦 |

把相应辅助权重设为零可以逐项消融。单分支分类消融可设置
`dignn_fusion: text` 或 `topology`，同时将 HSIC 与未使用分支的重建权重
设为零。`text` 分类分支的重建只保留 `dignn_text_recon_weight`；
`topology` 分支可保留结构和边重建。更改实验时同步修改 `result_name`，
以免不同实验共用输出目录。

## 验证

```bash
python -m unittest discover -s tests -p 'test_dignn*.py' -v
```

测试覆盖文本/拓扑输入及梯度隔离、图批处理、节点置换、单节点与无边事件、
完全图、负采样边界、各辅助项的梯度、HSIC 退化情形、无标签推理、
缓存与训练入口，以及三个配置下真实 `EINTrainer.train_epoch` 的参数更新。
入口测试模拟数据集和日志初始化，不需要加载文本编码器或完整数据集。

本次验证：18 项测试通过，语法编译与 `git diff --check` 通过。
另外，从三个数据集的 seed 0 真实训练缓存各取前 8 个事件，在 CPU 上
分别执行 3 次参数更新，损失与梯度均有限；移除标签和状态的推理结果
与原输入一致，保存/加载 `state_dict` 后预测一致。该检查没有使用验证集
或测试集，也未进行完整训练或报告分类性能。

| 数据集 | 抽取事件数 | 总训练损失：第 1 次 → 第 3 次更新 |
| --- | --- | --- |
| Pheme | 8 | 1.3131 → 0.9424 |
| Weibo | 8 | 1.5085 → 1.1935 |
| DRWeibo | 8 | 1.2928 → 1.0466 |

验证环境使用已有 PyTorch 2.9.1 + PyG 2.5.3 临时依赖目录：

```bash
PYTHONPATH=/tmp/nodeigm-test-deps /home/fzu/miniconda3/envs/torch-dl/bin/python -m unittest discover -s tests -p 'test_dignn*.py' -v
```

这条命令只适用于当前机器的测试环境；完整训练入口仍需 README 所列的
原项目依赖，例如文本编码器与 `torch_scatter`。

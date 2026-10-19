# DIGNN：支持反对能量场与文本双分支谣言检测

当前默认是 WL 拓扑 + [可微软话题属性分支](DIGNN_TOPICS.md)。
`dignn_text_mode: mean` 可恢复旧文本分支。WL 拓扑第一版实现与配置见
[DIGNN_WL.md](DIGNN_WL.md)。下文保留为旧 energy/structural 模式的说明。

参考论文：Zhixun Li 等，*The Devil is in the Conflict: Disentangled
Information Graph Neural Networks for Fraud Detection*，ICDM 2022，
DOI: 10.1109/ICDM54844.2022.00131。实现位于 `model/DIGNN.py`。

论文把图的拓扑与属性分别编码，利用注意力决定每个样本对两个视图的
依赖程度，同时保留各视图的信息并约束跨视图依赖。本实现将该思想迁移到
当前项目的事件级谣言检测，并不是原论文节点级欺诈检测实验的严格复现。

设置 `dignn_topology_features: energy` 可使用下述能量场模式；设置为 `structural` 可回到原六维
结构基线；它保留原参数形状，可用于加载旧 DIGNN 检查点。能量版的投影层和
结构解码器形状发生变化，需要重新训练，不能直接加载六维版的完整检查点。

## 能量场节点输入

实现位于同一文件的 `StanceEnergyField`。`0=support`、`1=deny` 表示回复对父帖
的关系，`-1=unknown`。节点场源是直接收到的两类回复数量的 `log1p`，
`b_v=[log(1+S_v), log(1+D_v)]`。缺失标签不默认支持，也不读取 `node_state`。

对每个耦合尺度 `tau`，固定一个有源关系能量：

```text
E(phi) = fidelity/2 * sum_v ||phi_v - b_v||^2
       + tau/2 * sum_(p,c) ||A_parent[r] phi_p - A_child[r] phi_c||^2
```

父/子、support/deny/unknown 各有一组可学习的正对角映射，共 12 个参数，
映射元素为 `exp(map_bound * tanh(parameter))`，初始化为 1。没有固定的符号
翻转或通道交换。固定映射时能量严格凸，展开非负投影梯度更新；按每个事件
单独计算 Hessian 上界并选择安全步长，保证拼批不会改变单图的场解。
默认三个尺度 `[0.25, 1.0, 4.0]`，每个尺度更新 8 步，这是近似场响应，不声称
8 步已经收敛。所有尺度并行沿稀疏边计算，不构造稠密邻接或矩阵逆。

固定描述量共 16 维：原六维结构量、双场源 2 维、根/支持父帖/反对父帖/未知
角色 one-hot 4 维、出边标注覆盖率、未知出边数量的 `log1p`、根深度的 `log1p`、
根可达标记。角色描述与“收到的回复”描述分别保留。

每个尺度再加入场值 2 维、源残差 2 维、源保真能量 1 维、按三类边及父/子
端点分组的局部关系能量 6 维，即 `16 + 11*K` 维，默认共 49 维。每条边的
关系能量均分给两端。随后使用现有残差 GIN、根与全节点均值读出及视图融合。
场值与能量不是概率或事实真假；能量是学习的关系兼容性代价。

内层能量迭代产生节点输入，外层事件分类损失训练关系映射。不会另加一个
将可训练总能量无限压低的损失。若开启结构重建，目标只取固定的 16 维描述量，
不重建可训练场值或能量。文本分支继续仅读取 `x`，拓扑节点输入不读取文本向量。
立场本身来自文本标注，两分支的统计独立性并不因此得到保证。

## 从论文到传播树

| 论文设计 | 当前实现 |
| --- | --- |
| `MLP(A)` 编码固定大小的邻接矩阵行，式 (1) | 能量场/结构描述量 + GIN，适配不同节点数量和未见过的事件 |
| `MLP(X)` 属性编码，式 (1) | 独立文本 MLP，不沿传播边混合文本 |
| 样本级加性注意力，式 (2)–(4) | 两个事件表示通过共享 `Linear → Tanh → Linear → softmax` 融合 |
| 节点级交叉熵，式 (5) | 事件级 NLL，输出图级谣言类别 |
| 保留各视图输入的信息，式 (7) | 文本 MSE、结构特征 MSE、有向边的采样 BCE 重建 |
| 变分互信息排斥项，式 (8) | 事件级归一化 RBF HSIC 独立性正则，属于替代目标 |

`structural` 基线输入的六个结构特征为：`log(1 + 入度)`、`log(1 + 出度)`、
源帖标记、零入度标记、零出度标记、`log(1 + 事件节点数)`。
该基线只读取传播边、事件归属和源帖位置，不读取文本、谣言标签或立场。
两种模式的 GIN 都在无向化的传播连接上聚合，使用残差和 LayerNorm；
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

`L_text` 重建原始节点文本向量；`L_structure` 在能量模式重建固定 16 维描述量，
在结构基线重建六维结构特征。
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
注意力列顺序为拓扑、文本。能量模式额外报告 `energy_initial_energy [B,K]`、
`energy_final_energy [B,K]`、`energy_label_coverage [B]`，验证和推理也可读取。
共享训练器记录辅助损失总和。
`encode_views(data)` 还暴露 `energy_source [N,2]`、`energy_fields [N,K,2]`、
`energy_residuals [N,K,2]`、`energy_source_energy [N,K]`、
`energy_relation_energy [N,K,3,2]`。最后两轴是 S/D/unknown 与父/子端点角色。

## 输入与缓存

必要输入是 `x [N, in_feats]` 和 `edge_index [2, E]`。
能量模式优先成对使用 `directed_edge_index` 和 `directed_edge_stance`。
若有旧 `edge_stance` 而缺少有向标签，则与其原 `edge_index` 配对，绝不与
另一个排序/长度的边张量混用。旧树按根最短距离恢复父子方向；不能定向的
断开分量、有环或多父回复会报错。可信有向缓存支持森林，根不可达节点的深度
明确标为未知。`structural` 基线保留任意图的原输入行为。

重复边与标签同步合并，已知标签可补充同边的 unknown；同边支持/反对冲突
会报错。自环同步移除。没有标签的边显式为 unknown，覆盖率诊断为零；此时
场源为零，仍可用固定结构/角色特征运行，但不会获得支持反对能量信息。

支持 PyG `Data` 和 `Batch`。批次节点按事件连续排列，源帖默认是每个事件的
第一个节点，与当前缓存一致。也支持 `root_index` 或 `rootindex [B]`，
批处理后的值为整个批次中的全局节点索引。
允许单节点事件、无边图和含孤立节点的图；不接受零节点事件或跨事件边。

推理不需要 `y`、`user_state`、`node_state` 或时间戳；能量信号需要已有边立场。
原训练器仍读取 `user_state` 作为共享接口的参数，但模型不使用它。
三个配置复用原有 ResGCN/GCN Word2Vec 缓存、固定划分和文本编码器。
复用已有立场标注，不修改缓存内容，不需要新增预训练阶段。

## 运行与消融

```bash
python main.py --config_filename configs/EIN/Pheme_DIGNN_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_DIGNN_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/DRWeibo_DIGNN_word2vec.yaml --seed 0
```

去掉 `--seed` 运行 0–4 五个种子，添加 `--device cpu` 使用 CPU。
输出位于 `experiments/EIN/<dataset>/dignn_attention_hsic_undirected_valloss_word2vec/`。
这些是初始参数，未经完整数据集调参，不能据此声称优于现有模型。
运行能量实验时应使用新的 `result_name`，以与原六维实验分开。

```yaml
dignn_topology_features: energy  # structural：旧六维消融/旧检查点
dignn_energy_scales: [0.25, 1.0, 4.0]
dignn_energy_steps: 8
dignn_energy_fidelity: 1.0
dignn_energy_map_bound: 0.5
dignn_energy_step_fraction: 0.9
result_name: dignn_stance_energy_s3_steps8
```

| 配置 | 默认 | 用途 |
| --- | --- | --- |
| `dignn_num_layers` | 回退到 `n_layers_conv`，否则 3 | 结构 GIN 层数 |
| `dignn_topology_features` | `energy` | `energy` 或 `structural` |
| `dignn_energy_scales` | `[0.25,1.0,4.0]` | 有限正数的非空序列，也可逗号分隔字符串 |
| `dignn_energy_steps` | 8 | 正整数，各尺度更新步数 |
| `dignn_energy_fidelity` | 1.0 | 正的场源保真系数 |
| `dignn_energy_map_bound` | 0.5 | 映射对数幅度界，范围 `[0,2]`；0 是固定单位映射 |
| `dignn_energy_step_fraction` | 0.9 | 安全上界步长的比例，范围 `(0,1]` |
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

能量版测试额外覆盖稠密解析解、已知七节点场/局部能量、可学习映射的下降性和
梯度、逐事件步长与拼批一致性、旧缓存方向恢复、立场对齐和冲突、缺失标签、
固定重建目标及训练更新。2026-10-12 在 Windows CPU 的 PyTorch 2.8.0、
PyG 2.6.1 下，30 项 DIGNN 测试通过，包括三个数据集配置下的实际
`EINTrainer.train_epoch` 参数更新；数据集和文本编码器初始化使用测试替身。
语法编译与 `git diff --check` 通过。没有完整训练或分类效果报告。

本次测试命令使用隔离的临时 PyG 依赖目录，未改动 Conda 环境包：

```powershell
$env:PYTHONPATH = 'C:\Users\admin\.codex\tmp\ein-dignn-energy-20261012'
$env:PYTHONUTF8 = '1'
& 'D:\Software\Anaconda\python.exe' -m unittest discover -s tests -p 'test_dignn*.py' -v
```

以下是原六维基线的历史验证记录。

历史测试覆盖文本/拓扑输入及梯度隔离、图批处理、节点置换、单节点与无边事件、
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

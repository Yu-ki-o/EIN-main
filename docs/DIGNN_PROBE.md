# DIGNN 结构探测系统

`model/DIGNN.py` 的默认拓扑分支改为 `StanceStructureProbe`，配置名为
`dignn_topology_features: probe`。它实现“立场条件的多尺度结构响应编码”：
从源帖注入单位信号，计算信号在不同关系算子和关系顺序下的响应，再编码响应分布。
不使用入度、出度、深度等手工量作为节点属性，也不读取节点文本、真假标签或节点状态。

现有文本编码、软话题模块、融合层、分类器及 EINTrainer 的四元组返回接口保持兼容。
旧的 `wl`、`energy`、`structural` 模式仍可显式选择。新拓扑参数与旧模型不同，
需要重新训练；旧检查点须使用原来的拓扑模式和超参数加载。

## 输入与算子

优先成对读取 `directed_edge_index` 和 `directed_edge_stance`：
边 `u → v` 表示 v 回复 u，`0=support`、`1=deny`、`-1=unknown`，标签相对于直接父帖。
旧缓存只把 `edge_stance` 与它原本对应的 `edge_index` 配对，再按源帖恢复树方向。
去除自环、排序和合并重复边时同步处理标签；重复边的已知标签冲突会报错。
当前沿用有根传播森林验证，拒绝跨事件边、多父节点和环。

设 `A_r[u,v]=1` 表示存在关系 r 的父→子边，列向量信号满足：

```text
Q_r = A_r.T - diag(A_r @ 1)
K_r(t) p = exp(t Q_r) p
p0 = e_root
```

`Q_r` 的每一列和为 0；信号沿父→子方向传播，叶节点保留收到的信号。
这里的流出量用于定义守恒算子，不作为节点属性输入神经网络。
保留讨论中的单位边速率，没有把出边归一化成总速率为 1 的离散随机游走。

全图算子 `Q_all` 包含所有边，包括未知标签边；`Q_s`、`Q_d` 分别仅使用支持、反对边。
反对不赋负权，不做“反对的反对等于支持”的根相对状态转换。

每个尺度 t 的六个通道按以下顺序存储：

| 索引 | 通道 | 计算 |
| --- | --- | --- |
| 0 | 全图 | `K_all(t) p0` |
| 1 | 支持 | `K_s(t) p0` |
| 2 | 反对 | `K_d(t) p0` |
| 3 | 先支持后反对 | `K_d(t) K_s(t) p0` |
| 4 | 先反对后支持 | `K_s(t) K_d(t) p0` |
| 5 | 顺序差异 | `channel_3 - channel_4` |

前五个通道非负且每个事件的总质量为 1，第六个通道允许负值、总和为 0。
组合通道的每一阶段都运行 t，所以总探测时间为 2t；这是原讨论的定义，
不能把它们与单通道 t 描述成相同总时间的探测。
全图响应允许信号经过多次支持/反对交替的路径，避免只探测两段关系造成完全不可达。

## 节点编码与事件读出

默认尺度为 `[0.25, 1.0, 4.0]`，得到 `R ∈ R^[N,3,6]`。
每个节点使用根脉冲值和展平后的响应，经两层 MLP 编码：

```text
h_v = MLP([p0(v), flatten(R_v)])
m_g = sum_{v in g} flatten(R_v)^2
z_topology = Readout([h_root, mean_{v in g}(h_v), m_g])
```

神经网络先做非线性变换，再做排列不变的读出；额外保留每个通道的二阶矩。
直接对原始响应求和会得到上述常数，无法表示结构差异，因此不采用这种读出。
源帖标记来自探测初始条件，不是度特征。新模式不再经过 WL 或 GIN。
`n_layers_conv` 仍用于旧拓扑模式，不控制 probe 的两层节点 MLP。

默认输出维度仍是 `[事件数, hidden_dim]`，沿用原注意力/均值/单分支融合接口。
probe 与 WL 一样禁用旧重建损失和 HSIC；现有 topic 辅助损失仍按原配置工作。

**分支边界：** probe 不读取 `x`，其可训练编码器能从分类损失获得梯度。
默认 `topics` 文本分支沿用原实现，话题交互本身仍会使用传播边和 stance。
若实验要求文本分支也仅依赖 `x`，设置 `dignn_text_mode: mean`；不能把默认 topics
组合声称为两路完全不共享结构信息。立场本身来自文本，亦不意味着统计独立。

## 数值实现、缓存和计算成本

固定探测响应在 CPU 上用 float64、SciPy `expm_multiply` 的稀疏矩阵指数作用计算，
不显式构建稠密 `N×N` 邻接或完整矩阵指数。需要 NumPy 和 SciPy >= 1.9（使用 `traceA`）；
本地验证环境为 PyTorch 2.6.0+cpu、PyG 2.7.0、SciPy 1.15.3。
模型接收的响应转换为节点编码器的 dtype/device；可训练的 MLP 和融合可在 GPU 上运行。
固定尺度和固定离线 stance 不参与梯度；这版没有可学习传播速率或可学习探测尺度。

每个事件独立求解，batch 中的其他事件不会改变响应。LRU 缓存键包括节点数、根位置、
有向边与标签、尺度及消融模式；不包含文本和真假标签，也不从训练集建立词表。
缓存只保存 CPU 数值响应，不保存计算图，不进入 `state_dict`；缓存失效不改变数值含义。
加载同长度的新尺度 buffer 后会重新计算。`cache_size=0` 可禁用缓存。

稀疏算子存储为 O(N+E)，响应存储为 O(NKC)，K 为尺度数、C 为通道数。
完整时间还依赖指数作用算法的迭代量、尺度和最大流出量，不能简单宣称总计算是固定的
O(N+E)。高分叉事件、大尺度和小缓存命中率可能让 CPU 探测成为训练瓶颈；需用真实数据
测吞吐。缓存按事件数量限额，默认 256；大图较多时可降低该值以控制内存。

## 配置与运行

Pheme、Weibo、DRWeibo 的普通 DIGNN 配置已启用 probe，结果名称改为
`dignn_<dataset>_probe_topics_v1`。已有 `_tuned.yaml` 文件继续显式使用原 energy 模式，
不把过去的调参结果冒称为新模型调参结果。

```yaml
dignn_topology_features: probe
dignn_probe_scales: [0.25, 1.0, 4.0]
dignn_probe_mode: ordered
dignn_probe_cache_size: 256
dignn_text_mode: topics
```

```bash
python main.py --config_filename configs/EIN/Pheme_DIGNN_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_DIGNN_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/DRWeibo_DIGNN_word2vec.yaml --seed 0
```

设备通过 `--device cpu` 或 `--device cuda:0` 覆盖，其他训练流程沿用项目入口。

三组探测消融可通过 `dignn_probe_mode` 切换：

| 配置 | 保留响应 | 用途 |
| --- | --- | --- |
| `untyped` | all | 全图无类型探测 |
| `typed` | all、s、d | 增加关系类型 |
| `ordered` | 全部六个通道 | 增加顺序组合和差异 |

这三个模式的输入维度不同，需要分别训练。比较时固定文本模式、优化参数与数据划分。

`model.encode_views(data)` 暴露 `probe_responses [N,K,C]`、`topology_nodes [N,H]`、
`topology_graph [B,H]`；通道名称见 `model.topology_encoder.channels`。
`model.get_diagnostics()` 返回停止梯度的 `probe_mass [B,K,C]`、
`probe_second_moment [B,K,C]`、`probe_label_coverage [B]` 及现有融合/topic 诊断。
这些响应是人工探测量，不是用户态度概率或谣言真实性分数。

## 验证与边界

```bash
python -X utf8 -m unittest discover -s tests -p "test_dignn*.py" -v
```

`-X utf8` 避免 Windows 默认 GBK 对项目中 UTF-8 源码的解码错误。
新增测试覆盖稀疏解与独立稠密矩阵指数对照、质量守恒、未知标签、交替长路径、
高分叉星形图解析解、单节点/孤立分量、节点置换、拼批、文本无泄漏、前后向及优化器更新、
旧缓存对齐、缓存淘汰、检查点恢复和尺度变更；原 WL/energy/structural/topic 与训练入口
也包含在同一回归命令中。CUDA 测试在 CUDA 不可用时跳过。

三条长度为二的分支，A 的序列为 `ss, sd, dd`，B 为 `sd, sd, ds`，
两图形状和逐层支持/反对数量相同。t=1 时顺序差异的平方和分别约为 0.266 和 0.331，
该机制例子已纳入测试。这不证明超越关系 GNN，也不证明谣言分类性能提高。

源帖单点探测不是完整图同构判别器；不连通分量收不到信号，其内部边结构不能由这次
根探测恢复。全图通道能走过未知边，但 S/D 单独及组合通道不能越过未知边，不能把缺失
标注当作完整关系观测。当前实现没有增加局部探针或更长的关系切换序列。
真实数据准确率、F1、效率和多种子稳定性需通过训练与消融实验验证。

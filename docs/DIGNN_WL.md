# DIGNN stance-WL 第一版

当前默认属性分支已升级为[可微软话题版本](DIGNN_TOPICS.md)。下文记录原 WL + mean 版本；
设置 `dignn_text_mode: mean` 可恢复此版本的前向路径和参数结构。

实现位于 `model/DIGNN.py`，默认 `dignn_topology_features: wl`。

## 编码流程

1. 对父节点到回复节点的有向传播边读取 stance：`0=支持`、`1=反对`、`-1=未知`。
   优先成对使用 `directed_edge_index` / `directed_edge_stance`。旧缓存的
   `edge_stance` 只与其对应的 `edge_index` 配对，然后根据根节点恢复方向。
   标签缺失时使用未知，不能默认支持。沿用传播森林检查；未知深度的孤立节点被保留。
2. `StanceWLTokenizer` 从根/普通节点标记开始，分别哈希入边和出边的
   `(stance, 邻居上轮标签)` 多重集，提取第 0 到 R 轮模式。
   每个 `(轮次, 模式)` 的出现次数构成稀疏图特征，每种出现的模式对应一个 token。
   WL 模式互相重叠，并非对原图做互斥子图划分。
3. token 输入为两个类型哈希嵌入、尺度嵌入及频次投影之和，再做 LayerNorm。
   频次投影输入为 `log1p(count)` 和 `count / 所有尺度的总出现次数`。
4. `SubstructureCombination` 在每张图内进行多头自注意力和残差 FFN，再学习
   行归一化的软分配矩阵 S。按 `S * mass` 聚合到组合 token：特征取加权平均，
   新 mass 取加权次数之和。下一层重新输入组合 mass 的对数和占比。
   全部层保持每张图的总 mass = `节点数 * (R + 1)`。
5. 最后一层使用带 mass 先验的注意力读出得到图向量。属性分支继续使用原来的
   节点 MLP、根节点与节点均值拼接读出。两分支经现有注意力融合后分类。

可微部分从嵌入开始，包括注意力、软分配、读出和融合；WL 标签细化和计数不参与反向传播。
软组合是特征组合，不保证对应原图的连通子图。本版不计算图间核矩阵，属于受图核启发的编码器。
本版采用纯图内自注意力，未加入讨论中的子结构连接矩阵/关系偏置。

## 哈希与归纳推断

采用确定性的 BLAKE2b 128 位摘要，先按完整摘要统计类型，避免 Python `hash()`
的进程随机性以及 batch 内临时编号。摘要的两个 64 位部分分别映射到两个可训练嵌入表。
这是一种固定的特征哈希近似，而非拟合全数据集词表；验证/测试样本不会改变编码规则、
增加参数或重新编号。新模式也可以推断，但两个桶同时碰撞仍可能导致类型嵌入相同，
未见模式的表示质量需要通过验证集检验。增大 `dignn_wl_buckets` 可减少这种风险。

CPU LRU 缓存只保存固定 WL 计数，键包含节点数、根节点、有向边及其 stance。
它不缓存学习表示，不包含文本或标签，不写入 state_dict；关闭缓存和重新加载检查点
均不改变编码。训练中的其他图不会改变一张图的 token 或组合槽位数。

## 损失与兼容接口

WL 第一版仅使用融合后的分类交叉熵。`auxiliary_loss()` 和 `physics_loss()` 为零；
不使用随机掩码、对比学习、KL、互信息、HSIC、结构重建或文本重建。
在 `wl` 模式下，旧配置中的四个重建/独立性权重即使非零也不生效；
`loss_weights` 会反映实际的全零权重。没有单独的拓扑分类头。

保留 `forward -> (log_probs, U, S, D)` 接口，后三项为训练器兼容占位；
`encode_views()` 返回 `topology_graph`、`text_graph`，WL 模式另有
`topology_tokens`、`topology_token_mass`、`topology_token_mask`、`wl_tokens`。
WL 模式不返回旧的节点级 `topology_nodes` 和 `structure_features`。

`dignn_topology_features: energy` / `structural` 可显式选择旧编码器及其旧损失。
新 WL 检查点与这两种模式不兼容，必须重新训练。

## 参数与运行

| 参数 | 默认值 | 含义 |
|---|---|---|
| `dignn_topology_features` | `wl` | 编码模式 |
| `dignn_wl_rounds` | 2 | 正整数，保留 0、1、2 轮模式 |
| `dignn_wl_buckets` | 8192 | 每个哈希嵌入表的桶数，至少 2 |
| `dignn_wl_heads` | 4 | 注意力头数，必须整除 hidden_dim |
| `dignn_wl_clusters` | `[64,32,16]` | 每层组合槽位上限，列表长度决定组合层数 |
| `dignn_wl_cache_size` | 256 | CPU 缓存图数，0 关闭 |
| `dignn_wl_log_attention` | false | 是否保留脱离梯度的完整注意力与分配矩阵 |

未设置 clusters 时，根据 `dignn_num_layers`（再回退到 `n_layers_conv`，默认 3）
生成从 64 开始逐层减半的上限，最少 1。每张图实际使用的槽位数是当前有效 token 数
与该层上限的较小值；默认配置是上限，不保证每层严格减半。

完整诊断矩阵可能很大，默认只记录各层有效 token 数和双分支权重。
初始注意力的计算量随 token 数平方增长；当前按 batch 最大 token 数 padding，
不截断子结构。大传播图应降低 batch_size 或 WL 轮数，完整训练前应测量实际显存。

三个非 tuned 的 DIGNN 配置已切换为 WL v1，并使用新的 result_name。
优化超参数沿用原值，尚未针对 WL 调参。三个 tuned 配置显式固定为以前的 energy
默认模式，避免历史配置因默认值变化而静默切换架构；重现历史检查点还需匹配其实际编码模式。

```powershell
python main.py --config_filename configs/EIN/Weibo_DIGNN_word2vec.yaml --device cuda:0 --seed 0
```

## 验证

`tests/test_dignn_wl.py` 检查 stance 模式计数、方向敏感性、节点置换/图顺序/批处理
不变性、padding、组合 mass 守恒、未知 stance、旧缓存对齐、文本与结构分离、
标签不进入特征、检查点重载，以及仅分类损失到注意力和软组合的实际梯度。
原 DIGNN 及 energy 测试保留；集成测试通过真实 EINTrainer 单步训练检查接口。

```powershell
python -X utf8 -m unittest discover -s tests -p 'test_dignn*.py' -v
```

这些检查不代表真实数据上的分类效果，完整训练与超参数选择仍需实验。

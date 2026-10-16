# DIGNN stance-WL：第一版与第二版

实现位于 `model/DIGNN.py`，默认 `dignn_topology_features: wl`。
三个普通 DIGNN 配置当前启用第二版信息瓶颈；`dignn_ib_enabled: false` 恢复第一版。
构造模型时未提供该开关仍按第一版处理，兼容旧配置和第一版检查点。

## 第二版：拓扑变分信息瓶颈

在 WL 层级组合和图级读出之后，将拓扑向量 h 输入 `TopologyInformationBottleneck`：

```text
mu = Linear_mean(h)
log_variance = clamp(Linear_log_variance(h), -10, 6)
q(z|T) = Normal(mu, diag(exp(log_variance)))
训练：z = mu + exp(0.5 * log_variance) * epsilon，epsilon ~ Normal(0, I)
验证/测试：z = mu
```

T 是拓扑分支实际观察到的 stance-WL 描述，并非含全部文本属性的原始图。
均值映射以单位矩阵初始化，log_variance 映射以零权重、-2 偏置初始化。
方差范围限制用于数值稳定。最终融合与独立拓扑分类头都只使用瓶颈后的 z，
没有从瓶颈前的 h 绕过压缩直接参与预测的路径。

训练目标是变分形式的信息瓶颈代理目标：

```text
L = CE_fusion + alpha * CE_topology + beta(epoch) * KL
KL = mean_graphs [0.5 * sum_latent(mu^2 + exp(log_variance) - 1 - log_variance)]
```

KL 单位为 nats：潜变量维度求和，图样本维度平均，不除以 hidden_dim。
其期望给出 I(Z;T) 的上界；拓扑分类交叉熵用于保留标签预测信息。
这不是直接的互信息数值估计，也不保证找到最小充分表示或提高分类准确率。
辅助分类头防止最终分类仅依赖属性分支而使拓扑表示缺少监督。
若配置了 `class_weights`，拓扑辅助分类采用同样的正类别权重；默认普通交叉熵。

| 参数 | 模型缺省值 | 作用 |
|---|---|---|
| `dignn_ib_enabled` | false | true 启用第二版，仅支持 wl 模式 |
| `dignn_ib_beta` | 0.001 | KL 的最终权重 |
| `dignn_ib_classification_weight` | 0.1 | 独立拓扑分类交叉熵权重 |
| `dignn_ib_warmup_epochs` | 10 | KL 线性预热轮数，0 表示从开始即使用完整 beta |

普通配置显式设置 enabled=true。EINTrainer 每轮调用 `set_epoch(epoch)`，
beta(epoch)=beta*min(1, epoch/warmup_epochs)。第 1 轮为 beta/10，第 10 轮达到 beta。
自定义训练循环也应调用 `set_epoch`；未调用时按第 1 轮权重计算。
预热轮次由训练循环管理，不保存在模型 state_dict 中；恢复训练应传入实际 epoch。

`forward` 先计算不依赖标签的编码和融合预测，再在训练模式读取 `data.y` 计算辅助分类。
当辅助分类权重大于 0 时，训练 batch 必须提供每图一个标签。`encode_views()` 不读取标签，
验证/测试可以不提供标签，不采样，也不施加辅助损失。
`classification_loss()` 仍是融合分类损失；`auxiliary_loss()` 返回 alpha*CE_topology+beta*KL，
训练器将它加一次。验证的 val_loss 仍是融合分类损失。
第二版没有随机掩码、对比学习、HSIC 或重建损失。

`encode_views()` 新增 `topology_graph_pre_ib`、`ib_mean`、`ib_log_variance`、`ib_kl`、
`ib_logits`；`topology_graph` 为训练采样的 z 或评估均值。
`get_diagnostics()` 及训练器诊断日志包含 `ib_kl`、`ib_classification`、`ib_kl_weight`、
`ib_posterior_std`，其中分类项仅训练时记录。

消融：

- `dignn_ib_enabled: false`：完全恢复第一版，移除随机编码器和辅助分类头，兼容第一版 state_dict。
- `dignn_ib_beta: 0`：仅移除压缩惩罚，仍保留随机编码与拓扑辅助分类，不等于第一版。
- `dignn_ib_classification_weight: 0`：移除独立拓扑分类监督，保留融合分类和 KL。

第二版新增参数，不能直接严格加载第一版或 energy/structural 检查点。应使用匹配配置训练。
普通配置已使用 `dignn_<dataset>_wl_v2_ib` 独立结果目录；消融时也应设置独立 result_name。
目前系数是初始设置，未经过完整数据训练调参。

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

## 第一版损失与兼容接口

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

三个非 tuned 的 DIGNN 配置已切换为 WL v2，并使用新的 result_name。
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
`tests/test_dignn_ib.py` 另检查 KL 与 torch.distributions 的解析结果一致、训练采样与
评估确定性、预热系数、损失相加一次、两个辅助目标的梯度路径、目标不进入特征、
缺失训练标签报错、检查点重载、禁用 IB 恢复第一版，以及训练器实际读取诊断项。

```powershell
python -X utf8 -m unittest discover -s tests -p 'test_dignn*.py' -v
```

这些检查不代表真实数据上的分类效果，完整训练与超参数选择仍需实验。

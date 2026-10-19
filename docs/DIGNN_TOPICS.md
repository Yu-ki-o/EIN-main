# DIGNN 可微软话题第一版

`model/DIGNN.py` 的 WL 默认模型现为：立场 WL 拓扑编码 + 源帖条件化软话题 + 话题关系交互。
本版不新增话题内 WL 模式统计，不需要 LLM 调用或新的离线话题标签。

## 编码和交互

1. 原 WL 拓扑编码器保持不变，不读取节点文本。属性编码器仍是逐节点 MLP。
2. 每个事件的源帖单独保留。评论分配到最多 K 个槽位；实际容量是 `min(K, 评论数)`。
   由可学习槽位、源帖和评论均值初始化事件槽位。归一化 query/key 的相似度经 softmax 得到
   S，按 S 加权平均得到话题表示，中间用 GRU 更新槽位。最后一轮直接使用加权平均表示。
   整个分配和聚合过程可求导，不用 argmax 或离散聚类。根节点对应的 S 行为零。
3. 以源帖为独立 anchor，构造 K+1 个节点。话题自注意力学习语义关联；节点质量作为 attention
   的先验。源帖与话题的拼接、绝对差、逐元素乘积输入关系 MLP，再做注意力读出。
   读出使用簇大小作为特征，不按源帖相似度或簇大小硬删除评论。
4. 使用对齐、去重、定向后的原传播边，按支持/反对/未知三类计算
   `B_r = Sbar.T @ A_r @ Sbar`。Sbar 的源帖行是严格的 anchor one-hot。
   不构造 N*N 邻接矩阵或 E*K*K 边外积；先沿边累加软归属，再做矩阵乘法。
5. 对每类 B 分别进行沿出边读取回复、沿入边读取父帖的消息聚合。关系行归一化，
   六组独立线性映射保留方向和类型；另用各类入/出软边计数的 log1p 保留规模信息。
   更新后的话题再进行源帖关系读出。
6. 原双视图注意力融合全局 WL 表示与话题交互表示，输出分类结果。
   此时拓扑编码器仍不直接读取文本；话题交互表示同时依赖文本和传播关系，不能称为完全独立双分支。

无评论事件用源帖双份拼接读出，不创建虚假话题。无边图、孤立节点和有向森林沿用原输入规则。
未知立场独立成类；父帖相对的反对不解释为反对源帖。没有父帖文本上下文补全或显式源帖立场预测。

## 损失

`L = L_class + 0.05 * L_topic_semantic + 0.01 * L_topic_diversity`

- 语义约束使用停止梯度的原始输入 x，非零向量余弦相似度 >=0.7 的节点对视为高置信正对，
  <=0.2 的视为负对，中间区域不监督。用 S 两行内积作为软共簇概率，做 BCE。
  小事件枚举所有无序对，大事件最多均匀采样 pair_budget 个对；正负组分别均值后取平均。
  这是固定文本特征的启发式伪监督，不是人工话题标签；尤其 Word2Vec 需要检查阈值适用性。
- 多样性项惩罚不同槽位归属列的余弦重叠，按两簇质量乘积加权。它抑制重复的活跃槽位，
  不强制各簇大小均衡，也不保证消除单簇坍塌。语义约束和诊断需要结合使用。
- 两个辅助项按事件平均，无评论、单评论或 K=1 时为零。验证和推理不计算辅助项。
  全部 topic 权重为零时不采样语义对。单拓扑消融自动关闭话题辅助项。
- WL 模式仍不使用旧版文本/结构/边重建和 HSIC 损失；energy/structural 保留原行为。

## 参数和消融

三个非 tuned 的 DIGNN 配置已显式开启 topics，result_name 改为 `dignn_<dataset>_wl_topics_v1`。
未改变数据缓存、数据划分、文本特征和原训练优化超参数。tuned 配置仍是旧 energy 模式。

| 配置 | 默认 | 含义 |
|---|---|---|
| dignn_text_mode | WL: topics；其余: mean | topics 或旧版 mean；topics 第一版仅支持 WL |
| dignn_topic_clusters | 4 | 每事件话题槽位上限 |
| dignn_topic_steps | 3 | 分配轮数 |
| dignn_topic_temperature | 0.5 | 正的 softmax 温度 |
| dignn_topic_semantic_relations | true | 话题和源帖节点间语义自注意力 |
| dignn_topic_propagation | true | 三类双向话题传播交互 |
| dignn_topic_semantic_weight | 0.05 | 固定输入语义伪监督权重 |
| dignn_topic_diversity_weight | 0.01 | 归属列防重复权重 |
| dignn_topic_pair_budget | 256 | 每事件语义节点对采样上限 |
| dignn_topic_log_assignments | false | 是否保留完整软分配与软边诊断 |

旧 WL 消融/检查点：设 `dignn_text_mode: mean`，原模型参数结构和键保持兼容。
新 topics 模型需要重新训练，不能严格加载旧 mean 检查点。
设置 propagation=false 去除话题传播；再设置 semantic_relations=false，可仅保留源帖关系读出。
这两项不会去除全局 WL 分支；完整去拓扑对照要同时使用 `dignn_fusion: text` 和 propagation=false。
单拓扑对照使用 `dignn_fusion: topology`。新实验应独立命名结果目录。

## 接口和诊断

`forward -> (log_probs, U, S, D)` 和现有 EINTrainer 接口不变，后三项仍是兼容占位。
`encode_views` 的 `text_graph` 是传播交互之前的纯语义表示；`topic_graph` 是用于最终融合的交互表示。
`topic_assignment [N,K]`（root 行零）、`topic_edges [B,3,K+1,K+1]`、`topic_mass [B,K]`
可用于梯度检查或分析；无效槽位严格为零。`topic_mask` 只表示容量，不是学到的活跃话题标签。
源帖列/行始终是 coarse graph 的 0；三类顺序支持、反对、未知；行父帖、列回复。

常规 diagnostics 保存脱离梯度的 mass、mask、分配熵、读出权重、有效簇数。
有效簇数是 `exp(H(归一化簇质量))`，源帖单节点事件为 0；它是占用分布指标，不是真实话题数。
开启 log_assignments 时额外保存 S 和 B，训练大图时注意内存。
话题仅为潜在语义簇，不给支持/反对、真实性或证据可信度等强语义解释。

## 验证

```powershell
$env:PYTHONPATH = 'C:\Users\admin\.codex\tmp\ein-dignn-energy-20261012'
$env:PYTHONUTF8 = '1'
& 'D:\Software\Anaconda\python.exe' -X utf8 -m unittest discover -s tests -p 'test_dignn*.py' -v
```

新测试覆盖分类损失对聚类参数的梯度、typed coarse graph 与稠密参考一致、质量守恒、
源帖位置和节点/事件置换不变性、拼批一致性、无评论/无边/森林、输入隔离、语义辅助目标、
旧模式/消融、检查点往返，以及现有训练器在三个配置下的实际单步优化。
完整数据集训练和性能提升尚未验证。

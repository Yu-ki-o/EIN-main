# StancePath：立场响应轨迹与预测充分性约束下的随机路径压缩

本实现对应“谣言检测创新点设计”对话最后整合的方案，研究名称暂记为
**StancePath**。模型文件为 `model/StancePath.py`，两阶段训练器为
`trainer/StancePath_trainer.py`，统一入口为 `main.py`。

## 建模与实现对应

1. **真实回复路径**：从原始父→子边提取所有根到叶路径，不按 stance 重连节点，
   不对路径数或长度做隐式截断。支持=0、反对=1，均相对于直接父帖。
   相邻两次反对不会被转换为对源帖的支持。
2. **路径内状态演化**：固定文本向量经可训练线性投影后，将源帖、父帖、当前回复及
   stance embedding 输入交互编码器。条件变化函数读取前序状态、当前交互和相邻
   stance 对的 embedding，乘 sigmoid 更新门后，以残差形式更新状态。
   首条边使用独立 START 标记。按树深度并行计算，同一个公共前缀只编码一次。
3. **轨迹 token**：拼接当前状态与状态增量，经线性变换后加入原图深度的正弦位置编码。
   深度表示回复层级，不表示真实时间。
4. **随机选路**：对每条路径的非源帖 token 做注意力池化；所有路径摘要均值形成
   排列不敏感的集合概览。选择器用路径摘要与概览预测 `q_i`。
   训练时使用 Binary Concrete：

   ```text
   u_i ~ Uniform(0,1)
   m_i = sigmoid((logit(q_i) + logit(u_i)) / temperature)
   ```

   推理时 `m_i = 1[q_i >= threshold]`，没有固定 top-k 或至少选一条的补救规则。
   未选出任何路径时，使用始终保留的源帖完成分类。
5. **路径并集去重**：每个原始节点只有一个 token。训练时节点门为
   `w_v = 1 - product_{i: v in P_i}(1-m_i)`，源帖门固定为 1；二值门时这就是
   所选路径的节点并集。连续门时它是可微的并集近似。
6. **选后交互**：零门节点先移除，再对保留节点做多头注意力。连续训练门通过
   `log(w_v)` 作用于注意力的 key 权重以及最终池化权重。注意力可加入深度差、
   祖先关系、是否同属一级分支的偏置，并用门控残差更新 token。
   分类器只接收这些 token 的池化结果，选择器集合概览不直接进入分类器。

这里没有先让全部路径交互、再丢弃输出。离散推理中，被删路径的独有内容无法通过
注意力或池化进入分类结果；选择器本身仍允许观察全部候选路径。训练是连续松弛，
低权重路径仍可能有小的贡献，不能将其解释为训练时已经完全删除。

## 预测充分性与压缩目标

每个事件计算：

```text
CE_sel = CrossEntropy(student(selected input), y)
CE_teacher = CrossEntropy(frozen_teacher(full input), y)
L_suff = relu(CE_sel - stop_gradient(CE_teacher) - epsilon)
L_mask = sum_i [q_i*log(q_i/rho) + (1-q_i)*log((1-q_i)/(1-rho))]
L = CE_sel + lambda*L_suff + beta*L_mask
```

先逐事件计算铰链与 KL，再对 batch 取均值，不能先平均 CE 再取铰链。
默认 KL 按路径求和，对应对话中的公式。`stance_path_kl_reduction: mean` 可用于
消融，但会改变长路径集合的正则尺度。KL 是对 Bernoulli `q_i` 和稀疏先验 `rho`
计算的，不对一次采样得到的连续 `m_i` 计算。期望保留路径数记录为 `sum(q_i)`。

该约束限制学生相对于完整教师的预测损失，是软正则，不能保证每个事件都满足约束，
也不是严格的因果充分性或跨域不变性证明。

## 两阶段训练

- **教师阶段**：使用全部路径，优化分类 CE；按验证集指标选择教师 checkpoint。
- **学生阶段**：用选出的教师初始化学生，然后冻结教师并固定为 eval 模式；
  学生优化上述总目标，轨迹编码器、交互层和分类器继续学习。
  学生验证和最终测试均使用确定性的离散路径选择。

两阶段各自的 epoch 上限、patience 和学习率独立配置。训练时不使用测试集选模型。
每个优化 batch 按 `stance_path_micro_batch_size` 分段反向传播，默认 1，
避免同时保留多个事件的全局注意力计算图。

最终 `best_model.pth` 只包含学生参数、模型配置和选择元数据，不包含教师参数。
恢复推理：

```python
from trainer.StancePath_trainer import load_stance_path_checkpoint

model, metadata = load_stance_path_checkpoint("path/to/best_model.pth", device="cpu")
logits = model(data)  # 单个 PyG Data、PyG Batch 或事件列表；无需 y 或教师
```

## 三个数据集运行

```bash
python main.py --config_filename configs/EIN/Pheme_StancePath_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_StancePath_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/DRWeibo_StancePath_word2vec.yaml --seed 0
```

省略 `--seed` 会运行项目原有的 0–4 五个种子；可加 `--device cpu`。
配置使用原有 200 维 Word2Vec、6/2/2 划分和对应的数据缓存键。
本模型与 directed ResGCN/SHPA 缓存兼容；缓存齐全时无需加载文本编码器。
结果写入：

```text
experiments/EIN/<dataset>/stance_path_sufficiency_valloss_word2vec/seed_<seed>/
  best_teacher.pth
  best_model.pth
  history.json
  test_results.json
  run.log
```

`test_results.json` 分别记录稀疏学生、同一学生的源帖单独输入对照，以及完整教师。
包括准确率、F1、路径保留比例、唯一节点保留比例（含源帖）、源帖独立预测比例、
期望/实际每事件路径数，以及包含候选轨迹计算和设备传输的推理耗时。
`auc` 沿用本项目的硬预测 AUC，另有 `auc_probability` 记录概率 AUC。
这些单次 wall-clock 耗时不是经过预热、多次重复的严格性能基准。

## 输入要求与超参数

- `x: [N, in_feats]`：预计算固定文本特征，模型内部 detach，投影层仍可训练。
- `directed_edge_index: [2,E]`：原始父→子关系；对应 `directed_edge_stance: [E]`。
  兼容只有 `directed_edge_index`、`edge_index` 和 `edge_stance` 的旧缓存：
  按 `edge_index` 的有向端点 `(父节点, 子节点)` 查找标签，重新对齐到原始回复边。
  两套边的顺序、数量可以不同；不会直接按位置拼接标签，也不会改变回复树。
  已存在的 `directed_edge_stance` 优先使用；它长度错误或标签非法时会明确报错。
  若没有 directed 字段，也接受已经是有向树的 `edge_index`/`edge_stance`。
- 源帖默认是节点 0，可用单个本地索引 `root_index` 或 `rootindex` 指定。
- 缺失 stance、冲突标签、多父节点、环或断连均报错；不静默补边或猜测 stance。
  缺失的离线标签应先按项目的 `stance_detection_cn.py` / `stance_detection_en.py`
  流程准备，并在需要时重建缓存。
- 训练/指标计算需要每事件一个 `y`；模型预测本身不依赖标签。
- 预设 `rho=0.2`、`beta=0.001`、`lambda=1.0`、`epsilon=0.05`、温度 `0.67`、
  推理阈值 `0.5` 是未调优的研究起点，阈值和其他参数应只用训练/验证数据调整。

相关消融开关：`stance_path_use_stance: false` 去掉 stance；
`stance_path_interaction_layers: 0` 去掉跨路径交互；
`stance_path_sufficiency_weight: 0` 去掉充分性项；
`stance_path_kl_weight: 0` 去掉压缩正则；完整输入学生对照可以同时设置
`stance_path_use_selection: false`、充分性权重与 KL 权重为 0。

所有候选轨迹仍要编码，训练连续松弛也仍处理全部非零门节点；
离散推理的节省主要发生在后续全局注意力，其复杂度随保留的唯一节点数平方增长。
这里不隐式截断大事件，也不承诺整个模型的耗时按路径保留比例下降。

## 验证

```bash
python -m pytest tests/test_stance_path.py -q
```

测试覆盖公共前缀去重、原始局部 stance、被删分支隔离、空选择、源帖单节点、
节点重编号不变性、批处理、随机门控梯度、逐事件充分性铰链、教师无梯度、
Bernoulli KL、三个数据集配置及缓存、两阶段训练与学生 checkpoint 独立恢复。
未安装 PyG 时仅 PyG Batch 测试跳过，其余测试仍运行实际 PyTorch 模型和训练器。
旧缓存回归测试覆盖边重排、无向边副本、标签缺失/冲突、错误长度，以及旧缓存的
两阶段训练与 checkpoint 恢复。

# KPG 适配说明

实现集中在 `model/kpg.py`，入口是 `main.py -> EIN_KPG_supervisor -> KPGTrainer`。
来源：[作者仓库](https://github.com/kkkkk001/KPG)，固定版本
`7b41f6647fba7f23b8d461a85df33bdf4c699ce8`。参考文件为 `generators.py`、
`KPG_main_final.py`、`utils.py` 和 `train_gcn.py`。

这是 KPG 的连续帖子特征适配版，不是论文完整文本管线的等价复现。

## 保留和调整

* 原始 GRU token CVAE 保留为 `TokenEncoder/TokenDecoder/TokenCVAE`，不在默认
  训练链路中使用。只调整类名、设备归属和 None 比较，没有复制作者的全局随机种子副作用。
* 当前数据只有 word2vec/E5 等帖子向量，无作者的词 ID 序列及 5000 维词频空间。
  因此默认 `CVAE` 在父帖、根帖条件下重构回复向量，用 MSE 替换词级 CE，保留高斯
  后验、重参数采样、KL 和标准高斯先验采样。没有伪造 token 或把 embedding 当词 ID。
* 保留作者的两路两层 TD/BU GCN、均值池化和拼接分类；ENS 保留两层 GCN、
  父节点表示拼接、候选评分及 epsilon 局部/全局搜索。全局节点的父节点未选中时仍接到根。
* 用紧凑的逐事件张量替换作者的固定 padding 缓冲区。候选概率按事件归一化；
  padding 不参与图池化；eval 关闭 dropout。局部无候选时退回全局候选。
* 保留 modified rollout：当前类别概率加未来步骤的平均概率；没有可用后续节点时
  提前结束 rollout，不补造空动作。rollout 不调用 CRG。
* ENS 使用作者主脚本的 `-log p(action) * exp(-delta_reward) * (1.5-p_y)`；
  训练中不接受参考类别奖励下降的动作。推理参考类别来自原始图的奖励分类器预测，
  完全不读取 `y`。累计下降计数不在改善后清零，参考类满 4 次且其他类至少下降过
  一次时停止；另有明确的总节点预算与尝试步数上限。预算包含根，修正原缓冲区的
  `max_size-1` 差一语义。推理停止时保留当前树，不复制作者多类别结果标记的回退分支。
* 候选不足时按父节点分支数和节点顺序加权采样上下文，补至 threshold+1。
  作者 `utils.py` 每次追加 5 个、按旧步数判断不足，本适配采用论文中的实际剩余候选阈值。
  不在每次试探后重新生成未选候选，候选在一次构树期间保持一致。
* CRG 的交替学习使用选中图的真实节点回复对，并按重构前后分类反馈差异加权。
  原始根帖重构在无可用回复对时使用。连续空间没有词长采样、teacher forcing。
* 核心阶段依次是：奖励分类器预训练、真实对 CRG 预热、ENS/CRG 交替训练、冻结构图模块、
  新的分类器训练。CRG 预热是适配新增，避免从未训练生成器补充候选。
  交替更新按事件执行，图按规模降序；没有复制作者 batch 内交替调度。
* 最终图缓存于内存，避免分类每个 epoch 重复构图；`forward` 也支持对新图在线构图。
  推理使用内容和树结构导出的独立随机流，不受标签、batch 划分或外部 RNG 状态影响。
* 不包含论文最终单独训练的 BERT 分类分支和均值融合；结果应标注为 **KPG feature adapter / graph-only**。

## 训练

```bash
python main.py --config_filename configs/EIN/Pheme_KPG_word2vec.yaml --device cuda:0
```

沿用项目的数据划分、文本编码、日志、验证集选模和测试返回接口。`main.py` 默认执行
五个 seed。服务器上的现有数据目录保持不变。其他数据集可复制其原有 YAML 的数据/编码
设置，将 `base_model` 改为 `KPG` 并加入示例中的 `kpg_*` 参数。

默认阶段轮数：奖励模型 30，CRG 预热 5，交替训练 5，最终分类器最多 `n_epochs=200`。
`kpg_max_nodes: 0` 使用**训练集**节点数中位数乘 `kpg_tau`，不访问验证/测试规模。
也可设置明确预算，如 `kpg_max_nodes: 32`。rollout 和逐事件构树成本较高，可先以较小预算
在服务器验证流程，再使用实验预算。节点特征维数必须与 `in_feats` 一致。

## 测试与检查点

`best_model.pth.m` 是完整 `KPG.state_dict()`，包含奖励模型、CRG、ENS、最终分类器、
节点预算和阶段完成标记。作者的旧权重不能直接加载（特征空间和模块命名不同）。

YAML 设置 `kpg_checkpoint` 为这个文件的路径、`kpg_test_only: true`，即可跳过训练。
测试必须传 `--seed` 匹配原训练 seed/数据划分，不允许把同一个检查点默认跨五个划分测试：

```bash
python main.py --config_filename configs/EIN/Pheme_KPG_word2vec.yaml --device cuda:0 --seed 0
```

单次训练也可使用 `--seed 0`；不传时训练仍执行原有五个 seed。
训练所用 word2vec 编码器及处理后数据也必须保持一致。

纯模型接口（不需要标签）：

```python
model = KPG(in_feats, hidden_dim, num_classes, args).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
model.eval()
with torch.no_grad():
    log_probs, _, _ = model(batch.to(device))
    key_graph, _ = model.construct_graph(one_graph.to(device))
```

输入为 PyG `Data/Batch`，根节点位于每个图的第 0 行；优先使用 `directed_edge_index`，
否则要求 `edge_index` 是从父到子的树。返回 key graph 中 `original_node_id=-1` 表示生成节点，
`is_generated` 标记来源。不要把生成节点解释为真实观测回复。

## 本地验证

`tests/test_kpg.py` 覆盖标签独立推理、批处理一致性、root-only 扩充、规模预算、
原始父节点缺失时的根连接、非法树拒绝、ENS/CRG 梯度、完整阶段训练与检查点重载。
运行：`python -m pytest tests/test_kpg.py -q`。这些是合成图验证，不能替代服务器数据集训练。

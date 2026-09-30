# SHPA 实现与运行

对应 `my model.pdf` 正文第 2.2–2.4 节，按用户确认的正文公式实现。
核心文件为 `model/SHPA.py`，已接入 `main.py` 和现有 `EINTrainer`。

- 公式（2）：使用 Gemma 对每条父帖–回复边生成的标签（0 支持、1 否认），
  划分两个无向图；两个图保留全部节点，独立进行三层 GCN 传播。
- 公式（3）：批内其他事件的同通道表示为正样本，全部异通道表示为负样本，
  排除自身；不使用真假标签构造正负样本。计算采用 cosine similarity 和
  logsumexp，batch size 为 1 时对比损失为零。
- 公式（4）–（6）：两路独立注意力池化，拼接后接一个线性分类器。
  输出 log-softmax，使用 NLL 得到等价的交叉熵损失。
- 公式（7）：`Lcls + shpa_lambda_contrastive * Lctr`。
  对比损失通过训练器已有的 `auxiliary_loss()` 接口加入，仅在训练中计算。

正文未给出跨通道交换算子和额外原图分支的计算公式，因此本实现采用正文的两路独立编码。
GCN 使用标准对称归一化和自环，ReLU 激活；空关系图仍保留节点自身特征。
注意力池化使用可学习的线性标量评分及图内 softmax。
这些是正文未细化处的明确实现选择。正文未指定 dropout 和 weight decay，配置默认均为 0。

## 使用旧实验缓存

三个配置分别为：

- `configs/EIN/Pheme_SHPA_word2vec.yaml`
- `configs/EIN/Weibo_SHPA_word2vec.yaml`
- `configs/EIN/DRWeibo_SHPA_word2vec.yaml`

SHPA 使用现有 `graph-resgcn-tree` 缓存键，匹配相同数据集、seed、622 划分、
Word2Vec/tokenizer、特征维度及原缓存的 `undirected: false` 设置。
运行时会打印实际复用路径；在内存中分别无向化支持/否认图，不修改旧缓存。
命中三个 split 的缓存时不加载 Word2Vec 或 Gemma，也不重新划分数据。
缓存不存在时走原有数据构建流程。

旧缓存如果只有 `edge_stance`，模型使用与之对应的 `edge_index`；
新缓存同时具有 `directed_edge_index` 和 `directed_edge_stance` 时优先使用这一对。
模型不会用节点 `state` 替代边立场，也不会将缺失标签默认为支持。
训练和推理均需要边立场标签，只有对比损失在推理时省略。

```bash
conda activate ein
python main.py --config_filename configs/EIN/Pheme_SHPA_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_SHPA_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/DRWeibo_SHPA_word2vec.yaml --seed 0
```

可加 `--device cpu`。省略 `--seed` 会沿用项目入口运行 seed 0–4，
这是五次随机划分实验，不是另外实现的严格五折交叉验证。
默认隐藏维度 128、batch size 128、学习率 0.0005、三层 GCN、温度 0.2；
Pheme/Weibo/DRWeibo 的对比系数分别是 2.0/0.5/1.0。
设置 `shpa_lambda_contrastive: 0.0` 可运行 NoCon 消融。

三个配置都提供同一个关系通道骨干选择字段：

```yaml
shpa_backbone: gcn     # 可选：gcn、resgcn、bigcn
shpa_num_layers: 3     # 同时控制所选骨干的传播层数
```

`gcn` 对应论文正文中的普通 GCN；`resgcn` 和 `bigcn` 直接复用项目
`model/DualBackboneOnly.py` 中与现有模型匹配的 ResGCN/BiGCN 节点编码器。
支持和否认通道各自拥有一套独立骨干参数，后续仍使用 SHPA 的独立注意力池化、
拼接分类与跨样本通道对比损失。三种选择使用相同图缓存。
BiGCN 的既有结构至少包含两层，因此当 `shpa_num_layers` 小于 2 时仍使用两层。
修改骨干做对比实验时，建议同时修改配置中的 `result_name`，避免覆盖同一实验目录。

## 大模型位置

项目已有 `stance_detection_cn.py` 和 `stance_detection_en.py`，
调用的都是 `google/gemma-2-9b-it`；它们负责离线标注 `stance_label`。
本机已找到完整权重缓存：

```text
/home/fzu/wencheng/.cache/huggingface/hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819
```

现有图缓存中已有立场标签，训练 SHPA 无需再次调用 Gemma。

## 验证

```bash
python -m unittest discover -s tests -p 'test_shpa.py' -v
```

测试覆盖公式（3）数值和梯度、关系图划分、旧缓存字段兼容、前向/反向、
批内图独立性、单节点/空关系、非法标签，以及推理关闭对比损失。
完整训练指标需要实际训练后测量；本实现不预设论文中的实验结果。

# GroupGain：回复群组图与条件判别增益模型

本次按 `rumor_group_gain_model_prompt.md` 的模型规格实现，沿用现有固定节点特征。`main.py` 已通过 `EIN_GroupGain_supervisor` 接入专用分阶段训练器，包括教师／学生验证选模、冻结教师、贡献头预训练、联合训练及最终测试。保留既有数据划分、特征缓存、其他模型和原本的未提交修改。

## 模型文件与接口

- `model/GroupGain.py`：独立节点编码、关系残差 GNN、集合上下文编码器、增益头、发送者／读出门控、冻结教师目标、训练阶段 API、checkpoint。
- `model/group_gain_grouping.py`：真实回复边适配、确定性完整连接分组、映射／成员回溯、二值关系去重、诊断及外部缓存内容键。
- `tests/test_group_gain.py`、`tests/test_group_gain_grouping.py`：性质与信息隔离验证。
- `scripts/smoke_group_gain.py`：合成／已有真实固定特征缓存上的分阶段运行检查。
- `trainer/GroupGain_trainer.py`：主入口使用的分阶段训练、验证早停、独立 checkpoint 和评价。
- `configs/EIN/{DRWeibo,Weibo,Pheme}_GroupGain_word2vec.yaml`：三个数据集的 ID Word2Vec 配置。

`GroupGain(in_feats, hid_feats=128, num_classes=2, args=None, **settings)` 的 `forward(data)` 返回 `[batch_size, num_classes]` **logits**。`EINGroupGain` 是现有 EIN trainer 的四元组适配类，返回 `(logits, U, S, D)`，并用 `classification_loss` 显式调用交叉熵。状态占位输出不参与训练，`physics_loss` 为零。

```python
from model.GroupGain import GroupGain, EINGroupGain

model = GroupGain(200, 128, 2, variant='full_model').to(device)
logits = model(data)                 # data 可为 Data 或 PyG Batch
probabilities = logits.softmax(-1)   # 仅评价时转概率

group_graphs = model.prepare(data)   # 固定预处理，适合每个事件只做一次
logits = model(group_graphs)
members = group_graphs[0].group_members
```

## 两个模块

模块 A 在 GNN 前分组。只比较相同**原始父节点**、相同局部关系的回复，固定原始特征 cosine 默认阈值 0.95。先按唯一特征内容排序，再做确定性完整连接阈值分组，重复次数不会改变 linkage；相似度分块计算，不建立整个 batch 的 N×N 矩阵。零特征、多父、环、关系冲突及孤立回复保留 singleton，诊断中记录异常。源帖单独作为群组 0。

每个节点先经过共享 Linear/ReLU，再对群组做逐维 max。Dropout 放在群组化之后。原关系按 `(source_group, target_group, type)` 去重，反向传播采用单独的 inverse type。原始群组大小、重复次数、度数与边数只出现在诊断中，不进入预测表示。异常数据中的群组内原关系保留且去重，并显式记录；它与 backbone 的 self transform 分开。

模块 B 用冻结、eval 的未门控教师对诱导群组子图重新编码，计算有符号的 `CE(Q[S], y) - CE(Q[S+k], y)`。目标只用于训练事件；声明为非训练 split 的输入会被拒绝。上下文默认以 0.5 概率取全部其他群组，其余取随机排列前缀，每事件 2 个候选、每候选 1 个上下文。可关闭教师完整图正确性筛选，以检查教师偏差。

学生用独立群组表示和 `mean(psi(u_j), j in S)` 预测实数增益；头输入没有标签、教师正确类别概率或教师增益。Huber 保留目标正负号。推理用全部其他群组上下文，事件内通过总和减当前项批量构造，`rho = sigmoid(predicted_gain / 0.2)`。源帖 rho 恒为 1。门控真实乘到发送消息，并用于读出；邻居归一化使用去重边数，不把门控放进分母。K=0 时回复读出为零，源帖通路仍可分类。

教师是训练函数的外部参数，模型不保存教师或目标缓存。完整学生 checkpoint 可独立无标签推理。

## 数据约定与已检查的数据

优先读取 `reply_edge_index`、`directed_reply_edge_index`、`directed_edge_index`；它们必须是真实父→回复边。只有显式指定 `edge_direction='parent_to_child'` 才允许回退到 `edge_index`，不从无向边猜原始方向。单图用 `root_index`／`rootindex`；缺失时只允许唯一零入度源帖。孤立源帖或方向异常时应显式给根索引。PyG Batch 先 `to_data_list()`，正确恢复各图边与根的局部编号。

主入口的专用训练器适配仓库 `TreeDataset`／`ResGCNTreeDataset` 已有的“源帖是原节点 0”约定：缓存缺根字段时显式补 `root_index=0`，在诊断中记录来源，再交给模型。各 split 仅预处理一次，保留原始成员索引与缓存身份。对于旧 Weibo 缓存中跨 train／val 或 train／test 的相同事件 ID，逐字段核对缓存图内容及标签，确认完全相同后只排除训练副本，保留验证集、测试集，避免泄漏；日志及 `grouping.json` 的 `input_events`、`excluded_overlap_events` 记录原始数量、排除的 ID、缓存索引与保留 split，原磁盘缓存不改写。同 ID 内容或标签冲突、split 内重复、val／test 重叠仍拒绝运行。缺 ID／可对齐原文件名时，用绑定具体缓存和 split 的索引 ID，并记录这一回溯限制。

`graph_id`／`event_id` 优先保留；老缓存没有 ID 时产生带 `content:` 前缀的内容指纹。该指纹不能代替跨数据版本的原始事件 ID，正式缓存应补充稳定事件标识和特征版本。`grouping_cache_key` 支持 dataset/split/version/settings，增益缓存若自行添加还须绑定教师 checkpoint hash；本次没有持久化增益缓存。

默认 **generic，无立场**。现有 `directed_edge_stance` 不会自动被使用。仓库的离线 Gemma 标注针对父帖，0 表示相信／同意、1 表示不同意／质疑。启用它时必须声明目标及原始整数映射，例如：

```python
model = GroupGain(
    200, 128, 2, stance_target='parent',
    relation_mapping={'generic': 0, 0: 1, 1: 2},
)
# 三个稳定关系 ID：0 generic，1 原始 stance 0，2 原始 stance 1。
# 如果旧缓存只有 edge_stance，必须先确认它与真实有向边逐条对齐，
# 再显式转换为 directed_edge_stance；模型不会自动对齐无向边立场。
```

源帖立场不会被混作父帖立场。显式 `reply_edge_type` 可用关系名字，或已声明映射内的整数 ID。

只读检查发现以下既有数据及 seed=0、622 划分；标签继续保留数值 0/1，仓库未提供可靠的事件标签语义说明，未重解释为真假／谣言。

| 数据集 | 事件数 | label 0 / 1 | train / val / test |
| --- | ---: | ---: | ---: |
| DRWeibo | 6019 | 3172 / 2847 | 3611 / 1204 / 1204 |
| Weibo | 4664 | 2351 / 2313 | 2798 / 933 / 933 |
| Pheme | 5740 | 3650 / 2090 | 3444 / 1148 / 1148 |

## 训练阶段 API

正式训练应沿用既有完整事件划分，用验证集选教师 checkpoint，再冻结；以下是阶段调用方式，不是已完成的完整实验。

```python
import torch
from model.GroupGain import GroupGain, compute_gain_targets, freeze_teacher

teacher = GroupGain(200, 128, 2, variant='module_a_only').to(device)
optimizer = teacher.init_optimizer()
# 教师每个训练 batch：完整图 + 随机掩码图，均未门控
optimizer.zero_grad()
teacher.teacher_training_loss(train_batch, mask_probability=0.5).backward()
optimizer.step()
# 正式实验在这里用验证集早停选出 checkpoint，然后冻结：
freeze_teacher(teacher)

student = GroupGain(200, 128, 2, variant='full_model').to(device)
student.initialize_from_teacher(teacher)
graphs = student.prepare(train_batch)
targets = compute_gain_targets(teacher, graphs, split='train')

# 短暂贡献头预训：仅更新集合编码器及gain_head，期间冻结其他模块。
head_parameters = student.gain_head_parameters()
head_ids = {id(parameter) for parameter in head_parameters}
for parameter in student.parameters():
    parameter.requires_grad_(id(parameter) in head_ids)
head_optimizer = torch.optim.Adam(head_parameters, lr=5e-4)
head_optimizer.zero_grad()
if targets.samples:  # K=0 或筛选后无目标时跳过
    student.gain_loss(graphs, targets).backward()
    head_optimizer.step()
for parameter in student.parameters():
    parameter.requires_grad_(True)

# 联合训练：CE + lambda_gain * Huber，教师始终冻结。
optimizer = student.init_optimizer()
optimizer.zero_grad()
student.compute_loss(graphs, teacher=teacher).backward()
optimizer.step()

student.save_checkpoint('checkpoints/group_gain/student.pt', metadata={
    'label_mapping': None,  # 正式训练中填入经确认的数据集标签含义
    'dataset': 'DRWeibo', 'split': '622', 'feature_version': 'word2vec',
})
inference_model = GroupGain.load_checkpoint('checkpoints/group_gain/student.pt').eval()
logits = inference_model(unlabeled_batch)
```

模型默认 hidden=128、layers=2、dropout=0.2、lr=5e-4、weight_decay=1e-4、lambda_gain=1。参考正确性筛选默认开启；它只影响贡献目标，不丢分类样本。训练阶段的学习率和预算由各自 optimizer／外部循环控制。`compute_loss` 对需要增益监督的模型要求教师或显式 `GainTargets`，避免静默变成纯 attention 训练。现有通用 EIN trainer 没有教师生命周期，完整 A+B 不能仅注册一个类就完成正确分阶段训练。

主入口因此使用 `GroupGainTrainer`，直接消费 `GroupGain` logits；`EINGroupGain` 适配类仍可用于自定义训练。

## 从 main.py 训练

```bash
python main.py --config_filename configs/EIN/DRWeibo_GroupGain_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Weibo_GroupGain_word2vec.yaml --seed 0
python main.py --config_filename configs/EIN/Pheme_GroupGain_word2vec.yaml --seed 0
```

省略 `--seed` 时主入口顺序运行 seeds `[0,1,2,3,4]`；`--device cpu` 可覆盖配置中的 CUDA。三个配置保持特征维度 200、Word2Vec、622 划分、原语言／分词设置与 max_hop。默认 hidden=128、layers=2、dropout=0.2、batch=128，未做数据集专门调参。`undirected: True` 用于复用原有 GCN 的 `resgcn-tree` 特征缓存；模型读取真实 `directed_edge_index`，在群组图中建立单独的反向关系类型。

| 配置参数 | 默认值 | 用途 |
| --- | ---: | --- |
| `group_gain_teacher_max_epochs` | 100 | 教师完整图＋掩码图训练上限 |
| `group_gain_teacher_patience` | 20 | 教师验证早停 patience |
| `group_gain_teacher_lr` | 0.0005 | 教师学习率 |
| `group_gain_teacher_mask_probability` | 0.5 | 教师非根群组掩码概率 |
| `group_gain_gain_head_epochs` | 5 | 冻结 backbone，只训练贡献头与上下文编码器 |
| `group_gain_gain_head_lr` | 0.0005 | 贡献头预训练学习率 |
| `group_gain_student_max_epochs` | 100 | 联合训练上限 |
| `group_gain_student_patience` | 20 | 学生验证早停 patience |
| `group_gain_student_lr` | 0.0005 | 学生学习率 |

教师和学生各自按 `selection_metric: val_loss` 选模。增益目标只在 train split 上生成，验证和测试不调用教师产生贡献监督；最终测试只评价验证集选出的学生 checkpoint。`attention_control` 与非增益 baseline 直接按学生预算训练，日志记录跳过教师的阶段；正式贡献比较须另外匹配预训练预算。

全训练集均为 K=0 时，日志和 checkpoint 明确记录空增益项被跳过，只训练源帖分类。如果存在回复，但正确性筛选整轮没有生成任何目标，则报出具体错误，提示延长教师训练或显式关闭筛选，避免把未获得增益监督的运行当成完整 B 训练。

输出沿用 `experiments/EIN/<dataset>/group_gain_full_undirected_valloss_word2vec/seed_<seed>/`，包括 `run.log`、`best_teacher.pth`、`best_model.pth`、`history.json` 和 `grouping.json`。`main.py` 在方法目录生成 `summary_val_loss.txt` 汇总各 seed。学生 checkpoint 可以通过 `GroupGain.load_checkpoint(...).eval()` 单独用于无标签推理。

既有早期检测 CLI 也可加载学生 checkpoint，例如：

```bash
python main.py --config_filename configs/EIN/DRWeibo_GroupGain_word2vec.yaml --seed 0 --eval_only --checkpoint_path experiments/EIN/DRWeibo/group_gain_full_undirected_valloss_word2vec/seed_0/best_model.pth --early_test_root dataset/DRWeibo/early_detection/60min
```

`--early_test_root` 应指向已有且包含 `raw/` JSON 的实际早期检测目录；上面是路径格式示例。eval-only 不训练教师或贡献头。

评价优先使用 checkpoint 中的模型与分组设置，并检查其中已声明的数据集和 seed；误传教师 checkpoint 会被拒绝。增益目标不写入学生 checkpoint。

checkpoint 保存参数、配置、特征维度、关系映射、分组设置、seed、根处理约定和标签映射 metadata。`load_checkpoint` 的 map_location 接受设备字符串或 `torch.device`。

## 可用对照

`variant` 支持 `root_only`、`original_graph`、`simple_dedup`、`feature_only_grouping`、`module_a_only`、`module_b_only`、`full_model`、`attention_control`、`leave_one_out_only`。除相应分组／增益开关外复用同一关系 backbone。

- `module_a_only` 与非增益 baseline 的 rho 恒为 1。
- `module_b_only` 在 singleton 图上学习条件增益。
- `attention_control` 保持同样头和门控，但 `compute_loss` 关闭增益损失。
- `leave_one_out_only` 所有目标都用全部其他群组上下文。
- `group_pooling='max'/'mean'`；`gate_mode='both'/'message_only'/'readout_only'`。
- `group_cos_threshold=0.90/0.95/0.98` 可按验证集做敏感性比较，当前没有调参结果。

第一版没有实现 grouping-after-GNN 对照、近重复 JS 重采样损失、自动 GPU 降 batch、自然近重复诊断。主入口已提供五 seed 调度和正常评价流程，但未启动完整长实验。正式比较需要为 baseline 匹配教师预训练带来的额外计算预算。

## 验证与限制

现有可用 CPU 环境为 Python 3.11、PyTorch 2.9.1+cu128、PyG 2.5.3；默认 Python 3.12 没有 PyTorch。GPU 驱动当前不可用。没有安装依赖或改动原有环境。

完整 `main.py --help` 的实际导入检查首先因缺少 `jieba` 失败；轻量验证环境还缺少旧主入口依赖的 `gensim`、`nltk` 和 `torch_scatter` 等。集成测试因此抽取真实入口函数和 CLI 控制体，替换文本编码／数据构造，运行真实 GroupGain、分阶段 trainer、优化器、checkpoint 和摘要流程。它不代表完整主入口在该轻量环境中已成功导入；正式训练须使用满足仓库依赖的环境。

```bash
PYTHONPATH=/tmp/nodeigm-test-deps /home/fzu/miniconda3/envs/torch-dl/bin/python -m unittest discover -s tests -p 'test_group_gain*.py' -v
PYTHONPATH=/tmp/nodeigm-test-deps /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/smoke_group_gain.py --synthetic
PYTHONPATH=/tmp/nodeigm-test-deps /home/fzu/miniconda3/envs/torch-dl/bin/python scripts/smoke_group_gain.py --data-cache dataset/DRWeibo/dataset_cache/tmp_kagnn_smoke/seed_0/train/processed/data.pt --report experiments/EIN/DRWeibo/group_gain_smoke_cpu/report.json
```

性质测试覆盖多群组图的受限精确叶子复制 1/5/10/50 次、群组特征与去重关系、eval logits、排列、非零／零特征、同／异关系、多父／环、1600 节点深链、非零根索引、不同大小 batch、诱导子图信息隔离、正负 CE 增益、标签无关推理、上下文外信息隔离、教师冻结、有效梯度、门控两处作用和独立 checkpoint。

实际结果：32 项单测全部通过。合成 3 图（含仅源帖事件）与 DRWeibo 真实训练缓存 4 图均完成教师 2 步、贡献头 2 步、联合训练 2 步、保存加载及无标签推理。smoke 使用 hidden=16、micro_batch=8，关闭教师正确性筛选以确保小样本有目标；没有验证早停或 held-out 评价。

主入口集成新增 6 项验证，与原测试合计 38 项通过。三份配置分别验证缓存复用与专用 supervisor；合成不同 train/val/test 事件完成真实教师 1 epoch、贡献头 1 epoch、学生 1 epoch 的流程，只在最后调用一次 test，增益目标只来自 train。还验证了教师冻结、阶段梯度、checkpoint 分组设置优先、无需教师的 eval-only，以及 `--seed 0 --device cpu` 的真实 argparse／YAML／动态路由／摘要写入控制流程（隔离未安装的文本编码和旧模型依赖）。

专用 trainer 另外在既有 DRWeibo Word2Vec 缓存每 split 各取两类各一图，完成三个阶段各 1 epoch 的 CPU 运行检查，train 94→81 群组、val 62→62、test 32→29；使用真实父子边、可对齐原文件名 ID，未发生 split ID 误报。全 K=0 训练集也通过明确跳过空增益阶段的检查。所有本轮 trainer 运行产物保存在自动清理的临时目录，未写入默认正式实验目录。这些小规模检查没有提供完整数据集或五 seed 性能结果。

真实小图固定特征 F=200，原始节点数 `[17,38,20,21]`，群组数（含根）`[17,37,20,18]`，合计 96→92。8 个条件增益目标正／负各半，范围约 `[-0.01397,0.34176]`；回复门控均值约 0.15666。教师无梯度，贡献头预训未改 backbone，联合训练三个头均有有效梯度，恢复后无标签 logits 最大差为 0。CPU 整个 smoke 约 0.75 秒（不含 import）。可复核 `experiments/EIN/DRWeibo/group_gain_smoke_cpu/report.json`。这些结果只验证执行和隔离性质，不能据此推断泛化性能或门控质量。

合成高出度预处理检查：seed=0、1000 个随机回复、32 维特征、CPU 单线程、阈值 0.95，产生 1001 个群组（含源帖），实测分组耗时约 0.313 秒。这只说明该输入上的预处理时间。

受限精确复制稳定性依赖副本进入原群组、不增加新类型连接、默认 max 和 eval 关闭 dropout。它不意味着语义改写、跨分支重复、任意新用户、零特征 singleton 副本或整棵子树复制都严格不变。没有声称 MaxPool 原创、群组统计独立、增益是因果效应或检测性能提升。

教师定义的增益与门控学生中的真实贡献可能不同；正确性筛选也不保证事实可靠性。完整实验应记录参考性能、增益正负／误差、门控分布、分组错误和教师偏差。现有 EIN 的 F1 是 binary、正类数值 1，AUC 使用 hard prediction；正式新实验应明确标签含义、概率 AUC／F1 定义和缺类 AUC undefined，再决定如何兼容既有报告。没有改写旧指标结果。

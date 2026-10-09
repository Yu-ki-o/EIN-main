# 三个数据集的跨数据集 OOD 实验

本设置借鉴所给论文的**单源跨数据集零样本评估形式**：只在源数据集训练与选模，在完整留出的目标数据集上测试。使用现有 DRWeibo、Weibo、Pheme 的六个有向组合，不需要改写 `model/` 内的网络。

这不是论文中 CSDA 方法的复现，也不包含 Twitter-COVID19、Weibo-COVID19 或目标域 80 条标注样本的少样本实验。DRWeibo 与 Weibo 的双向迁移衡量同语言、同平台的数据来源变化；涉及 Pheme 的四个方向同时包含语言、平台和数据来源变化。应分别解释这两类结果。

| 训练源域 | 测试目标域 | 提供的配置 |
| --- | --- | --- |
| DRWeibo | Weibo | `configs/ood/DRWeibo_to_Weibo_BiGCN_e5.yaml` |
| DRWeibo | Pheme | `configs/ood/DRWeibo_to_Pheme_BiGCN_e5.yaml` |
| Weibo | DRWeibo | `configs/ood/Weibo_to_DRWeibo_BiGCN_e5.yaml` |
| Weibo | Pheme | `configs/ood/Weibo_to_Pheme_BiGCN_e5.yaml` |
| Pheme | DRWeibo | `configs/ood/Pheme_to_DRWeibo_BiGCN_e5.yaml` |
| Pheme | Weibo | `configs/ood/Pheme_to_Weibo_BiGCN_e5.yaml` |

## 数据协议

- 每个源数据集先去重，再按源域标签分层划出训练集与验证集，默认 80%/20%。**相同源域、相同种子对两个目标域使用相同的训练/验证样本**。
- 目标域不参与训练和选模。其标签只用于检查重复样本的标签冲突、最终评估与审计统计，不决定源域抽样数量、类别配比或超参数。
- 先排除目标测试集中与原始源域有相同根帖 ID 或规范化源帖文本的样本，再对目标内部去重。目标域的重合处理不会反过来改变源域训练集。DRWeibo 和 Weibo 存在内容重合，因此这一步尤其重要。
- 去重是根帖 ID 与规范化文本的精确匹配，不保证发现改写、近义或跨语言重复。最终测试集规模可能小于目标原始数据集；报告结果时应同时给出每个方向的实际样本数。
- 同一重复组若出现相互冲突的标签，整组排除，并在审计报告中记录数量。
- `experiment_mode: ood` 使用独立 manifest 和缓存。原有 `id`、历史 `strict_ood` 缓存或结果不能混用，也不能直接视为新协议的结果。
- manifest 固定每条记录的来源与内容摘要。数据修改后需要重建划分；不同模型比较时必须使用同一套 manifest。

划分以 manifest 为准；原 ID 配置的 `split: '622'` 与 `k` 不控制 OOD 样本，配置生成器会移除这两项。

所有操作从项目根目录运行，并使用安装了项目依赖的 Python 环境。本机可使用 `/public/wc/.conda/envs/ein/bin/python` 代替下文的 `python`。

## 生成六组固定划分

```bash
python scripts/prepare_ood_splits.py \
  --all-pairs \
  --output-dir dataset/ood_splits \
  --seeds 0 1 2 3 4
```

每个方向会产生例如 `dataset/ood_splits/drweibo_to_weibo/seed_0.json` 的 manifest。YAML 使用 `seed_{seed}.json`，训练时自动替换种子。每份 manifest 的 `diagnostics` 包含样本与去重统计，全部方向的汇总位于 `dataset/ood_splits/audit_summary.json`；先检查这些统计再启动训练。覆盖已有划分需要显式传入 `--overwrite`。

当前工作区已生成六个方向各五个种子的 30 份清单。2026-10-09 对当前数据生成的样本数如下；不同种子的样本身份会变化，分层划分的数量一致：

| 源域 → 目标域 | 源域训练 | 源域验证 | 去重后目标测试 |
| --- | ---: | ---: | ---: |
| DRWeibo → Weibo | 4756 | 1190 | 3523 |
| DRWeibo → Pheme | 4756 | 1190 | 5729 |
| Weibo → DRWeibo | 3606 | 902 | 4961 |
| Weibo → Pheme | 3606 | 902 | 5729 |
| Pheme → DRWeibo | 4584 | 1145 | 5946 |
| Pheme → Weibo | 4584 | 1145 | 4508 |

这是划分审计结果，不是模型检测性能。源域内部去重分别移除了 DRWeibo 的 73 条、Weibo 的 156 条、Pheme 的 11 条记录。两个中文迁移方向还从目标侧移除了与源域重合的记录，因此目标规模不同；详细计数以清单为准。

也可以只建立一个方向：

```bash
python scripts/prepare_ood_splits.py \
  --protocol cross_dataset --sources DRWeibo --target Weibo \
  --output-dir dataset/ood_splits/drweibo_to_weibo \
  --seeds 0 1 2 3 4
```

不要根据目标域测试得分反复修改训练划分、模型配置或停止条件。新超参数在对应源域验证集上确定；六份示例配置沿用各自源域的现有 BiGCN 模板，作为运行起点，并不代表已完成 OOD 调优。

## 运行现有模型

六份示例使用相同的冻结多语言编码器 `intfloat/multilingual-e5-base`，输入维度为 768。`e5_local_files_only: true` 要求本机已有模型文件；如果缺失，先准备模型，或在允许下载的环境中改为 `false`。中文和英文都必须使用同一个编码器及文本预处理设置。

先运行一个方向、一个种子：

```bash
python main.py \
  --config_filename configs/ood/DRWeibo_to_Weibo_BiGCN_e5.yaml \
  --seed 0 --device cuda:0
```

六个方向各运行默认的五个种子：

```bash
for config in configs/ood/*_BiGCN_e5.yaml; do
  python main.py --config_filename "$config" --device cuda:0
done
```

这会启动 30 次完整训练，耗时取决于模型与设备。划分脚本只准备数据，不会自动启动训练。日志按目标域和独立结果名写入，例如 `experiments/EIN/Weibo/ood_drweibo_to_weibo_bigcn_e5/`；报告各方向的五种子均值和标准差，不要只报告最佳种子。

默认六份 BiGCN 配置使用 `EINTrainer`，测试输出 Accuracy、使用类别概率计算的 ROC-AUC、原有正类 F1、Macro-F1 和两个类别各自的 F1。其他模型若使用独立 trainer，保留其自己的指标实现，不能假设已经同步提供这些指标。新旧 AUC 的计算方式不同，不宜直接比较历史数值。

相同源域的两个方向使用同样的训练/验证 manifest 和源域超参数；目前主入口仍分别运行两个配置。若后续增加“一次训练评估多个目标域”的调度，仍应保持这些划分不变。

## 将已有模型 YAML 转为 OOD

数据入口继续构造现有图数据字段，训练仍通过 `main.py` 的 supervisor 分派到原来的模型。例如保留现有 ResGCN 语义变化模型：

```bash
python scripts/generate_ood_configs.py \
  --base-config configs/EIN/DRWeibo_ResGCN_UncertaintySemanticChange_word2vec.yaml \
  --manifest 'dataset/ood_splits/drweibo_to_weibo/seed_{seed}.json' \
  --output configs/ood/DRWeibo_to_Weibo_ResGCN_UncertaintySemanticChange_e5.yaml

python main.py \
  --config_filename configs/ood/DRWeibo_to_Weibo_ResGCN_UncertaintySemanticChange_e5.yaml \
  --seed 0 --device cuda:0
```

生成器保留模型结构与模型专用参数，并设置 OOD manifest、目标数据集、源域验证、独立结果名、统一编码器维度和至少 72 的 `max_hop`。默认推荐选择**源域**的已有配置作模板，随后只用源域验证集调参。若 manifest 尚未生成，可用 `--dataset Weibo` 显式指定目标，先生成 E5 配置。

为了防止沿用目标域信息，生成器清除旧数据缓存名、结果路径与预训练/测试 checkpoint，把已有分类权重重置为均等权重，并关闭 SEEGraphMAE 的目标域 test-time training。P2T3 会从头训练，不能把此结果当作包含原有额外预训练数据的完整 P2T3 设置。需要外部预训练或目标域适配的方法必须单独制定和报告协议；当前零样本入口会拒绝这类不受控配置。兼容模型结构不等于复现各模型论文的全部训练流程。

如需同中文语言的 Word2Vec 对照，先生成 manifest，再运行：

```bash
python scripts/generate_ood_configs.py \
  --base-config configs/EIN/DRWeibo_ResGCN_UncertaintySemanticChange_word2vec.yaml \
  --manifest 'dataset/ood_splits/drweibo_to_weibo/seed_{seed}.json' \
  --embedding word2vec \
  --output configs/ood/DRWeibo_to_Weibo_ResGCN_UncertaintySemanticChange_word2vec.yaml
```

此时 Word2Vec 仅在该 manifest 的**源域训练文本**上训练，验证与测试使用同一词表和权重。它不复用原 ID 模式的整个数据集 Word2Vec。中文与英文之间的迁移不支持这一方式，必须使用共享多语言编码器。比较模型时应固定编码器种类，不能把 E5 与 Word2Vec 的差异当作网络改进。

六份主配置只实施跨数据集实验。若使用划分工具中的 `drweibo_theme` 或 `pheme_event`，它们是额外的跨话题/跨事件研究，不是所给文章中的跨数据集主设置；DRWeibo 缺失话题的记录不能当作正常领域，需按该协议显式排除。

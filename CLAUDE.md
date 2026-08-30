# SQuaDE — 给 agent 的操作说明

退化鲁棒的 AIGC 检测。核心主张:**不同退化杀死不同深度的取证线索**,所以先量化损伤,
再据此在三个深度专家之间分配权重。方法动机见 `README.md`。

**每个设计决策背后的实测在 `docs/design_experiments.md`,探针的具体跑法在
`docs/probe_protocol.md`。动手前先读这两份 —— 里面记录了已经踩过的坑和已被证伪的假设,
能省掉重复劳动。**

---

## 流水线顺序(有硬依赖,不能跳)

```
utils/preprocess.py            退化生成 + manifest
        │
        ├─► cache_features.py --layers all  ─► probe_layers.py   ← 定三个抽头层
        │                                          │
        ├─► training/train_classifier.py           │             ← 退化估计器,可与探针并行
        │        train → predict                   │
        │                                          ▼
        └─► cache_features.py --layers <探针选出的三层>
                     │
                     ▼
            training/train_WeightandExpert.py  stage1 → stage2 → eval
                     │
                     ▼
                Inference.py
```

`train_classifier` 与 `probe_layers` 互不依赖,可并行。其余都是硬依赖。

## 一个门禁

`probe_layers.py` 跑完会输出**证伪检查**。若不通过(各退化桶的最优层几乎重合),
说明「按退化路由深度」这个自由度本身没有价值,**停下来报告,不要继续建专家**。

已经验证过后果:在退化的合成数据上强行往下走,Stage 2 会给出 oracle gap ≈ 0,
三小时白费。通过标准和五个排查分支见 `docs/probe_protocol.md`。

---

## 会静默出错的地方

这些不会报错,只会让结果悄悄失效。改动相关代码前先确认没有破坏它们。

| # | 纪律 | 破坏后的症状 |
|---|---|---|
| 1 | 特征提取全程 **不 resize**(`do_resize=False`) | HF processor 默认压到 224x224,抹掉浅层指纹、洗掉 crop 证据、与退化码字的 resize 维打架 |
| 2 | 抽层后**全层统一** `model.norm` | 跨层范数差 293 倍,探针测到的是范数不是信息量,三个抽头层必选错 |
| 3 | 标准化统计量**只在训练集上**标定 | 验证集分布泄漏进标准化参数,所有消融数字一起虚高 |
| 4 | `freeze()` 必须同时 `eval()` + `requires_grad_(False)` | 只关梯度不关 dropout,权重 MLP 学的是噪声上的平均 |
| 5 | 权重 MLP **永远不接受图像特征** | 退化驱动路由变成内容驱动路由,novelty 失效且从指标上看不出来 |
| 6 | preprocess 的归一化在退化**之前** | 先退化再缩放会重采样掉 JPEG 的 8x8 块网格、平均掉噪声、改掉模糊的有效 σ |
| 7 | train/val 按**源图分组**划分 | 同一张图的干净版与退化版被拆到两侧,模型记住内容再认出来 |
| 8 | Stage 2 喂**预测**码字而非真值 | 测试时拿不到真值,用真值训是 train/test 失配,A3 数字虚高 |

第 5 条在 `models/weights_mlp.py` 里用运行时断言焊死了,别绕过它。

## 诊断输出要看,不要只看最终指标

三个脚本会主动报告失败模式,它们**在最终 AUC 上看不出来**:

- `probe_layers.py` 的**并列检测** —— AUC 饱和时 argmax 只取第一个,选层实质是任意的
- `train_WeightandExpert.py stage2` 的**路由表** —— 权重若不随码字变化,名义上有路由、
  实质是恒定重加权,A2→A3 的增益与退化无关,不能作为 novelty 证据
- `utils/preprocess.py` 的**跨 split 源图组计数** —— 必须为 0

---

## 环境

`pip install -r requirements.txt`,外加一步 pip 管不了的:DINOv3 是 gated repo,
需先在 HuggingFace 页面同意协议,再 `hf auth login`。

**不要升级 huggingface_hub 到 1.x** —— transformers 4.x 硬约束 `<1.0`,升了会让
`import transformers` 直接 ImportError。装 CLI 要写 `pip install "huggingface_hub[cli]<1.0"`。

耗时参考(实测 MPS / ViT-H+ / 512x512):缓存约 1.05 s/图。独显上远快于此,且
`--batch-size` 可调大(MPS 上反而是 batch=1 最快,原因见 `cache_features.py` 的
`DEFAULT_BATCH` 注释)。`cache_features.py` 支持断点续传,中途挂掉重跑同一条命令即可。

---

## 报告结果时

- **`docs/design_experiments.md` 里的数字全部来自合成数据,绝对值没有外部效度。**
  可迁移的是机制结论,不是准确率。不要把那些数字写进论文,也不要拿它们当基线比较。
- 核心指标不是原始 AUC,而是**关闭 oracle gap 的百分比** `(A3-A2)/(A3o-A2)`。
  `train_WeightandExpert.py eval` 会直接算出来。
- 报告要分退化档给,不要只给总体数字(`eval` 已按退化强度分档输出)。
- 消融数字要 ≥3 个随机种子并带误差棒。单种子的差异在小验证集上标准误可达 ±0.03,
  别把噪声当结论 —— 这个错误在开发过程中犯过一次,记录在 `docs/design_experiments.md` 第 6 节。

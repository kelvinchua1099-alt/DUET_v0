# E0 层级探针 —— 运行协议

决定浅/中/深三个抽头层，同时是整个方法的**可行性检验**。

> 必须在训练任何专家之前跑完。若证伪检查不通过，Stage 1/2 全部白做 —— 与其建完三个
> 专家、跑完 Stage 2 才发现 oracle gap ≈ 0，不如在这里花两三个小时先把前提验掉。

---

## 0. 规模怎么定

探针的分辨力由**验证集里每个退化桶的样本数**决定，而不是总样本数。

每张源图产出 2 行（干净 + 退化），退化行落进 14 个码字之一。所以：

```
每个 dim 桶的验证样本数 ≈ 源图数 × val_frac ÷ 6
AUC 的标准误 ≈ 0.5 / sqrt(该桶样本数)      （AUC≈0.8、正负各半时的粗略量级）
```

| 源图数 | 缓存耗时(Mac) | 磁盘 | 每个 dim 桶验证样本 | AUC 标准误 | 用途 |
|---|---|---|---|---|---|
| 500 | ~18 min | 0.24 GB | ~25 | ±0.10 | **试跑**，查配置错误 |
| 2,000 | ~70 min | 0.96 GB | ~100 | ±0.05 | 最小可用 |
| **5,000** | **~2.9 h** | **2.4 GB** | **~250** | **±0.03** | **推荐** |
| 15,000 | ~8.7 h | 7.2 GB | ~750 | ±0.018 | 想按 level 细分桶时 |

耗时按实测 **1.06 s/图**（MPS、ViT-H+、512×512、batch=1、`--layers all`）估。独显上快得多。

> **缓存 33 层与缓存 3 层的耗时几乎相同**（1064 vs 1062 ms/图）。backbone 无论如何都要
> 跑完 32 个 block，`--layers` 只决定存哪几层的池化结果。所以不要为了省时间少缓存层数 ——
> 省不到，只会让探针扫不全。差别只在磁盘（0.24 vs 0.02 MB/行）。
>
> 这个「同价」是优化后的结果。优化前逐层调 `model.norm` + 池化，33 层要 ~4 s/图，
> 其中 3 s 是 kernel 启动开销。见 `design_experiments.md` 第 8 节。

**先跑 500 张的试跑。** 配置错误（标签推反、路径不对、码字列错位）在试跑里 18 分钟就
暴露，在正式跑里要等三小时。

分桶模式取 `--bucket-mode dim`（6 个退化维 + clean，共 7 桶）。`level` 模式把每维再按
强度拆开，桶数涨到 15，同样源图数下每桶样本少一半以上 —— 除非源图上万，否则不要用。

---

## 1. 命令

```bash
# ---- 试跑：500 张，约 20 分钟，只为验证配置 ----
python utils/preprocess.py --data data/raw --out-csv data/manifest_pilot.csv --limit 500
python cache_features.py   --manifest data/manifest_pilot.csv --out cache/probe_pilot --layers all
python probe_layers.py     --cache cache/probe_pilot --out probe/pilot --bucket-mode dim --min-bucket 15

# ---- 正式：全量 ----
python utils/preprocess.py --data data/raw --out-csv data/manifest.csv
python cache_features.py   --manifest data/manifest.csv --out cache/probe --layers all
python probe_layers.py     --cache cache/probe --out probe/full --bucket-mode dim --min-bucket 60
```

`cache_features.py` 支持断点续传，中途挂掉重跑同一条命令即可，已完成的样本会跳过。

参数说明：

| 参数 | 取值 | 理由 |
|---|---|---|
| `--layers all` | 33 层全取 | 探针就是要扫全部深度；训练阶段才只缓存选中的 3 层 |
| `--bucket-mode dim` | 7 桶 | 见上，`level` 需要上万源图 |
| `--min-bucket` | 试跑 15 / 正式 60 | 低于此数的桶直接丢弃，避免用几十个样本算出的噪声 AUC 去选层 |
| `--val-frac` | 默认 0.25 | manifest 若已有 `split` 列（preprocess 默认会写）则以它为准 |
| `--l2` | 默认 1e-3 | 3842 维、样本数远小于维度，正则不能关 |

---

## 2. 跑完先看三件事，顺序不能反

### ① 并列检测

```
⚠️ 有桶在最优 AUC 上并列 N 层
```

N > 3 就说明该桶的 AUC 饱和了（多半接近 1.0），`argmax` 只取第一个，**选层实质是任意的**。
这时热力图无效，别往下读。原因通常是任务对该桶太容易 —— 真假差异过于明显、或退化太弱。

### ② 证伪检查

```
证伪检查:各桶最优层 = {'jpeg>0': 29, 'blur>0': 31, 'crop>0': 6, ...}
  最优层跨度 25 层,不同取值 5 个
```

**通过标准**（三条同时满足）：

- 最优层跨度 ≥ 3
- 不同取值 ≥ 3
- 最大并列 ≤ 3

通过 → 「不同退化杀死不同深度的线索」有了直接证据，这张热力图直接进论文当 Figure 2。

不通过 → **停下来，不要继续建专家**。按下面的分支排查。

### ③ README 主张的方向性

```
*smudged    最优层 = 29
*shattered  最优层 = 6
```

README 主张 smudged（压缩/模糊/噪声）应偏深、shattered（裁剪/缩放）应偏浅。
方向对上了，这是最强的一张图；方向反了，也是有价值的发现，但整个 story 要重写。

---

## 3. 不通过时的排查分支

| 症状 | 可能原因 | 处理 |
|---|---|---|
| 所有桶 AUC ≈ 1.0 | 真假差异太明显，任务过易 | 换更难的生成器；或确认没有把某个与真假完全相关的伪线索（分辨率、格式、文件名）泄漏进去 |
| 所有桶 AUC ≈ 0.5 | 标签推反 / 码字列错位 | 查 manifest：`label` 分布、`d0..d5` 列序是否为 `[jpeg,blur,resize,noise,jitter,crop]` |
| 最优层全挤在深层 | 池化把浅层证据丢了 | 浅层证据是分布性质，检查缓存的 `pool` 是否为 `cls+mean+std`；试 `--no-prenorm` 对照，看 prenorm 那两个标量贡献多少 |
| 最优层全挤在 0-2 层 | 退化太弱，深层语义没被触动 | 提高强度谱；或确认归一化没有在退化之后执行（会抹掉证据） |
| 跨度够但 `*shattered` 也偏深 | crop 被缩放回 512，分辨率线索没了 | 这是 preprocess 纪律 4 的已知代价，见该文件说明 |

---

## 4. 产物与去向

```
probe/full/
├── heatmap.csv       AUC[33 层 × 桶]，画图用
├── heatmap.txt       终端可读版 + 退化维共现矩阵
└── selected.json     选出的三层 + 证伪结论 + 各桶样本数
```

`selected.json` 里的 `layers_list` 直接喂给下一步，不用手抄：

```bash
LAYERS=$(python -c "import json;print(','.join(map(str,json.load(open('probe/full/selected.json'))['layers_list'])))")
python cache_features.py --manifest data/manifest.csv --out cache/train --layers $LAYERS
```

选层同时给出两个版本，都要看：

- `unconstrained` —— 无约束的最优三层
- `band_separated` —— 强制浅/中/深各取一层（实际采用的）

两者目标值的**差距**本身是结论：差距接近 0 说明最优三层天然分散，深度多样性是数据给的；
差距大说明最优三层其实挤在一起，深度多样性是被约束硬加上去的 —— 论文里必须说明这一点，
否则 reviewer 复现时会发现无约束搜索给出完全不同的三层。

---

## 5. 共现矩阵：一定要看一眼

```
多重退化样本: 0/10000 (0%)
```

用 `utils/preprocess.py` 生成的数据，每图只施加一种退化，这个数**应该是 0**。
不是 0 说明数据不是本脚本生成的，或者被重复处理过（干扰图上又叠了一层）。

多重退化为 0 时，边缘分桶等价于单退化隔离，归因完全干净，没有混杂需要担心。

# 当前进度与恢复指南

最后更新 2026-08-26。分支 `cloud`。

---

## 一句话状态

**E0 探针门禁已通过,抽头层定为 20/24/28。可以开始建三个专家(Stage 1)。**
但在动手前请先看下面「悬而未决的两件事」——其中一件可能推翻整个方案的价值。

---

## 恢复环境

```bash
source /workspace/SQuaDE/env.sh
```

**必须先跑这一句。** 系统级的 `pip` 包装在 `/` overlay 上(20 GB,容器重启就没了),
真正完整的环境在 `/workspace/venv`。`env.sh` 会把 PATH、`HF_HOME`、`SQUADE_TAXONOMY`
一起设好。这台机器的两个坑写在 `env.sh` 的注释里:

* GPU 是 Blackwell(sm_120),PyPI 默认的 cu124 轮子里**没有对应 kernel** ——
  `torch.cuda.is_available()` 返回 True,但跑任何算子都报
  `no kernel image is available`。必须 cu128(现装的是 torch 2.11.0+cu128)。
* DINOv3 是 gated repo,token 要在 `HF_HOME` 下,否则报 **401** 而不是网络错误。

---

## 东西都在哪(全部在 /workspace,持久卷)

| 路径 | 内容 | 体积 |
|---|---|---|
| `/workspace/SQuaDE` | 代码(= 本仓库,已推 `cloud` 分支) | 44 MB |
| `/workspace/venv` | Python 环境,torch 2.11.0+cu128 | 12 GB |
| `/workspace/.hf_cache` | DINOv3 权重 + HF token | 3.2 GB |
| `/workspace/data/ntire/_zips` | NTIRE train 6 个 shard,277,643 张 | 107 GB |
| `/workspace/data/ntire/shard_5` | 已解压的 shard_5,27,643 张 | 11 GB |
| `/workspace/data/raw` | 10,000 源图软链 + 自造退化图(**门禁没过,别用**) | 15 GB |
| `/workspace/data/ntire_val` | val 10,000 + val_hard 2,500 + 三份带退化真值的标签 | 4 GB |
| `/workspace/cache/probe_val` | val 的 33 层特征(**当前有效的那份**) | 2.4 GB |
| `/workspace/cache/probe_full` | train 自造退化的 33 层特征(门禁没过) | 4.8 GB |
| `/workspace/probe/val` | 有效的热力图 + Figure 2 + selected.json | — |
| `/workspace/logs` | 全部运行日志、竞赛页元数据快照 | — |
| `/workspace/logs/transcript` | **本次会话完整记录**(见下) | 3 MB |

`probe/val/` 的四个文件也已入库(选层结论要有出处)。


### 会话记录

`/workspace/logs/transcript/` 下三份,**刻意不入 git**(2 MB 的会话数据,且含大量
中间过程,不属于代码):

| 文件 | 说明 |
|---|---|
| `session_raw.jsonl` | Claude Code 的原始记录,2 MB |
| `session.md` | 可读版,工具输出截断到 2000 字符,308 KB |
| `session_full.md` | 可读版,工具输出完整,476 KB |

用 `tools/render_transcript.py` 重新生成(该脚本已入库):

```bash
python tools/render_transcript.py /workspace/logs/transcript/session_raw.jsonl -o out.md [--full]
```

⚠️ 一个不显然的坑,写在该脚本的 docstring 里:**会话中途插进来的用户消息不会变成
`type: user` 记录**。助手正在跑工具时用户发的话走的是队列 —— 先记一条
`type: queue-operation, operation: enqueue`,再塞进某个 tool_result 的开头。
本次会话 170 条 user 记录里只有 8 条带文本,另外 16 条真实发言全在 queue-operation 里。
只按 `type=user` 导出会丢掉三分之二的对话。

---

## 走到这一步的关键结论

### 1. NTIRE train 的图是**干净源图**,退化要自己施加

竞赛 Data 页 Train 行写的是 "(Provided as distortion pipeline)"。实测佐证:跨 6 个
shard 抽 360 张,**JPEG 亮度量化表逐一相同**(标准 IJG 表,q≈93)。全库 277,643 张
100% 是 `.jpg`。

副作用:任何以 train 为源的「干净档」都自带一层没记在码字里的轻度压缩。
对所有图一视同仁,不造成桶间偏差,但 clean 基线整体偏高,**报论文时必须说明**。

### 2. 在 train 上自造退化 —— 门禁没过,原因已查清

10,000 源图 / 20,000 行,最大并列 5 层(要 ≤3)。排查掉两个嫌疑:

* **不是泄漏**:train/val 源图交集 0;val 干净图在 train 里的最大余弦 0.9872,
  ≥0.95 的仅 1.07%,且多为同标签的相似场景。
* **不是分辨率**:按源图短边分层后,第 0 层 AUC 只掉 2.5 点、第 20 层掉 0。
  (分辨率确实与真假相关 —— 真图短边中位 896、假图 768,且 26% 的假图是 1024×1024;
  但它只值 2.5 点。)

**真因:退化根本没伤到检测器。** 第 23 层,clean 99.99,而谱里最重的几档 ——

| 最重档 | AUC | vs clean |
|---|---|---|
| jpeg q=30 | 99.90 | −0.09 |
| blur σ=2.0 | 99.83 | −0.16 |
| resize 0.25× | 99.98 | −0.00 |
| noise σ=0.10 | 99.52 | −0.47 |

没有线索被杀死,就没有可路由的对象。这正是 CLAUDE.md 预言的
「Stage 2 会给出 oracle gap ≈ 0」。

### 3. 换到 NTIRE val 的官方退化真值 —— 门禁全过

val / val_hard / test-public 的标签文件里带 `distortions` 与 `distortion_scales`
两列,**官方逐图给出了施加了哪些畸变、各自多强**。同一套代码:

| 证伪标准 | 要求 | train 自造 | **val 官方真值** |
|---|---|---|---|
| 最优层跨度 | ≥3 | 7 | **9** ✅ |
| 不同取值 | ≥3 | 5 | **10** ✅ |
| 最大并列 | ≤3 | 5 ❌ | **1** ✅ |

22 个桶的并列层数全是 1。退化掉幅变成 0.65~2.62 点(有伤了)。
README 的方向也对上了:`*smudged` 最优层 27(偏深) vs `*shattered` 23(偏浅)。

两个变量同时变了(生成器更新 + 退化更真实),**尚未分离出主因** —— 见下。

---

## 悬而未决的两件事

### A. oracle 增益可能只是噪声(**优先级最高**)

探针层面的天花板只有 **+0.37 AUC 点**(逐桶最优层 98.61 vs 单一最优层 98.23),
而且取自 33 层的 max,带 winner's curse 偏差;每桶只有约 100 个验证样本。
选出的三层还只覆盖其中 26%。

**这个数字决定三个专家值不值得建。** 检验脚本已写好:

```bash
source env.sh && export SQUADE_TAXONOMY=ntireval
python tools/bootstrap_oracle.py --cache /workspace/cache/probe_val --out /workspace/probe/val
```

它给出 bootstrap 区间 + 置换零假设(= 完全没有真实深度偏好时,纯靠 max-over-33-layers
能刷出多少)。真实增益必须显著高于置换基线,否则先扩数据再说。约 5 分钟。

### B. 带退化真值的图只有 10,000 张

| 来源 | 图数 | 状态 |
|---|---|---|
| val | 10,000 | ✅ manifest + 特征都有 |
| val_hard | 2,500 | 已下载,**未建 manifest**(多 jpeg_ai / adv_embed_* / cheng2020 等,码本要扩) |
| test-public | 2,500 | 标签已下到 `data/ntire_val/_dl/test_labels.csv`,**图未下**,24 种畸变 |

上限 15,000。而 277k 的 train 因为门禁过不了用不上。

**消歧实验**(约 40 分钟,代码全现成):用 val 那套退化配方(含 downscale/crop/pixelate)
施加到 train 的 10,000 张图上,重跑探针。

* 过了 → 主因是退化太弱,277k 全部可用,数据量问题解决
* 没过 → 主因是生成器太老,只能守着这 15,000 张,三个专家(约 1.9M 参数)
  在 1 万多张上训,过拟合风险要认真对待

---

## 下一步命令

```bash
source env.sh
export SQUADE_TAXONOMY=ntireval

# A. 先验 oracle 增益是不是噪声(5 min)—— 建议先跑这个
python tools/bootstrap_oracle.py --cache /workspace/cache/probe_val --out /workspace/probe/val

# B. 若 A 通过,缓存选定的三层,进 Stage 1/2
python cache_features.py --manifest /workspace/data/manifest_val.csv \
       --out /workspace/cache/train_val --layers 20,24,28
python training/train_WeightandExpert.py stage1 --cache /workspace/cache/train_val --out runs/squade
# Stage 2 还需要退化估计器的**预测**码字(不能用真值,见 CLAUDE.md 纪律 8):
python training/train_classifier.py train   --manifest /workspace/data/manifest_val.csv --out runs/deg
python training/train_classifier.py predict --manifest /workspace/data/manifest_val.csv \
       --ckpt runs/deg/best.pt --pred-out /workspace/cache/pred_codes.csv
```

⚠️ `training/train_classifier.py` 的 `DegradationDataset` 假设图是 512×512 且做 8 对齐
随机裁剪。NTIRE val 的图**尺寸五花八门**(1200 张抽样里 667 种尺寸,26% 短边 < 512),
跑之前先确认那段裁剪逻辑在小图上不会炸。这一步还没验过。

---

## 码本(`SQUADE_TAXONOMY`)

| 值 | 维度 | 用途 |
|---|---|---|
| `synthetic` | 6 | `utils/preprocess.py` 的自造谱,历史默认 |
| `ntire` | 8 | 官方训练管线 7 组 + 我们加的 geometric |
| `ntire7` | 7 | 严格只有官方训练管线的 7 组 |
| **`ntireval`** | **19** | **val 标签的 19 个畸变名,当前实验用的就是它** |

缓存目录的 `config.json` 会记 `taxonomy`,`probe_layers.py` 读到不一致会直接拒绝,
不会静默串味。

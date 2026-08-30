"""E0 层级探针 —— 决定浅/中/深三个抽头层。

    python probe_layers.py --cache cache/probe --out probe/

**这一步必须在训练任何专家之前跑。** 它同时是选层工具和整个方法的可行性检验:
若所有退化桶的最优层都挤在同一处,「按退化路由深度」这个自由度本身就没有价值,
应当回头改设计 —— 而不是先建三个专家、跑完 Stage 2 才发现 oracle gap ≈ 0。

流程:
    1. 读全层缓存 (M, 33, 3840)
    2. 逐层用**训练集**统计量做固定标准化(不是 LayerNorm,理由同 experts 决策 2)
    3. 每层拟合一个 **线性** 探针(Linear(D,1) + logistic,LBFGS 全批次)
    4. 在验证集上按退化桶分别评估 AUC -> 热力图 [33 层 x 桶]
    5. 穷举 C(33,3) 选三层 + 证伪检查

两个刻意的选择:

* **线性,不是 MLP**  探针要测的是「这一层含多少可分信息」。带隐层的话测到的是
  「层的信息量 x MLP 容量」的混合物,换个隐层宽度热力图就变、选出的层跟着变。
  线性探针测的是线性可分性这一个明确的量,也是 UniFD 以来的标准协议,数字对外可比。

* **每层一个探针,而不是每(层,桶)一个**  因为 Stage 1 里每个专家只有**一个**共享
  分类头,在所有退化状态上一起训,自适应发生在权重而非头上。所以正确的问法是
  「用一个共享的头,第 ℓ 层在这个退化子集上表现如何」,而不是「为这个退化专门配一个头
  能到多好」。后者可用 --per-bucket 额外跑,两者之差 = 「按退化分专家」能多赚多少,
  是一个有用的对照,但不是主线。

探针是量具,读完 AUC 就丢弃,不进最终模型。最终模型里只有 3 个专家。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_features import DEG_DIMS, FAMILIES, N_LEVELS, TAXONOMY_NAME, load_cache  # noqa: E402

# README 的核心二分(smudged 抹掉浅层信号指纹 -> 应听深层;shattered 毁掉深层全局构图
# -> 应听浅层)现在由码本给出,不再写死在这里 —— 换成 NTIRE 分组时退化族跟着换,
# 见 utils/deg_taxonomy.py。FAMILIES 里可以有第三族(如 ntire 的 photometric),
# 汇总桶按「本族有、其余族全无」的互斥口径建,归因才干净。
SMUDGED = FAMILIES.get("smudged", ())
SHATTERED = FAMILIES.get("shattered", ())


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(scores) == 0 or labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):                     # 并列取平均秩
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- 分桶

def make_buckets(codes: np.ndarray, mode: str, min_n: int) -> dict[str, np.ndarray]:
    """边缘分桶:桶 `jpeg=2` 收纳所有 code[jpeg]==2 的样本,不管身上还有没有别的退化。

    严格的做法是单退化隔离(某维>=1 且其余全 0),归因最干净,但真实数据里这类样本
    往往只有个位数,拟合不出可信 AUC。边缘分桶样本量够,代价是有混杂 —— 只要各维
    退化是**独立随机**施加的,其余维度在桶内外分布均衡,只贡献噪声而非系统性偏差。
    独立性由 --out 里的共现矩阵给出,跑完务必看一眼。
    """
    b: dict[str, np.ndarray] = {"clean": codes.sum(1) == 0}
    for i, d in enumerate(DEG_DIMS):
        if mode == "level":
            for lv in range(1, N_LEVELS):
                b[f"{d}={lv}"] = codes[:, i] == lv
        else:
            b[f"{d}>0"] = codes[:, i] >= 1
    # 退化族汇总桶,直接检验「smudged 听深层 / shattered 听浅层」这条主张。
    # 口径是**互斥**的:本族至少中一维,且其余任何族一维都没中 —— 否则一张同时被
    # 模糊和裁剪的图会同时进两个桶,两桶的最优层被同一批样本拉到一起,证伪检查失效。
    fam_idx = {f: [DEG_DIMS.index(x) for x in dims if x in DEG_DIMS]
               for f, dims in FAMILIES.items()}
    for f, own in fam_idx.items():
        if not own:
            continue                     # 该族在当前码本里一维都没有(如 ntire7 的 shattered)
        others = [i for g, ix in fam_idx.items() if g != f for i in ix]
        hit = codes[:, own].sum(1) >= 1
        b[f"*{f}"] = hit & (codes[:, others].sum(1) == 0) if others else hit
    return {k: np.flatnonzero(v) for k, v in b.items() if v.sum() >= min_n}


def cooccurrence(codes: np.ndarray) -> str:
    """退化维度之间的相关性。边缘分桶的有效性依赖于它们近似独立。"""
    on = (codes > 0).astype(np.float64)
    lines = ["      " + "".join(f"{d[:6]:>8}" for d in DEG_DIMS)]
    for i, d in enumerate(DEG_DIMS):
        row = f"{d[:6]:>6}"
        for j in range(len(DEG_DIMS)):
            if i == j:
                row += f"{'—':>8}"
            elif on[:, i].std() < 1e-9 or on[:, j].std() < 1e-9:
                row += f"{'n/a':>8}"
            else:
                row += f"{np.corrcoef(on[:, i], on[:, j])[0, 1]:>8.2f}"
        lines.append(row)
    n_multi = int((on.sum(1) >= 2).sum())
    lines.append(f"\n多重退化样本: {n_multi}/{len(codes)} ({n_multi / len(codes):.0%})")
    lines.append("相关性 |r| 若普遍 < 0.15,边缘分桶的混杂可忽略;若某对 |r| > 0.3,"
                 "该桶的结论需要谨慎解读。")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 探针

def fit_probe(X: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, iters: int = 120) -> torch.Tensor:
    """logistic 线性探针,LBFGS 全批次。返回权重 (D+1,),末位是偏置。"""
    w = torch.zeros(X.shape[1] + 1, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w[:-1] + w[-1]
        loss = F.binary_cross_entropy_with_logits(z, y) + l2 * w[:-1].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def probe_scores(X: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    return (X @ w[:-1] + w[-1]).cpu().numpy()


# --------------------------------------------------------------------------- 选层

def select_layers(H: np.ndarray, bucket_names: list[str], n_layers: int, k: int = 3):
    """穷举 C(L,k),最大化 mean_b [ max_{ℓ∈S} AUC[ℓ,b] ]。

    这个目标刻意与下游的 oracle 指标对齐:「假如我能为每个退化桶挑最合适的那一层,
    这 k 层组合能达到多好」—— 正是 A3-o 在层选择这一步的对应物。
    """
    cols = [i for i, n in enumerate(bucket_names) if not n.startswith("*")]   # 汇总桶不计入目标
    M = np.nan_to_num(H[:, cols], nan=0.5)
    best, best_s = None, -1.0
    for S in itertools.combinations(range(H.shape[0]), k):
        s = float(M[list(S)].max(0).mean())
        if s > best_s:
            best, best_s = S, s
    # 强制浅/中/深分散的版本,与无约束最优对比
    b3 = n_layers / 3.0
    bands = [range(0, int(b3) + 1), range(int(b3), int(2 * b3) + 1), range(int(2 * b3), n_layers + 1)]
    bb, bb_s = None, -1.0
    for S in itertools.product(*bands):
        if len(set(S)) < k:
            continue
        s = float(M[list(S)].max(0).mean())
        if s > bb_s:
            bb, bb_s = S, s
    return {"unconstrained": (list(best), best_s), "band_separated": (list(bb), bb_s)}


def oracle_gain(S: np.ndarray, y: np.ndarray, bks: dict[str, np.ndarray]) -> float:
    """「逐桶挑最优层」相对「全局最优的单一层」多赚的 AUC(未乘 100)。

    S: (L, N) 各层探针在验证集上的打分。这是路由的**天花板** —— 真实的路由器
    还要自己猜对该用哪层,不可能超过它。
    """
    names = list(bks)
    H = np.array([[auc(S[li][bks[n]], y[bks[n]]) for n in names] for li in range(S.shape[0])])
    fixed = H[int(np.nanargmax(np.nanmean(H, axis=1)))]
    return float(np.nanmean(np.nanmax(H, axis=0)) - np.nanmean(fixed))


def permutation_null(S: np.ndarray, y: np.ndarray, bks: dict[str, np.ndarray],
                     n: int, seed: int = 0) -> np.ndarray:
    """保持桶大小与重叠结构、打散成员,得到「没有任何真实深度偏好时的 oracle 增益」。

    为什么非要有它:跨度/不同取值/并列这三个判据**都没有零假设校准**。桶越小,
    每层的 AUC 估计越噪,argmax 跳得越远 —— 跨度反而越大。实测在官方 val 上
    (桶 89~325)跨度 16 层看着很漂亮,而纯噪声的期望增益就有 +0.569 点,
    点估计 +0.673 几乎全被吃掉(p=0.17)。只看跨度会把噪声当成信号放行。

    注意零假设**不是**打乱标签:那会把 AUC 打到 0.5,方差最大的地方,
    得到的基线虚高且与实际情形无关。这里只打乱「谁属于哪个桶」。
    """
    rng = np.random.default_rng(seed)
    N = S.shape[1]
    return np.array([
        oracle_gain(S, y, {k: rng.choice(N, size=len(v), replace=False) for k, v in bks.items()})
        for _ in range(n)])


def falsification(H: np.ndarray, bucket_names: list[str]) -> dict:
    """最优层是否真的随退化移动?不移动 = 方法前提不成立。"""
    arg = {n: int(np.nanargmax(H[:, i])) for i, n in enumerate(bucket_names)}
    core = [v for k, v in arg.items() if not k.startswith("*")]
    # AUC 打平时 argmax 只返回第一个,选层其实是任意的 —— 必须报出来,
    # 否则会把「任务太简单,所有层都饱和」误读成「第 4 层最好」。
    ties = {}
    for i, n in enumerate(bucket_names):
        col = H[:, i]
        ties[n] = int(np.sum(col >= np.nanmax(col) - 1e-9))
    return {"argmax_per_bucket": arg, "spread": int(max(core) - min(core)),
            "n_distinct": len(set(core)), "n_tied_at_max": ties}


# --------------------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="cache_features.py --layers all 的输出")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--min-bucket", type=int, default=40, help="样本少于此数的桶直接丢弃")
    ap.add_argument("--bucket-mode", default="level", choices=["level", "dim"])
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-prenorm", action="store_true", help="不把 prenorm 两个标量并入探针输入")
    ap.add_argument("--perm", type=int, default=200,
                    help="置换检验次数;0 = 跳过(不建议 —— 跨度判据本身不含零假设校准)")
    ap.add_argument("--per-bucket", action="store_true",
                    help="额外为每(层,桶)单独拟合一个探针,给出「按退化分专家」的上界对照")
    ap.add_argument("--seed", type=int, default=0, help="置换检验的随机种子")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    feats, prenorm, meta, cfg = load_cache(args.cache)
    L = feats.shape[1]
    n_layers = L - 1
    labels = np.array([m["label"] for m in meta], dtype=np.float32)
    codes = np.array([m["code"] for m in meta])

    # 划分:与其他脚本同一套 image_id 哈希,保证各阶段的验证集一致
    if any(m.get("split") for m in meta):
        tr = np.array([i for i, m in enumerate(meta) if m["split"].lower() not in ("val", "valid", "test")])
        va = np.array([i for i, m in enumerate(meta) if m["split"].lower() in ("val", "valid", "test")])
    else:
        h = np.array([int(hashlib.sha256(m["image_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                      for m in meta])
        va, tr = np.flatnonzero(h < args.val_frac), np.flatnonzero(h >= args.val_frac)

    buckets = make_buckets(codes[va], args.bucket_mode, args.min_bucket)
    names = sorted(buckets, key=lambda n: (n.startswith("*"), n))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"缓存      : {args.cache}  ({cfg['backbone'].split('/')[-1]})")
    print(f"码本      : {TAXONOMY_NAME}  {len(DEG_DIMS)} 维 x {N_LEVELS} 档  {DEG_DIMS}")
    if cfg.get("taxonomy") and cfg["taxonomy"] != TAXONOMY_NAME:
        raise SystemExit(
            f"缓存是用码本 {cfg['taxonomy']!r} 建的,当前 SQUADE_TAXONOMY={TAXONOMY_NAME!r} —— "
            f"码字列序会对不上,分桶全错。设成一致再跑。")
    print(f"层        : {L} 个抽头位置 (0=embedding 输出, 1..{n_layers}=各 block 输出)")
    print(f"样本      : 训练 {len(tr)} / 验证 {len(va)}   真假比 {labels.mean():.2f}")
    print(f"桶        : {len(names)} 个 -> " + ", ".join(f"{n}({len(buckets[n])})" for n in names))
    print(f"探针      : Linear({feats.shape[-1] + (0 if args.no_prenorm else 2)}, 1) x {L},"
          f" logistic + L2={args.l2}\n")

    co = cooccurrence(codes)
    print("退化维度共现矩阵(相关系数):"); print(co); print()

    ytr = torch.from_numpy(labels[tr]).to(device)
    H = np.full((L, len(names)), np.nan)
    Hpb = np.full((L, len(names)), np.nan) if args.per_bucket else None

    S = np.zeros((L, len(va)), np.float64)      # 逐层验证集打分,置换检验要用
    for li in range(L):
        Xtr = torch.from_numpy(np.asarray(feats[tr, li], dtype=np.float32))
        Xva = torch.from_numpy(np.asarray(feats[va, li], dtype=np.float32))
        if not args.no_prenorm:
            Xtr = torch.cat([Xtr, torch.from_numpy(np.asarray(prenorm[tr, li], dtype=np.float32))], -1)
            Xva = torch.cat([Xva, torch.from_numpy(np.asarray(prenorm[va, li], dtype=np.float32))], -1)
        # 固定统计量标准化,只用训练集(防泄漏);每层各算一组 —— 不同深度尺度差异极大
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        Xtr, Xva = ((Xtr - mu) / sd).to(device), ((Xva - mu) / sd).to(device)

        w = fit_probe(Xtr, ytr, args.l2)
        s_va = probe_scores(Xva, w)
        S[li] = s_va
        for bi, n in enumerate(names):
            idx = buckets[n]
            H[li, bi] = auc(s_va[idx], labels[va][idx])

        if args.per_bucket:                       # 上界对照:每桶单独配一个头
            btr = make_buckets(codes[tr], args.bucket_mode, args.min_bucket)
            for bi, n in enumerate(names):
                if n not in btr:
                    continue
                wb = fit_probe(Xtr[btr[n]], ytr[btr[n]], args.l2)
                idx = buckets[n]
                Hpb[li, bi] = auc(probe_scores(Xva, wb)[idx], labels[va][idx])
        print(f"  层 {li:>2}/{n_layers}  " + " ".join(f"{n}={H[li, bi]:.3f}" for bi, n in enumerate(names[:4]))
              + (" ..." if len(names) > 4 else ""))

    # ---- 热力图 ----
    hdr = "层  " + "".join(f"{n[:9]:>10}" for n in names)
    lines = [hdr, "-" * len(hdr)]
    col_best = np.nanargmax(H, axis=0)
    for li in range(L):
        row = f"{li:>2}  "
        for bi in range(len(names)):
            v = H[li, bi]
            row += f"{'  n/a' if np.isnan(v) else f'{v * 100:5.1f}'}{'*' if col_best[bi] == li else ' '}".rjust(10)
        lines.append(row)
    lines.append("\n* = 该桶的最优层。数字为 AUC x 100。")
    heat = "\n".join(lines)
    print("\n" + heat)

    # ---- 证伪检查 ----
    fal = falsification(H, names)
    print(f"\n证伪检查:各桶最优层 = {fal['argmax_per_bucket']}")
    print(f"  最优层跨度 {fal['spread']} 层,不同取值 {fal['n_distinct']} 个")
    worst_tie = max(fal["n_tied_at_max"].values())
    if worst_tie > 3:
        print(f"  ⚠️ 有桶在最优 AUC 上并列 {worst_tie} 层 —— argmax 只取第一个,"
              f"选层实质是任意的。多半是任务对该桶太容易(AUC 饱和),"
              f"结论无效;需要更难的数据或更强的退化。")
    # 置换检验才是真正的判据。跨度/不同取值/并列都只描述 argmax 散不散,
    # 没有任何零假设校准 —— 桶越小越容易"通过",方向是反的。
    perm = None
    deg_bks = {n: buckets[n] for n in names if n != "clean" and not n.startswith("*")}
    if args.perm > 0 and len(deg_bks) >= 2:
        point = oracle_gain(S, labels[va], deg_bks)
        null = permutation_null(S, labels[va], deg_bks, args.perm, args.seed)
        pval = float((null >= point).mean())
        perm = {"point": point, "null_mean": float(null.mean()),
                "null_p95": float(np.percentile(null, 95)),
                "net_gain": float(point - null.mean()), "p": pval, "n_perm": args.perm}
        print(f"  置换检验({args.perm} 次, {len(deg_bks)} 个退化桶): "
              f"逐桶最优比单一最优层多赚 {point * 100:+.3f} 点,"
              f"纯噪声均值 {null.mean() * 100:+.3f} 点 -> "
              f"净 {(point - null.mean()) * 100:+.3f} 点, p = {pval:.3f}")

    if worst_tie > 3:
        # 跨度大 + 并列多 = argmax 在饱和的 AUC 上随机跳,跨度是噪声撑出来的,不是信号。
        # 这里以前无条件打 ✅,只看跨度不看并列 —— 并列 27 的时候还打绿勾,极具误导性。
        print("  ❌ 并列层数超标 —— 跨度是 argmax 在饱和 AUC 上乱跳撑出来的,不是真实的深度偏好。"
              "本次分桶的结论**不成立**,换更粗的分桶或加大每桶样本量。")
    elif perm is not None and perm["p"] >= 0.05:
        # 这一支是后加的。此前只看跨度,官方 val 上跨度 16 层、不同取值 12 个,
        # 判定打 ✅ 并写着"可直接作为论文的 Figure 2" —— 而置换检验说 p=0.17,
        # 那 16 层跨度正是纯噪声的典型值。跨度大不等于信号强,只等于桶小。
        print(f"  ❌ 最优层的跨度**没有超过噪声水平**(p = {perm['p']:.3f} ≥ 0.05)。"
              f"跨度 {fal['spread']} 层看似可观,但在 {L} 层上取 max 本身就能刷出 "
              f"{perm['null_mean'] * 100:+.3f} 点。「按退化路由深度」这个自由度**未被证实有价值**,"
              "不要据此去建三个专家。要么加大每桶样本量,要么承认前提不成立。")
    elif fal["spread"] <= 2:
        print("  ⚠️ 所有退化的最优层几乎重合 —— 「按退化路由深度」这个自由度没有价值。"
              "继续往下建三个专家大概率得到 oracle gap ≈ 0。建议先回头检查:"
              "缓存的池化是否丢了浅层证据?退化强度是否太弱?")
    elif perm is None:
        print("  ⚠️ 跨度达标,但 --perm 0 关掉了置换检验 —— 这只说明 argmax 散得开,"
              "**不说明它超过了噪声**。下结论前请开着 --perm 跑一次。")
    else:
        print(f"  ✅ 最优层随退化移动,且显著超过置换零假设(p = {perm['p']:.3f})。方法前提成立。")
    fam_tags = [f"*{f}" for f in FAMILIES if f"*{f}" in names]
    for tag in fam_tags:
        print(f"  {tag:<13} 最优层 = {fal['argmax_per_bucket'][tag]}"
              f"   ({FAMILIES[tag[1:]]})")
    if "*smudged" in names and "*shattered" in names:
        print("  README 主张:smudged 应偏深、shattered 应偏浅。方向对上了这是最强的一张图;"
              "方向反了也是有价值的发现,但整个 story 要重写。")
    elif "*smudged" in names:
        print("  ⚠️ 当前码本里 shattered 族是空的,README 的二分只测到了一半 —— "
              "「shattered 应偏浅」这条**没有被检验**,不要在报告里当作已验证。")

    # ---- 选层 ----
    sel = select_layers(H, names, n_layers, args.k)
    print(f"\n选层(目标 = 各桶取三层中最优者的 AUC 均值):")
    for k2, (S, s) in sel.items():
        print(f"  {k2:<16} {S}   目标值 {s:.4f}")
    chosen = sel["band_separated"][0]
    print(f"\n采用 band_separated: shallow={chosen[0]} mid={chosen[1]} deep={chosen[2]}")
    print(f"  与无约束最优的差距 {sel['unconstrained'][1] - sel['band_separated'][1]:+.4f}"
          f"  (差距大说明最优的三层并不分散,深度多样性是被约束硬加上去的,需在论文里说明)")

    if args.per_bucket:
        d = np.nanmean(Hpb[chosen]) - np.nanmean(H[chosen])
        print(f"\n每桶单独配头 vs 共享头,在选中三层上的平均 AUC 差 = {d:+.4f}")
        print("  这是「按退化分专家」相对「按深度分专家 + 路由」的额外收益,可作对照消融。")

    # ---- 落盘 ----
    np.savetxt(out / "heatmap.csv", H, delimiter=",",
               header="layer_rows_x_" + "|".join(names), comments="")
    (out / "heatmap.txt").write_text(heat + "\n\n" + co + "\n", encoding="utf-8")
    (out / "selected.json").write_text(json.dumps({
        "layers": {"shallow": chosen[0], "mid": chosen[1], "deep": chosen[2]},
        "layers_list": list(chosen),
        "objective": sel["band_separated"][1],
        "unconstrained": {"layers": sel["unconstrained"][0], "objective": sel["unconstrained"][1]},
        "buckets": {n: int(len(buckets[n])) for n in names},
        "falsification": fal,
        "permutation": perm,
        "cache": str(args.cache), "backbone": cfg["backbone"], "pool": cfg["pool"],
        "taxonomy": TAXONOMY_NAME, "deg_dims": DEG_DIMS, "families": {k: list(v) for k, v in FAMILIES.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n写出 {out}/heatmap.csv, heatmap.txt, selected.json")
    print(f"下一步: python cache_features.py --manifest ... --out cache/train "
          f"--layers {','.join(map(str, chosen))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

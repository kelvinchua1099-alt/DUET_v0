"""oracle 增益的自助法 + 置换检验 —— 「逐桶挑最优层」赚到的那点 AUC 里,有多少是真的?

    SQUADE_TAXONOMY=ntireval python tools/bootstrap_oracle.py \
        --cache cache/probe_val --out probe/val

probe_layers.py 的热力图给出

    oracle 增益 = mean_b[ max_ℓ AUC(ℓ,b) ] - mean_b[ AUC(ℓ*,b) ]

其中 ℓ* 是所有桶共用的单一最优层。这个量**天生被高估**:每个桶只有约 100 个验证
样本,33 个层的 AUC 估计各带噪声,取 max 必然落在噪声的上侧(winner's curse)。
实测在 NTIRE val 上它是 +0.37 AUC 点 —— 这个数字直接决定「按退化路由深度」值不值得
建三个专家,所以不能就这么信。

两个对照:

* **bootstrap** 对验证样本有放回重采样,给出点估计的 95% 区间。
* **置换零假设** 保持每个桶的大小,但成员随机抽 —— 「层 x 桶」的结构被破坏,而
  样本量、AUC 水平、33 层取 max 这三件事原样保留。于是它给出的正是
  **完全没有真实逐桶深度偏好时,纯靠 max-over-33-layers 也能刷出来的增益**。

判读:真实增益必须显著高于置换基线。若落在里面,oracle gap 主要是选择偏差,
`train_WeightandExpert.py` 的标准误守卫会在 Stage 2 再拦一次。

注意桶是**可重叠**的(NTIRE val 每图 1~4 种退化,一张图同时进多个维度桶),
所以置换是「每个桶独立无放回抽 |b| 个样本」,而不是把样本划分成不相交的组 ——
后者会顺手抹掉重叠结构,让零假设比实际更容易被超过。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_features import load_cache  # noqa: E402
# oracle_gain 现在住在 probe_layers 里 —— 那边的证伪门禁也要用它,
# 两处各留一份实现迟早会漂移
from probe_layers import (  # noqa: E402
    auc, fit_probe, make_buckets, oracle_gain, probe_scores,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="cache_features.py --layers all 的输出")
    ap.add_argument("--out", required=True, help="结果写到这个目录(与 probe_layers 的 --out 同一个)")
    ap.add_argument("--bucket-mode", default="dim", choices=["level", "dim"])
    ap.add_argument("--min-bucket", type=int, default=60)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    dev = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    feats, prens, meta, cfg = load_cache(args.cache)
    lab = np.array([m["label"] for m in meta], np.float32)
    codes = np.array([m["code"] for m in meta])
    tr = np.array([i for i, m in enumerate(meta) if m["split"] != "val"])
    va = np.array([i for i, m in enumerate(meta) if m["split"] == "val"])
    L = feats.shape[1]
    ytr = torch.from_numpy(lab[tr]).to(dev)
    y = lab[va]

    print(f"缓存 {args.cache}   训练 {len(tr)} / 验证 {len(va)}")
    sc = Path(args.out) / f"probe_scores_l2{args.l2:g}.npy"
    if sc.exists():
        S = np.load(sc)
        if S.shape != (L, len(va)):
            raise SystemExit(f"{sc} 形状 {S.shape} 与当前缓存 {(L, len(va))} 不符,删掉重跑")
        print(f"复用已拟合的探针打分 {sc}", flush=True)
        return _analyze(args, S, lab[va], codes[va])
    print(f"拟合 {L} 个探针,保留验证集打分 ...", flush=True)
    S = np.zeros((L, len(va)), np.float64)
    for li in range(L):
        Xtr = torch.cat([torch.from_numpy(np.asarray(feats[tr, li], np.float32)),
                         torch.from_numpy(np.asarray(prens[tr, li], np.float32))], -1)
        Xva = torch.cat([torch.from_numpy(np.asarray(feats[va, li], np.float32)),
                         torch.from_numpy(np.asarray(prens[va, li], np.float32))], -1)
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        S[li] = probe_scores(((Xva - mu) / sd).to(dev), fit_probe(((Xtr - mu) / sd).to(dev), ytr, args.l2))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.save(sc, S)
    return _analyze(args, S, y, codes[va])


def _analyze(args, S: np.ndarray, y: np.ndarray, codes_va: np.ndarray) -> int:
    va_n = S.shape[1]
    # 只用退化桶。clean 与 * 汇总桶不进目标,与 probe_layers.select_layers 的口径一致
    all_b = make_buckets(codes_va, args.bucket_mode, args.min_bucket)
    buckets = {n: v for n, v in sorted(all_b.items()) if not n.startswith("*") and n != "clean"}
    print(f"退化桶 {len(buckets)} 个: " + ", ".join(f"{n}({len(v)})" for n, v in buckets.items()) + "\n")

    point = oracle_gain(S, y, buckets)
    print(f"点估计 oracle 增益 = {point * 100:+.3f} AUC 点\n")

    rng = np.random.default_rng(args.seed)
    boot, null = [], []
    for b in range(args.n_boot):
        # bootstrap:对验证样本有放回重采样。桶按「重采样后仍落在原桶里的位置」重建。
        pick = rng.integers(0, va_n, va_n)
        bk = {n: np.flatnonzero(np.isin(pick, idx)) for n, idx in buckets.items()}
        if all(len(v) >= 30 for v in bk.values()):
            boot.append(oracle_gain(S[:, pick], y[pick], bk))
        # 置换:保持桶大小与重叠结构,成员随机(每桶独立无放回抽)
        bk2 = {n: rng.choice(va_n, size=len(idx), replace=False) for n, idx in buckets.items()}
        null.append(oracle_gain(S, y, bk2))
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{args.n_boot}", flush=True)

    boot, null = np.array(boot), np.array(null)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    p = float((null >= point).mean())
    net = point - null.mean()
    print(f"\n■ bootstrap  ({len(boot)} 次)  均值 {boot.mean() * 100:+.3f}   "
          f"95% CI [{ci[0] * 100:+.3f}, {ci[1] * 100:+.3f}]")
    print(f"■ 置换零假设 ({len(null)} 次)  均值 {null.mean() * 100:+.3f}   "
          f"95 分位 {np.percentile(null, 95) * 100:+.3f}   (纯噪声也能刷出的增益)")
    print(f"\n★ 净增益 = {point * 100:+.3f} - {null.mean() * 100:+.3f} = {net * 100:+.3f} 点"
          f"   置换 p = {p:.3f}")
    print("  p < 0.05 且净增益明显 > 0 -> 逐桶深度偏好是真的,值得建三个专家;")
    print("  否则 oracle gap 主要是 max-over-33-layers 的选择偏差,先扩验证集再说。")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "oracle_gain_bootstrap.json").write_text(json.dumps({
        "point": point, "boot_mean": float(boot.mean()), "boot_ci95": list(ci),
        "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
        "net_gain": float(net), "perm_p": p, "n_boot": args.n_boot,
        "buckets": {n: int(len(v)) for n, v in buckets.items()},
        "cache": str(args.cache),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}/oracle_gain_bootstrap.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

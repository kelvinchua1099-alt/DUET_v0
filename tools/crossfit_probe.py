"""折外交叉拟合的层级探针 —— 在同一批数据内部测深度偏好,不引入分布偏移。

    SQUADE_TAXONOMY=ntirehard python tools/crossfit_probe.py \
        --cache /workspace/cache/vg_transfer --keep split=val \
        --out /workspace/probe/vh_crossfit --perm 400

要解决的问题:直接对半分会把每个桶也砍一半(val_hard 的桶从 37~931 掉到 ~19~465),
在 41 层上取 max 的噪声基线随之翻倍 —— 实测零假设从 +1.668 涨到 +3.124,
净增益却几乎不动(+0.856 -> +0.770),于是 p 从 0.000 掉到 0.068。**效应还在,
只是没了统计功效。**

交叉拟合把功效拿回来:

    A 半拟合探针 -> 给 B 半打分
    B 半拟合探针 -> 给 A 半打分
    拼成全部 N 个样本的**折外**打分,桶恢复原尺寸

每个样本的打分都来自没见过它的探针,所以不存在过拟合泄漏;拟合与打分同分布,
所以也不存在 val -> val_hard 那种偏移。这是这份数据上能做的最干净的一次测量。

划分按 group;cache_features 的 meta.csv 不写 group 列,退回 image_id —— 对
NTIRE val/val_hard 成立(每图自成一组),对 preprocess 产的同源配对数据**不成立**。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_features import load_cache                                      # noqa: E402
from probe_layers import (                                                 # noqa: E402
    fit_probe, make_buckets, oracle_gain, permutation_null, probe_scores,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default=None, help="col=value,只用满足条件的行")
    ap.add_argument("--bucket-mode", default="dim", choices=["level", "dim"])
    ap.add_argument("--min-bucket", type=int, default=40)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--perm", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    dev = torch.device(a.device) if a.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    feats, prens, meta, cfg = load_cache(a.cache)
    idx = np.arange(len(meta))
    if a.keep:
        col, val = a.keep.split("=", 1)
        idx = np.array([i for i in idx if meta[i][col] == val])
    if len(idx) == 0:
        raise SystemExit(f"--keep {a.keep} 筛出 0 行")

    gkey = "group" if "group" in meta[0] else "image_id"
    gs = [meta[i][gkey] for i in idx]
    half = {g for g in set(gs)
            if int(hashlib.sha256(f"{a.seed}|{g}".encode()).hexdigest()[:8], 16) % 2 == 0}
    fold = np.array([0 if g in half else 1 for g in gs])
    y = np.array([float(meta[i]["label"]) for i in idx])
    codes = np.array([meta[i]["code"] for i in idx])
    L = feats.shape[1]

    print(f"缓存 {a.cache}   --keep {a.keep}   {len(idx)} 行   按 {gkey} 分折")
    print(f"  折 0 = {int((fold == 0).sum())} 行   折 1 = {int((fold == 1).sum())} 行")
    print(f"拟合 {L} 层 x 2 折,取折外打分 ...", flush=True)

    S = np.zeros((L, len(idx)), np.float64)
    for li in range(L):
        X = torch.cat([torch.from_numpy(np.asarray(feats[idx, li], np.float32)),
                       torch.from_numpy(np.asarray(prens[idx, li], np.float32))], -1)
        for f in (0, 1):
            tr, va = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            # 标准化统计量只在该折的训练侧上标定 —— 否则打分侧的分布会漏进来
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
            w = fit_probe(((X[tr] - mu) / sd).to(dev),
                          torch.from_numpy(y[tr]).to(dev), a.l2)
            S[li, va] = probe_scores(((X[va] - mu) / sd).to(dev), w)
        if li % 10 == 0:
            print(f"  层 {li}/{L}", flush=True)

    all_b = make_buckets(codes, a.bucket_mode, a.min_bucket)
    bks = {n: v for n, v in sorted(all_b.items()) if n != "clean" and not n.startswith("*")}
    print(f"\n退化桶 {len(bks)} 个: " + ", ".join(f"{n}({len(v)})" for n, v in bks.items()))

    point = oracle_gain(S, y, bks)
    null = permutation_null(S, y, bks, a.perm, a.seed)
    p = float((null >= point).mean())
    net = point - null.mean()
    print(f"\n折外 oracle 增益 = {point * 100:+.3f} 点")
    print(f"置换零假设({a.perm} 次) 均值 {null.mean() * 100:+.3f}   "
          f"95 分位 {np.percentile(null, 95) * 100:+.3f}")
    print(f"\n★ 净增益 {net * 100:+.3f} 点   p = {p:.4f}   "
          f"-> {'✅ 深度偏好显著' if p < 0.05 else '❌ 未超过噪声'}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "crossfit.json").write_text(json.dumps({
        "point": point, "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)), "net_gain": float(net),
        "p": p, "n_perm": a.perm, "n_rows": int(len(idx)),
        "buckets": {n: int(len(v)) for n, v in bks.items()}, "cache": str(a.cache),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}/crossfit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

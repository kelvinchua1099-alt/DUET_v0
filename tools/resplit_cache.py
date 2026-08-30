"""按 meta 的某一列筛出子集、并重新划分 train/val —— 不重抽特征。

    python tools/resplit_cache.py --src /workspace/cache/vg_transfer \
        --out /workspace/cache/vh_only --keep subset=val_hard --val-frac 0.5

为什么需要它:vg_transfer 的划分是「val 拟合 / val_hard 打分」,那里存在**分布偏移**
(探针 AUC 从 ~95 掉到 ~82),逐桶最优层的差异可能反映的是"哪层在偏移下更稳",
而不是"哪层适合这种退化"。要排除这个替代解释,就得在 val_hard **内部**再测一次:
同一批分布、同样的分桶,只是拟合与打分都在里面。特征都在缓存里,没必要重抽 12500 张。

划分按 **group** 做,与 preprocess 的纪律 7 一致 —— 同一源图不能跨 split。
NTIRE val/val_hard 每图自成一组,所以这里等价于按图划分。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default=None, help="形如 col=value,只留该列等于 value 的行")
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    meta = list(csv.DictReader(open(src / "meta.csv", newline="", encoding="utf-8")))
    idx = list(range(len(meta)))
    if a.keep:
        col, val = a.keep.split("=", 1)
        if col not in meta[0]:
            raise SystemExit(f"meta.csv 没有列 {col!r};有的是 {sorted(meta[0])}")
        idx = [i for i in idx if meta[i][col] == val]
    if not idx:
        raise SystemExit(f"--keep {a.keep} 筛出 0 行")
    rows = [meta[i] for i in idx]

    # 按 group 重划 —— 同一源图必须同侧。cache_features 的 meta.csv 不带 group 列
    # (它只记 image_id),NTIRE val/val_hard 每图自成一组,退回 image_id 是等价的;
    # 但对 preprocess 产的数据(同源图有干净版+退化版)就**不等价**,所以要明确报出来。
    gkey = "group" if "group" in rows[0] else "image_id"
    if gkey != "group":
        print(f"[注意] meta.csv 无 group 列,按 {gkey} 划分。"
              f"这对 NTIRE val/val_hard 成立(每图自成一组),对有同源配对的数据**不成立**。")
    for r in rows:
        r.setdefault("group", r[gkey])
    groups = sorted({r["group"] for r in rows},
                    key=lambda g: hashlib.sha256(f"{a.seed}|{g}".encode()).hexdigest())
    k = int(round(len(groups) * a.val_frac))
    va_groups = set(groups[:k])
    for r in rows:
        r["split"] = "val" if r["group"] in va_groups else "train"

    out.mkdir(parents=True, exist_ok=True)
    sel = np.asarray(idx)
    for name in ("features.npy", "prenorm.npy"):
        arr = np.load(src / name, mmap_mode="r")
        np.save(out / name, np.asarray(arr[sel]))
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cfg = json.loads((src / "config.json").read_text())
    cfg["derived_from"] = str(src)
    cfg["derived_op"] = f"resplit_cache --keep {a.keep} --val-frac {a.val_frac} --seed {a.seed}"
    # 行集变了,原来的 manifest 指纹不再描述这份缓存 —— 留着会让下游误判成同一批数据
    cfg.pop("manifest_sha1", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    bad = sum(1 for g, v in {g: {r["split"] for r in rows if r["group"] == g}
                             for g in list(va_groups)[:200]}.items() if len(v) > 1)
    print(f"{len(meta)} -> {len(rows)} 行   组 {len(groups)}")
    print(f"  split {dict(Counter(r['split'] for r in rows))}")
    print(f"  label {dict(Counter(r['label'] for r in rows))}")
    print(f"  跨 split 的组(抽查 200 个): {bad}  (必须为 0)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

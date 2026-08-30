"""把多份缓存按行拼成一份 —— 不重抽特征。

    python tools/concat_cache.py --out /workspace/cache/mix \
        --src /workspace/cache/adv_vg6 \
        --src /workspace/cache/vgval6:train:/workspace/data/val_split_70_30.json

每个 --src 形如 `路径[:split标记[:白名单json]]`:
  * split标记  写进 meta 的 split 列(train / val)。省略则沿用原缓存的。
  * 白名单json  形如 {"train": [...], "held": [...]},只保留 image_id 在 train 里的行。
    用来把官方 val 的 70% 并进训练集,而 30% 一张都不进。

**必须校验的三件事**(不一致就直接拒绝,否则拼出来的东西是静默错的):
  1. backbone 与 layers 完全相同 —— 否则不同深度的特征会被当成同一层
  2. hidden_size 相同
  3. meta 的列集合相同 —— 码本不同的两份缓存拼起来,deg_* 列会错位

拼完不重划 split:训练/评测的归属由各源自己定,这里只负责搬运。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(d: str):
    p = Path(d)
    cfg = json.loads((p / "config.json").read_text())
    rows = list(csv.DictReader(open(p / "meta.csv", newline="", encoding="utf-8")))
    return p, cfg, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True,
                    help="路径[:split标记[:白名单json]],可重复")
    a = ap.parse_args(argv)

    specs = []
    for s in a.src:
        parts = s.split(":")
        specs.append((parts[0], parts[1] if len(parts) > 1 and parts[1] else None,
                      parts[2] if len(parts) > 2 else None))

    base_cfg, base_cols = None, None
    plans = []
    for path, mark, wl in specs:
        p, cfg, rows = load(path)
        if base_cfg is None:
            base_cfg, base_cols = cfg, set(rows[0])
        else:
            for k in ("backbone", "layers", "hidden_size"):
                if cfg.get(k) != base_cfg.get(k):
                    raise SystemExit(f"{path} 的 {k}={cfg.get(k)!r} 与首个源 "
                                     f"{base_cfg.get(k)!r} 不一致 —— 拼起来会静默错位")
            if set(rows[0]) != base_cols:
                raise SystemExit(f"{path} 的 meta 列与首个源不一致:\n"
                                 f"  多出 {sorted(set(rows[0]) - base_cols)}\n"
                                 f"  缺少 {sorted(base_cols - set(rows[0]))}")
        idx = list(range(len(rows)))
        if wl:
            keep = set(json.loads(Path(wl).read_text())["train"])
            idx = [i for i in idx if rows[i]["image_id"] in keep]
            if not idx:
                raise SystemExit(f"{path} 按 {wl} 的白名单筛出 0 行")
        sel = [rows[i] for i in idx]
        if mark:
            for r in sel:
                r["split"] = mark
        plans.append((p, np.asarray(idx), sel))
        print(f"  {path:<40} {len(rows):>7} -> {len(sel):>7} 行"
              + (f"   split={mark}" if mark else "")
              + (f"   白名单 {Path(wl).name}" if wl else ""))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in ("features.npy", "prenorm.npy"):
        parts = []
        for p, idx, _ in plans:
            arr = np.load(p / name, mmap_mode="r")
            parts.append(np.asarray(arr[idx]))
        stacked = np.concatenate(parts, axis=0)
        np.save(out / name, stacked)
        print(f"  {name}: {stacked.shape}  {stacked.nbytes/1e9:.2f} GB")

    rows_out = [r for _, _, sel in plans for r in sel]
    for i, r in enumerate(rows_out):
        r["idx"] = i                                  # idx 必须重编,否则下游按 idx 取会错
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)

    cfg = dict(base_cfg)
    cfg["derived_from"] = [s[0] for s in specs]
    cfg["derived_op"] = "concat_cache"
    cfg.pop("manifest_sha1", None)                    # 行集变了,原指纹不再描述这份缓存
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    from collections import Counter
    print(f"\n-> {out}   共 {len(rows_out)} 行   层 {cfg['layers']}")
    print(f"   split {dict(Counter(r.get('split','?') for r in rows_out))}")
    print(f"   label {dict(Counter(r['label'] for r in rows_out))}")
    ids = [r["image_id"] for r in rows_out]
    dup = len(ids) - len(set(ids))
    print(f"   重复 image_id: {dup}  (必须为 0)")
    return 1 if dup else 0


if __name__ == "__main__":
    raise SystemExit(main())

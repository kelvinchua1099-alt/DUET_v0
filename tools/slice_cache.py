"""从「全层」缓存里切出指定的几层,省掉一次完整的特征重抽。

    python tools/slice_cache.py --src /workspace/cache/probe_deg2 \
           --out /workspace/cache/train_deg2 --layers 20,24,28

features.npy 的布局是 (N, L, D),L 与 config.json 的 "layers" 一一对应,
所以按层取子集就是一次纯数组切片 —— 不需要重跑 DINOv3(20,000 行要 25 分钟)。

写出的 config.json 与直接用 cache_features --layers 20,24,28 得到的等价:
layers 换成子集,_fingerprint 重算(否则下游拿旧指纹去比会误判成配置不符)。
manifest_sha1 原样保留 —— 数据没变,变的只是留了哪几层。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import fingerprint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", required=True, help="逗号分隔,如 20,24,28")
    a = ap.parse_args(argv)

    src, out = Path(a.src), Path(a.out)
    cfg = json.loads((src / "config.json").read_text())
    want = [int(x) for x in a.layers.split(",")]
    have = list(cfg["layers"])
    missing = [l for l in want if l not in have]
    if missing:
        raise SystemExit(f"源缓存里没有这些层: {missing}(它有 {have})")
    idx = [have.index(l) for l in want]

    feats = np.load(src / "features.npy", mmap_mode="r")
    prens = np.load(src / "prenorm.npy", mmap_mode="r")
    if feats.shape[1] != len(have):
        raise SystemExit(f"features 第 1 维 {feats.shape[1]} 与 config 的层数 {len(have)} 不符")

    out.mkdir(parents=True, exist_ok=True)
    print(f"源 {feats.shape} -> 取层 {want}(下标 {idx})")
    # 逐块写,避免把整个 (N,33,3840) 拉进内存
    fo = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=feats.dtype,
                                   shape=(feats.shape[0], len(idx), feats.shape[2]))
    po = np.lib.format.open_memmap(out / "prenorm.npy", mode="w+", dtype=prens.dtype,
                                   shape=(prens.shape[0], len(idx), prens.shape[2]))
    step = 2048
    for b in range(0, feats.shape[0], step):
        fo[b : b + step] = feats[b : b + step][:, idx, :]
        po[b : b + step] = prens[b : b + step][:, idx, :]
    fo.flush(); po.flush()

    shutil.copy(src / "meta.csv", out / "meta.csv")
    cfg["layers"] = want
    cfg["_sliced_from"] = str(src)
    # 复用 cache_features.fingerprint:它只取 11 个指定键,不是"所有非下划线键"。
    # 自己再实现一遍,日后那份键列表一改就会静默不同步。
    cfg["_fingerprint"] = fingerprint(cfg)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(f"-> {out}  {fo.shape}  ({fo.nbytes / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

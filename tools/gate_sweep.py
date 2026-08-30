"""二元门的改进实验:硬路由阈值扫描 vs 软融合 vs 直接预测"该听哪组"。

    SQUADE_TAXONOMY=synthetic python tools/gate_sweep.py \
        --shallow-run runs/vg_shallow --deep-run runs/vg_deep --gate runs/gate.pt \
        --caches vgval_shallow,vgval_deep:官方val:all vgvh_shallow,vgvh_deep:val_hard:val

为什么值得扫:硬路由把门的每一次错判都放大成"整组换掉"的代价。实测门在官方数据上
判对率只有 74.8%,而两组之间的差在 val_hard 上有 5 个点 —— 四分之一的样本吃满这个差。
软融合按门的置信度加权,错判只按置信度比例损失,不会一步踩空。

阈值 τ 也要扫:门在两份官方数据上都把 55~58% 判成干净(真实 50%),说明它系统性偏向
"干净",τ=0 不是最优切点。**τ 必须在一份上选、另一份上报**,否则就是在测试集上调参。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.experts_mlp import ExpertBank                      # noqa: E402
from training.train_WeightandExpert import auc                 # noqa: E402


def read_cache(d):
    p = Path(d)
    return (np.load(p / "features.npy", mmap_mode="r"),
            np.load(p / "prenorm.npy", mmap_mode="r"),
            list(csv.DictReader(open(p / "meta.csv", newline="", encoding="utf-8"))))


def expert_scores(run, cache, idx, dev):
    f, p, _ = read_cache(cache)
    ck = torch.load(Path(run) / "stage1.pt", map_location="cpu")
    b = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                   dropout=ck["cfg"]["dropout"])
    b.load_state_dict(ck["bank"]); b.to(dev).freeze_experts()
    o = []
    for k in range(0, len(idx), 512):
        j = idx[k:k + 512]
        with torch.no_grad():
            z, _ = b(torch.from_numpy(np.asarray(f[j], np.float32).copy()).to(dev),
                     torch.from_numpy(np.asarray(p[j], np.float32).copy()).to(dev),
                     return_parts=True)
        o.append(z.cpu().numpy())
    return np.concatenate(o)


def gate_scores(gate, cache, idx, dev):
    g = torch.load(gate, map_location="cpu")
    net = nn.Sequential(nn.Linear(g["D"], g["hidden"]), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(g["hidden"], 1))
    net.load_state_dict(g["net"]); net.to(dev).eval()
    f, _, _ = read_cache(cache)
    mu, sd = g["mu"].to(dev), g["sd"].to(dev)
    o = []
    with torch.no_grad():
        for k in range(0, len(idx), 512):
            j = idx[k:k + 512]
            x = torch.from_numpy(np.asarray(f[j], np.float32).copy())
            o.append(net(((x.reshape(x.shape[0], -1).to(dev)) - mu) / sd).squeeze(-1).cpu().numpy())
    return np.concatenate(o)


def rank(z):
    return np.argsort(np.argsort(z)) / (len(z) - 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shallow-run", required=True)
    ap.add_argument("--deep-run", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--caches", nargs="+", required=True,
                    help="每项形如 shallowCache,deepCache:标签:split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    D = {}
    for spec in a.caches:
        pair, tag, sp = spec.split(":")
        cs, cd = (f"/workspace/cache/{x}" for x in pair.split(","))
        _, _, rows = read_cache(cs)
        idx = np.array([i for i, r in enumerate(rows) if sp == "all" or r.get("split") == sp])
        rr = [rows[i] for i in idx]
        y = np.array([float(r["label"]) for r in rr])
        dc = [c for c in rr[0] if c.startswith("deg_")]
        clean = np.array([sum(int(r[c]) for c in dc) == 0 for r in rr])
        D[tag] = dict(y=y, clean=clean,
                      rs=rank(expert_scores(a.shallow_run, cs, idx, a.device)),
                      rd=rank(expert_scores(a.deep_run, cd, idx, a.device)),
                      gz=gate_scores(a.gate, cs, idx, a.device))
        print(f"{tag}: {len(idx)} 行  干净 {int(clean.sum())}  "
              f"门 AUC {auc(D[tag]['gz'], (~clean).astype(float)):.4f}")

    def report(name, fn):
        out = []
        for tag, d in D.items():
            z = fn(d)
            out.append((auc(z, d["y"]), auc(z[d["clean"]], d["y"][d["clean"]]),
                        auc(z[~d["clean"]], d["y"][~d["clean"]])))
        cells = "".join(f"{o[0]:>9.4f}{o[1]:>9.4f}{o[2]:>9.4f}" for o in out)
        print(f"  {name:<26}{cells}")

    hdr = "".join(f"{'全量':>9}{'clean':>9}{'robust':>9}" for _ in D)
    print(f"\n{'':<28}" + "".join(f"{t:^27}" for t in D))
    print(f"  {'方案':<26}{hdr}")
    print("  " + "-" * (26 + 27 * len(D)))
    report("shallow-only", lambda d: d["rs"])
    report("deep-only", lambda d: d["rd"])
    report("oracle route", lambda d: np.where(d["clean"], d["rs"], d["rd"]))
    report("hard route τ=0 (现状)", lambda d: np.where(d["gz"] <= 0, d["rs"], d["rd"]))

    print("\n  ── 硬路由:阈值扫描 ──")
    for t in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        report(f"hard τ={t:+.1f}", lambda d, t=t: np.where(d["gz"] <= t, d["rs"], d["rd"]))

    print("\n  ── 软融合:按门的置信度加权(T = 温度,越大越平滑) ──")
    for T in (0.5, 1.0, 2.0, 4.0, 8.0):
        def f(d, T=T):
            w = 1.0 / (1.0 + np.exp(-d["gz"] / T))     # w=P(退化)
            return (1 - w) * d["rs"] + w * d["rd"]
        report(f"soft T={T:.1f}", f)
    report("固定 50/50 平均", lambda d: 0.5 * d["rs"] + 0.5 * d["rd"])
    print("\n  注:τ 与 T 必须在一份数据上选、另一份上报,否则是在测试集上调参。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

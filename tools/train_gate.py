"""二元判别器:这张图退化了吗 —— 决定走浅组还是深组。

    python tools/train_gate.py --cache /workspace/cache/ds_shallow --out /workspace/runs/gate.pt

**读的是浅组的特征**,这一点是架构上的关键:推理时先跑到浅组最深那层(L27),
此时浅组三个专家和这个门都已经算得出来。门说"干净"就直接出浅组的分并**提前退出**,
省掉 L28~L37 那 10 个 block;说"退化"才继续跑完深组。门若改读深层特征,
就必须先跑完全深度,提前退出的收益整个消失。

目标 = 是否施加过退化(码本的 d0),不是真假。训练/验证沿用缓存里的 split,
所以不会用验证集的分布去标定任何东西。
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
from training.train_WeightandExpert import auc                 # noqa: E402


def read_cache(d):
    p = Path(d)
    feats = np.load(p / "features.npy", mmap_mode="r")
    rows = list(csv.DictReader(open(p / "meta.csv", newline="", encoding="utf-8")))
    return feats, rows, json.loads((p / "config.json").read_text())


def flat(feats, idx, dev):
    x = torch.from_numpy(np.asarray(feats[idx], np.float32).copy())
    return x.reshape(x.shape[0], -1).to(dev)                   # (B, 3*3*H)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="浅组三层缓存")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)

    feats, rows, cfg = read_cache(a.cache)
    degcols = [c for c in rows[0] if c.startswith("deg_")]
    y = np.array([1.0 if sum(int(r[c]) for c in degcols) else 0.0 for r in rows], np.float32)
    tr = np.array([i for i, r in enumerate(rows) if r.get("split") != "val"])
    va = np.array([i for i, r in enumerate(rows) if r.get("split") == "val"])
    D = int(np.prod(feats.shape[1:]))
    print(f"缓存 {a.cache}  层 {cfg['layers']}  维 {D}  训练 {len(tr)} / 验证 {len(va)}")
    print(f"  退化占比 train {y[tr].mean():.3f}  val {y[va].mean():.3f}")

    # 标准化统计量只在训练集上标定(纪律 3)
    sub = tr[np.random.default_rng(a.seed).choice(len(tr), min(20000, len(tr)), replace=False)]
    xs = flat(feats, np.sort(sub), "cpu")
    mu, sd = xs.mean(0), xs.std(0).clamp_min(1e-6)

    net = nn.Sequential(nn.Linear(D, a.hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(a.hidden, 1)).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()
    mu_d, sd_d = mu.to(a.device), sd.to(a.device)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = np.random.default_rng(a.seed + ep).permutation(tr)
        run = 0.0
        for k in range(0, len(perm), a.batch_size):
            j = np.sort(perm[k : k + a.batch_size])
            x = (flat(feats, j, a.device) - mu_d) / sd_d
            t = torch.from_numpy(y[j]).to(a.device)
            opt.zero_grad()
            l = lossf(net(x).squeeze(-1), t)
            l.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run += float(l) * len(j)
        net.eval(); zs = []
        with torch.no_grad():
            for k in range(0, len(va), 1024):
                j = va[k : k + 1024]
                zs.append(net(((flat(feats, j, a.device) - mu_d) / sd_d)).squeeze(-1).cpu().numpy())
        z = np.concatenate(zs)
        A = auc(z, y[va]); acc = float(((z > 0) == (y[va] > 0.5)).mean())
        star = ""
        if A > best:
            best = A
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "D": D,
                        "hidden": a.hidden, "layers": cfg["layers"], "val_auc": A}, a.out)
            star = " *"
        print(f"[{ep:>2}/{a.epochs}] train={run/len(tr):.4f}  valAUC={A:.4f}  acc={acc:.3f}{star}")
    print(f"\n-> {a.out}   最好 val AUC {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

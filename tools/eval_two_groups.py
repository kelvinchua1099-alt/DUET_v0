"""浅/深两组专家在官方 val / val_hard 上的评测。

    python tools/eval_two_groups.py \
        --shallow-run /workspace/runs/vg_shallow --shallow-cache /workspace/cache/vgval_shallow \
        --deep-run    /workspace/runs/vg_deep    --deep-cache    /workspace/cache/vgval_deep \
        --tag "官方 val"  [--gate /workspace/runs/gate.pt]

报四个数,缺一不可:

    shallow-only   全部走浅组 [14,21,27]
    deep-only      全部走深组 [26,33,37]
    oracle route   用**真值**的 clean/degraded 挑组 —— 天花板,不可实现
    gated route    用训练出来的判别器挑组 —— 真实可交付的数

只报 oracle 是常见的自欺:实测二元门有错误率,而错一张的代价在 val_hard 上是 5 个 AUC 点。
oracle 与 gated 的差,就是这个门还欠多少。

**故意不依赖码本**:模型用 synthetic(6 维)训,而 val 是 ntireval(19 维)、val_hard 是
ntirehard(29 维)。走 load_cache 会拿 ACTIVE 码本去索引 meta 的 deg_* 列,直接 KeyError。
这里改成直接读 meta.csv,把所有 deg_* 列加起来判断"是否退化",与码本无关。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.experts_mlp import ExpertBank                      # noqa: E402
from training.train_WeightandExpert import auc, auc_se         # noqa: E402


def read_cache(d: str):
    """不经 load_cache —— 避开码本耦合。返回 (feats, prenorm, rows, cfg)。"""
    p = Path(d)
    cfg = json.loads((p / "config.json").read_text())
    feats = np.load(p / "features.npy", mmap_mode="r")
    prens = np.load(p / "prenorm.npy", mmap_mode="r")
    rows = list(csv.DictReader(open(p / "meta.csv", newline="", encoding="utf-8")))
    return feats, prens, rows, cfg


def scores(run: str, cache: str, idx: np.ndarray, device: str) -> np.ndarray:
    feats, prens, _, _ = read_cache(cache)
    ck = torch.load(Path(run) / "stage1.pt", map_location="cpu")
    bank = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                      dropout=ck["cfg"]["dropout"])
    bank.load_state_dict(ck["bank"])
    bank.to(device).freeze_experts()
    out = []
    for k in range(0, len(idx), 512):
        j = idx[k : k + 512]
        f = torch.from_numpy(np.asarray(feats[j], np.float32).copy()).to(device)
        p = torch.from_numpy(np.asarray(prens[j], np.float32).copy()).to(device)
        with torch.no_grad():
            z, _ = bank(f, p, return_parts=True)         # weights=None -> 均匀融合
        out.append(z.cpu().numpy())
    return np.concatenate(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shallow-run", required=True)
    ap.add_argument("--shallow-cache", required=True)
    ap.add_argument("--deep-run", required=True)
    ap.add_argument("--deep-cache", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--split", default="val", help="只评这个 split;all=全部")
    ap.add_argument("--held-out", default=None,
                    help="val_split_70_30.json —— 只评其中 held 名单里的图。"
                         "新模型训过这份数据的 train 部分,不加这个开关就是在训练集上自评")
    ap.add_argument("--gate", default=None, help="tools/train_gate.py 产出的判别器")
    ap.add_argument("--min-n", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    _, _, rows_s, cfg_s = read_cache(a.shallow_cache)
    _, _, rows_d, cfg_d = read_cache(a.deep_cache)
    if [r["image_id"] for r in rows_s] != [r["image_id"] for r in rows_d]:
        raise SystemExit("两个缓存的行序不一致 —— 必须由同一份全层缓存切出来")

    idx = np.array([i for i, r in enumerate(rows_s)
                    if a.split == "all" or r.get("split", "") == a.split])
    if a.held_out:
        keep = set(json.loads(Path(a.held_out).read_text())["held"])
        before = len(idx)
        idx = np.array([i for i in idx if rows_s[i]["image_id"] in keep])
        print(f"  [held-out] 按 {Path(a.held_out).name} 过滤:{before} -> {len(idx)} 行"
              f"(训练用的 train 名单已排除)")
        if not len(idx):
            raise SystemExit("held-out 名单与该缓存无交集")
    if not len(idx):
        raise SystemExit(f"--split {a.split} 选出 0 行")
    rows = [rows_s[i] for i in idx]
    y = np.array([float(r["label"]) for r in rows])
    degcols = [c for c in rows[0] if c.startswith("deg_")]
    sev = np.array([sum(int(r[c]) for c in degcols) for r in rows])
    clean = sev == 0

    zs = scores(a.shallow_run, a.shallow_cache, idx, a.device)
    zd = scores(a.deep_run, a.deep_cache, idx, a.device)

    print(f"\n{'='*76}\n{a.tag or a.shallow_cache}   {len(idx)} 行 "
          f"(干净 {int(clean.sum())} / 退化 {int((~clean).sum())})   "
          f"假 {int(y.sum())} / 真 {int((1-y).sum())}")
    print(f"  浅组层 {cfg_s['layers']}   深组层 {cfg_d['layers']}")
    se = auc_se(0.95, int(y.sum()), int((1 - y).sum())) * 100
    print(f"\n  {'方案':<18}{'全量':>10}{'clean':>10}{'robust':>10}    (±SE {se:.2f} 点)")
    print("  " + "-" * 56)

    def line(tag, z):
        c = f"{auc(z[clean], y[clean]):.4f}" if clean.sum() >= a.min_n else "  -  "
        r = f"{auc(z[~clean], y[~clean]):.4f}" if (~clean).sum() >= a.min_n else "  -  "
        print(f"  {tag:<18}{auc(z, y):>10.4f}{c:>10}{r:>10}")

    line("shallow-only", zs)
    line("deep-only", zd)
    # 组内先做秩归一化再拼 —— 两组的 logit 尺度不同,直接按掩码拼会在 clean/deg 交界
    # 处制造一个人为的分数断层,全量 AUC 会被这个断层而不是判别力决定
    rs = np.argsort(np.argsort(zs)) / (len(zs) - 1)
    rd = np.argsort(np.argsort(zd)) / (len(zd) - 1)
    line("oracle route", np.where(clean, rs, rd))

    if a.gate:
        import torch.nn as nn
        g = torch.load(a.gate, map_location="cpu")
        net = nn.Sequential(nn.Linear(g["D"], g["hidden"]), nn.GELU(), nn.Dropout(0.1),
                            nn.Linear(g["hidden"], 1))
        net.load_state_dict(g["net"]); net.to(a.device).eval()
        fs, _, _, _ = read_cache(a.shallow_cache)
        mu, sd = g["mu"].to(a.device), g["sd"].to(a.device)
        gz = []
        with torch.no_grad():
            for k in range(0, len(idx), 512):
                j = idx[k : k + 512]
                x = torch.from_numpy(np.asarray(fs[j], np.float32).copy())
                x = x.reshape(x.shape[0], -1).to(a.device)
                gz.append(net((x - mu) / sd).squeeze(-1).cpu().numpy())
        gz = np.concatenate(gz)
        pred_clean = gz <= 0                       # 门 <=0 判"干净" -> 走浅组、提前退出
        line("gated route", np.where(pred_clean, rs, rd))
        agree = float((pred_clean == clean).mean())
        print(f"\n  门的表现: AUC(退化 vs 干净) {auc(gz, (~clean).astype(float)):.4f}   "
              f"判对率 {agree*100:.1f}%   判为干净的比例 {pred_clean.mean()*100:.1f}% "
              f"(真实 {clean.mean()*100:.1f}%)")
        print(f"  提前退出省下的算力: {pred_clean.mean() * (1 - 27/40) * 100:.1f}% "
              f"(干净图跑到 L27 就停,省掉 L28~L37 共 10/40 个 block)")

    # 逐退化维
    print(f"\n  {'退化维':<34}{'n':>7}{'shallow':>10}{'deep':>10}{'差':>8}")
    print("  " + "-" * 69)
    for c in sorted(degcols):
        m = np.array([int(r[c]) > 0 for r in rows])
        if m.sum() < a.min_n or len(set(y[m])) < 2:
            continue
        s, d = auc(zs[m], y[m]), auc(zd[m], y[m])
        print(f"  {c[4:]:<34}{int(m.sum()):>7}{s:>10.4f}{d:>10.4f}{(d-s)*100:>+8.2f}")
    print("  注:各维桶可重叠(官方 val 每图 1~4 种退化),不能当独立观测。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

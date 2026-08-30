"""按数据来源分层报结果 —— 一个混合数字会掩盖各来源之间的此消彼长。

    SQUADE_TAXONOMY=synthetic python tools/eval_by_source.py \
        --run runs/ds_20_24_28 --cache /workspace/cache/ds_20_24_28 \
        --manifest /workspace/data/dataset/manifest.csv

为什么必须分层:合并数据集里各来源的难度差一个量级(CIFAKE 是 32x32 上采样 16 倍,
单独测 AUC 只有 0.82;SID/NTIRE 是原生 512,能到 0.97+)。CIFAKE 只占 14% 行数,
混着算的话它的表现会被完全稀释掉,看不出模型在最难的那部分上到底行不行。

clean / robust 也必须分开:那是竞赛的两个官方指标,而且把两者混进同一个排序算 AUC
会掺进"退化=更像假"的伪相关,得到的数既不是 clean 也不是 robust。

meta.csv 里没有 source 列(cache_features 不写它),所以按 image_id 回连 manifest 取。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import load_cache                                   # noqa: E402
from models.experts_mlp import BANDS, ExpertBank                        # noqa: E402
from training.train_WeightandExpert import auc, auc_se                  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True, help="恰好 3 层的缓存")
    ap.add_argument("--manifest", required=True, help="带 source 列的 manifest")
    ap.add_argument("--split", default="val", choices=["val", "all"])
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    src_of = {}
    with open(a.manifest, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            src_of[r["image_id"]] = r.get("source", "?")

    feats, pren, meta, cfg = load_cache(a.cache)
    idx = (list(range(len(meta))) if a.split == "all"
           else [i for i, m in enumerate(meta)
                 if m.get("split", "").lower() in ("val", "valid", "test")])
    if not idx:
        raise SystemExit("没有可用样本")

    ck = torch.load(Path(a.run) / "stage1.pt", map_location="cpu")
    bank = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                      dropout=ck["cfg"]["dropout"])
    bank.load_state_dict(ck["bank"])
    bank.to(a.device).freeze_experts()

    zs = []
    for k in range(0, len(idx), 512):
        j = idx[k : k + 512]
        f = torch.from_numpy(np.asarray(feats[j], dtype=np.float32).copy()).to(a.device)
        p = torch.from_numpy(np.asarray(pren[j], dtype=np.float32).copy()).to(a.device)
        with torch.no_grad():
            _, parts = bank(f, p, return_parts=True)
        zs.append(parts["expert_logits"].cpu())
    Z = torch.cat(zs).numpy()
    z = Z.mean(-1)                                    # A2:均匀权重
    y = np.array([float(meta[i]["label"]) for i in idx])
    sev = np.array([sum(meta[i]["code"]) for i in idx])
    src = np.array([src_of.get(meta[i]["image_id"], "?") for i in idx])
    rob = sev > 0

    print(f"模型 {a.run}   抽头层 {cfg['layers']}   样本 {len(idx)} (--split {a.split})\n")

    def line(tag, m):
        n = int(m.sum())
        if n < a.min_n or len(set(y[m])) < 2:
            return
        c, r = m & ~rob, m & rob
        f_c = f"{auc(z[c], y[c]):.4f}" if c.sum() >= a.min_n and len(set(y[c])) == 2 else "  -  "
        f_r = f"{auc(z[r], y[r]):.4f}" if r.sum() >= a.min_n and len(set(y[r])) == 2 else "  -  "
        se = auc_se(0.95, int(y[m].sum()), int((1 - y[m]).sum())) * 100
        print(f"  {tag:<16}{n:>8}{int(y[m].sum()):>8}{int((1-y[m]).sum()):>8}"
              f"{auc(z[m], y[m]):>10.4f}{f_c:>10}{f_r:>10}{se:>9.2f}")

    print(f"  {'来源':<16}{'n':>8}{'假':>8}{'真':>8}{'全量':>10}{'clean':>10}{'robust':>10}{'±SE(点)':>9}")
    print("  " + "-" * 80)
    line("全部", np.ones(len(y), bool))
    print("  " + "-" * 80)
    for s in sorted(set(src)):
        line(s, src == s)

    print(f"\n  逐专家(全量): " +
          "  ".join(f"{b}={auc(Z[:, i], y):.4f}" for i, b in enumerate(BANDS)))
    print("  注:只有同时含真假两类的来源才能算 AUC —— WildFake 全假、Complementary 全真,")
    print("     它们单独的 AUC 无定义,只在合并口径里体现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""选三层抽头 —— 直接在样本级打分上搜「均匀融合」的三层组合。

    SQUADE_TAXONOMY=ntireval python tools/pick_tap_layers.py \
        --cache /workspace/cache/vg_val --scores /workspace/probe/vg_val/probe_scores_l20.001.npy

为什么不用 probe_layers.select_layers:那个的目标函数是「各桶取三层中**最优**者的
AUC 均值」—— 一个 oracle 量,它假设推理时知道该听哪层。而实测下来路由不成立
(折外交叉拟合 net +0.117, p=0.32),真正在用的是**均匀平均三个专家**。目标函数
和实际用法不一致,选出来的层自然不是最优的。这里直接优化实际用法。

融合方式用**秩平均**:各层探针打分的尺度差很多(浅层范数比深层小两个量级),
直接相加等于给深层加权。AUC 本身就是秩统计量,秩平均既尺度无关又不引入
需要标定的参数(而 z-score 要用验证集统计量,那是一次轻微的泄漏)。

clean 组与 deg 组分开搜,因为二元路由是目前唯一被验证的自适应机制:
干净图走浅组(可提前退出、省算力),退化图走深组。
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_features import load_cache      # noqa: E402
from probe_layers import auc               # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scores", required=True, help="bootstrap_oracle 落盘的 (L,N) 打分")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--min-gap", type=int, default=2,
                    help="同组内相邻两层至少隔多少 —— 挨着的层高度相关,融合等于白加一个")
    a = ap.parse_args(argv)

    _, _, meta, cfg = load_cache(a.cache)
    va = [i for i, m in enumerate(meta) if m["split"] == "val"]
    S = np.load(a.scores)
    if S.shape[1] != len(va):
        raise SystemExit(f"打分 {S.shape} 与缓存的验证集 {len(va)} 行对不上")
    y = np.array([float(meta[i]["label"]) for i in va])
    sev = np.array([sum(meta[i]["code"]) for i in va])
    L = S.shape[0]

    # 秩归一化:尺度无关,且不需要任何用验证集标定的参数
    R = np.argsort(np.argsort(S, axis=1), axis=1).astype(np.float64) / (S.shape[1] - 1)

    masks = {"clean (走浅组)": sev == 0, "degraded (走深组)": sev > 0}
    combos = [c for c in itertools.combinations(range(L), a.k)
              if all(c[i + 1] - c[i] >= a.min_gap for i in range(a.k - 1))]
    print(f"缓存 {a.cache}   验证 {len(va)} 行   层 {L}   候选组合 {len(combos)}"
          f" (k={a.k}, 组内最小间隔 {a.min_gap})\n")

    best = {}
    for tag, m in masks.items():
        if m.sum() < 50 or len(set(y[m])) < 2:
            continue
        single = np.array([auc(R[li][m], y[m]) for li in range(L)])
        scores = np.array([auc(R[list(c)][:, m].mean(0), y[m]) for c in combos])
        order = np.argsort(-scores)
        b = combos[order[0]]
        best[tag] = b
        print(f"── {tag}   n={int(m.sum())}  (假 {int(y[m].sum())} / 真 {int((1-y[m]).sum())})")
        print(f"   最佳单层        L{int(single.argmax()):<2}          {single.max():.4f}")
        print(f"   前 {a.top} 个三层组合(均匀秩平均):")
        for r in order[:a.top]:
            c = combos[r]
            print(f"     {str(list(c)):<14} {scores[r]:.4f}   "
                  f"(比最佳单层 {(scores[r] - single.max()) * 100:+.2f} 点)")
        # 提前退出的代价曲线:浅组只有最深那一层决定算到哪
        if "clean" in tag:
            print(f"   限制最深层 <= K 时的最优组合(K 决定提前退出点,省下的算力 = 1 - K/{L - 1}):")
            for K in range(8, L, 4):
                ok = [r for r in order if combos[r][-1] <= K]
                if not ok:
                    continue
                c, s = combos[ok[0]], scores[ok[0]]
                print(f"     K={K:<2}  {str(list(c)):<14} {s:.4f}   "
                      f"省算力 {(1 - K / (L - 1)) * 100:4.1f}%   "
                      f"(比无限制 {(s - scores[order[0]]) * 100:+.2f} 点)")
        print()

    if len(best) == 2:
        ks = list(best)
        for tag in ks:
            other = [t for t in ks if t != tag][0]
            m = masks[tag]
            own = auc(R[list(best[tag])][:, m].mean(0), y[m])
            xfer = auc(R[list(best[other])][:, m].mean(0), y[m])
            print(f"交叉检验:{tag} 用自己的 {list(best[tag])} = {own:.4f}, "
                  f"用对方的 {list(best[other])} = {xfer:.4f}   差 {(own - xfer) * 100:+.2f} 点")
        print("  两组差得越少,说明分成两组的收益越小 —— 差 < 0.1 点就不值得多养一组专家。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

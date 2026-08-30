"""E0 probe heatmap — layer x degradation bucket, with AUC printed in every cell.

    python tools/plot_probe_heatmap.py --probe /workspace/probe/vg_transfer \
        --title "DINOv2 ViT-g" --out heatmap.png

One panel, two encodings — deliberately:

    text   = absolute AUC x 100        answers "how hard is this bucket"
    color  = points below that column's best layer, clipped   answers "where is its optimum"

Two panels would print the same 500-1000 numbers twice. Colouring by the absolute
value instead is useless whenever AUC saturates: on in-distribution data every layer
past ~15 sits at 99.x and renders as one flat block, hiding the per-bucket optimum
that the figure exists to show. Per-column normalisation is what makes depth
preference visible; the printed number keeps the absolute scale readable.

Clipping is required for the same reason: shallow layers sit 15-25 points below
their column max and would otherwise eat the whole ramp.

**The verdict line is the permutation p, not the spread.** Spread / distinct /
tie counts are descriptive only — none is calibrated against a null, and smaller
buckets inflate all three. Measured: NTIRE val showed a 16-layer spread that looked
decisive, while pure noise scored +0.569 of the +0.673 point estimate (p = 0.17).

Sequential magnitude => single hue, light -> dark (dataviz rule). Pale = at the
bucket's optimum. The right-hand strip is each layer's mean AUC over the degradation
buckets — the best single fixed layer, i.e. the no-routing baseline.

Layout is column-count aware: this figure is used with anywhere from 8 to 24 buckets.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# dataviz blue sequential ramp (100 -> 700), taken verbatim
BLUE = ["#eef5fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8981"
SURFACE, GRID, ACCENT = "#fcfcfb", "#e8e7e3", "#c0392b"
CLIP = 3.0            # points of AUC; beyond this a cell is simply "far from best"
DARK_AT = 1.45        # deficit past which the fill needs white text
CELL_W, CELL_H = 0.62, 0.195      # inches
PAD_L, PAD_R, PAD_B = 1.05, 0.35, 0.40


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="probe_layers.py --out directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)

    d = Path(a.probe)
    cols = open(d / "heatmap.csv").readline().strip().split("_x_")[1].split("|")
    H = np.loadtxt(d / "heatmap.csv", delimiter=",", skiprows=1) * 100        # (L, B)
    sel = json.loads((d / "selected.json").read_text())
    fal, nb = sel["falsification"], sel["buckets"]
    # 置换结果有两个来源:新版 probe_layers 直接写进 selected.json;
    # 旧版没有,但 tools/bootstrap_oracle.py 会单独写一份 —— 两处都要认,
    # 否则跑过检验的旧结果会被这张图标成"没跑",比不标更糟。
    perm = sel.get("permutation")
    bs = d / "oracle_gain_bootstrap.json"
    if perm is None and bs.exists():
        b = json.loads(bs.read_text())
        perm = {"point": b["point"], "null_mean": b["null_mean"],
                "net_gain": b["net_gain"], "p": b["perm_p"], "n_perm": b["n_boot"],
                "ci95": b.get("boot_ci95")}
    L, B = H.shape
    argmax = H.argmax(0)
    deficit = H.max(0, keepdims=True) - H

    # family buckets are unions of the dimension buckets, not fresh samples — they
    # only test the README dichotomy, so they sit apart, after a divider.
    dim = [i for i, c in enumerate(cols) if not c.startswith("*")]
    order = dim + [i for i, c in enumerate(cols) if c.startswith("*")]
    gap = len(dim)

    dim_only = [i for i in dim if cols[i] != "clean"]      # the no-routing baseline
    per_layer = H[:, dim_only].mean(1)
    best_fixed = int(per_layer.argmax())

    # ---------------------------------------------------------------- layout
    title = f"Layer-wise linear probe{'  ·  ' + a.title if a.title else ''}"
    grid_w, grid_h = CELL_W * B, CELL_H * L
    strip_w = CELL_W * 1.7
    fig_w = PAD_L + grid_w + 0.14 + strip_w + PAD_R
    title_lines = textwrap.wrap(title, width=max(40, int(fig_w * 8.2)))
    head_h = 0.30 * len(title_lines) + 1.34          # title block + 4 note lines
    foot_txt = "   ".join(f"{cols[i].replace('>0', '').replace('*', '')}={argmax[i]}"
                          for i in order)
    foot_lines = textwrap.wrap(foot_txt, width=max(60, int(fig_w * 11.5)))
    foot_h = 0.24 + 0.17 * len(foot_lines) + 0.46
    xlab_h = 0.95                                     # rotated bucket labels
    fig_h = head_h + grid_h + xlab_h + foot_h + PAD_B

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    left, right = PAD_L / fig_w, 1 - PAD_R / fig_w
    top, bottom = 1 - head_h / fig_h, (foot_h + xlab_h + PAD_B) / fig_h
    gs = fig.add_gridspec(1, 2, width_ratios=[grid_w, strip_w],
                          wspace=0.14 / ((grid_w + strip_w) / 2),
                          left=left, right=right, top=top, bottom=bottom)

    x0, y = 0.012, 1 - 0.30 / fig_h
    for ln in title_lines:
        fig.text(x0, y, ln, fontsize=15, color=INK, weight="bold")
        y -= 0.30 / fig_h
    y -= 0.06 / fig_h

    def note(txt, size=8.8, color=INK2, weight="normal"):
        nonlocal y
        fig.text(x0, y, txt, fontsize=size, color=color, weight=weight)
        y -= 0.235 / fig_h

    note(f"One linear probe per layer, fit on train, AUC scored per bucket on val.  "
         f"{L} tap points x {B} buckets.")
    note(f"Number = AUC x 100.   Colour = points below that column's best layer "
         f"(pale = at the optimum, clipped at {CLIP:.0f}).   Red box = column argmax.")
    if perm:
        note(f"Permutation test ({perm['n_perm']} draws): per-bucket best beats the single "
             f"best layer by {perm['point'] * 100:+.3f} pts;  pure noise alone scores "
             f"{perm['null_mean'] * 100:+.3f}  ->  net {perm['net_gain'] * 100:+.3f} pts, "
             f"p = {perm['p']:.3f}   ->   {'PASS' if perm['p'] < 0.05 else 'FAIL'}"
             + (f"   ·   bootstrap 95% CI [{perm['ci95'][0] * 100:+.3f}, "
                f"{perm['ci95'][1] * 100:+.3f}]" if perm.get("ci95") else ""),
             color=INK if perm["p"] < 0.05 else ACCENT, weight="bold")
    else:
        note("Permutation test NOT run — spread alone is not evidence; small buckets inflate it.",
             color=ACCENT, weight="bold")
    note(f"Descriptive only (uncalibrated): spread {fal['spread']}  ·  "
         f"{fal['n_distinct']} distinct  ·  max tie {max(fal['n_tied_at_max'].values())}   |   "
         f"selected taps {sel['layers_list']} ({sel['objective']:.4f}), "
         f"unconstrained {sel['unconstrained']['layers']} ({sel['unconstrained']['objective']:.4f})",
         color=INK3)

    # ------------------------------------------------------------- main grid
    ax = fig.add_subplot(gs[0], facecolor=SURFACE)
    M = deficit[:, order]
    ax.imshow(M, aspect="auto", cmap=LinearSegmentedColormap.from_list("b", BLUE),
              interpolation="nearest", vmin=0, vmax=CLIP)
    for j, ci in enumerate(order):
        for li in range(L):
            ax.text(j, li, f"{H[li, ci]:.1f}", ha="center", va="center", fontsize=5.9,
                    color="white" if M[li, j] >= DARK_AT else INK)
        ax.add_patch(Rectangle((j - .5, argmax[ci] - .5), 1, 1, fill=False,
                               edgecolor=ACCENT, linewidth=1.7, zorder=5))
    ax.axvline(gap - 0.5, color=INK3, linewidth=1.1, linestyle=(0, (4, 3)), zorder=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{cols[i].replace('>0', '')}  n={nb[cols[i]]}" for i in order],
                       fontsize=7.2, color=INK, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(L))
    ax.set_yticklabels(range(L), fontsize=6.4, color=INK2)
    ax.set_ylabel("transformer block (tap point)", fontsize=9.5, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    for lay in sel["layers_list"]:
        ax.get_yticklabels()[lay].set_color(ACCENT)
        ax.get_yticklabels()[lay].set_weight("bold")

    # ------------------------------------------- right strip: fixed-layer mean
    ax2 = fig.add_subplot(gs[1], facecolor=SURFACE)
    col = (per_layer.max() - per_layer)[:, None]
    ax2.imshow(col, aspect="auto", cmap=LinearSegmentedColormap.from_list("b", BLUE),
               interpolation="nearest", vmin=0, vmax=CLIP)
    for li in range(L):
        ax2.text(0, li, f"{per_layer[li]:.1f}", ha="center", va="center", fontsize=5.9,
                 color="white" if col[li, 0] >= DARK_AT else INK)
    ax2.add_patch(Rectangle((-.5, best_fixed - .5), 1, 1, fill=False,
                            edgecolor=ACCENT, linewidth=1.7, zorder=5))
    ax2.set_xticks([0])
    ax2.set_xticklabels([f"mean of the {len(dim_only)} degradations"], fontsize=7.2,
                        color=INK, rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_yticks([])
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_color(GRID)

    # ---------------------------------------------------------------- footer
    y = (foot_h - 0.10) / fig_h
    fig.text(x0, y, "Best layer per bucket:", fontsize=8.8, color=INK, weight="bold")
    for ln in foot_lines:
        y -= 0.17 / fig_h
        fig.text(x0, y, ln, fontsize=8.0, color=INK2)
    y -= 0.26 / fig_h
    fig.text(x0, y,
             f"Best single fixed layer = {best_fixed} (mean {per_layer[best_fixed]:.2f}) — the "
             "no-routing baseline any depth router must beat.   Buckets right of the dashed "
             "line are unions of the dimension\nbuckets (the smudged / shattered / photometric "
             "split), not independent samples.",
             fontsize=7.6, color=INK3, linespacing=1.5, va="top")

    out = Path(a.out) if a.out else d / "heatmap.png"
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print(f"-> {out}   ({fig_w:.1f}x{fig_h:.1f} in)")
    print(f"   {L} layers x {B} buckets   argmax {argmax.min()}~{argmax.max()}   "
          f"best fixed layer {best_fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""NTIRE val / val_hard 的官方退化真值 -> SQuaDE manifest。

    export SQUADE_TAXONOMY=ntireval
    python utils/manifest_ntire_val.py --labels data/ntire_val/_dl/val_labels.csv \
           --images data/ntire_val/val_images --out-csv data/manifest_val.csv

与 `utils/preprocess.py` / `preprocess_ntire.py` 的根本区别:**这里一张图都不生成**。
退化是 NTIRE 官方施加好的,`val_labels.csv` 的 `distortions` / `distortion_scales`
两列就是码字真值。本脚本只做「官方标注 -> 码字」的翻译和记账。

因此 preprocess 的纪律 1/2/3/4/6(存 PNG、确定性、跳过产物、归一化顺序)在这里
**全部不适用** —— 没有像素被改动。仍然成立的只有两条:

  * 纪律 5(不 resize):图**原样**喂给 cache_features,由 DINOv3Preprocessor 做原生
    512 center crop。绝不能为了对齐尺寸去缩放 —— downscale / pixelate / randomcrop
    这几维的证据正是重采样痕迹,再缩一次就洗掉了。
  * 纪律 7(划分不能泄漏):这里比 preprocess 简单 —— NTIRE 的干净图与退化图是
    **不同的图**,不存在「同一源图的干净版/退化版」配对,所以按 image_name 哈希划分
    就够了,不需要源图分组。这一点脚本会显式核验(见输出的「配对检查」)。

---------------------------------------------------------------------------
scale -> 档位

19 个畸变里 18 个的 scale 只有 5 个离散取值,直接按严重度排序映射到 1..5。
方向、以及 lincontrchange 的非单调顺序,定义在 `utils/deg_taxonomy.py`。

`downscale` 是唯一连续的(val 里 3708 个互不相同的值,均匀落在 0.3~0.8),按**等宽**
分箱到 5 档。等宽而非等频,是为了让「档 3」在 downscale 与其余维上表示的严重度大致
可比 —— weights_mlp 的序数输入 level/(K-1) 才有跨维一致的含义。
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache_features import DEG_DIMS, N_LEVELS, TAXONOMY_NAME  # noqa: E402
from utils.deg_taxonomy import (  # noqa: E402
    NTIREHARD_DIRECTION,
    NTIREVAL_CONTINUOUS,
    NTIREVAL_EXPLICIT_ORDER,
)

# ntirehard 是 ntireval 的超集,方向表也是 —— 用超集这一份即可覆盖两种码本
NTIREVAL_DIRECTION = NTIREHARD_DIRECTION

N_OFFICIAL_LEVELS = 5


def build_level_maps(rows: list[dict]) -> dict[str, list]:
    """扫一遍标注,给每个畸变定出「档位 1..5 -> scale 取值」的表。

    离散维:取观测到的唯一值,按 direction 排序。若唯一值不是 5 个,报出来 ——
    多半是标注里混进了别的东西,不该静默按个数硬分。
    连续维:等宽分箱,表里存箱边界。
    """
    seen: dict[str, list] = defaultdict(list)
    for r in rows:
        for d, s in zip(ast.literal_eval(r["distortions"]), ast.literal_eval(r["distortion_scales"])):
            seen[d].append(s)

    unknown = [d for d in seen if d not in DEG_DIMS]
    if unknown:
        raise SystemExit(
            f"标注里有码本 {TAXONOMY_NAME!r} 不认识的畸变: {sorted(unknown)}\n"
            f"码本维度是 {DEG_DIMS}。val_hard 比 val 多 jpeg_ai / adv_embed_* 等几种,"
            f"要跑 val_hard 得先在 utils/deg_taxonomy.py 里把它们加进 NTIREVAL_DIMS。")

    maps: dict[str, list] = {}
    for d in DEG_DIMS:
        if d not in seen:
            maps[d] = []
            continue
        if d in NTIREVAL_CONTINUOUS:
            lo, hi = NTIREVAL_CONTINUOUS[d]
            maps[d] = ("continuous", lo, hi)
            continue
        uniq = sorted(set(seen[d]))
        direction = NTIREVAL_DIRECTION[d]
        if direction == "explicit":
            order = NTIREVAL_EXPLICIT_ORDER[d]
            missing = set(uniq) - set(order)
            if missing:
                raise SystemExit(f"{d} 的显式顺序表缺了取值 {sorted(missing)}")
            maps[d] = list(order)
        else:
            maps[d] = uniq if direction == "asc" else list(reversed(uniq))
        if len(maps[d]) != N_OFFICIAL_LEVELS:
            print(f"[警告] {d} 观测到 {len(maps[d])} 个强度取值,不是 {N_OFFICIAL_LEVELS} 个:"
                  f" {maps[d]} —— 档位映射仍按顺序做,但跨维的『档 k』不再等严重度。")
    return maps


def to_level(dim: str, scale, level_map) -> int:
    """scale -> 1..5。"""
    m = level_map[dim]
    if isinstance(m, tuple) and m[0] == "continuous":
        _, lo, hi = m
        # 等宽 5 箱;desc 方向(倍率越小越坏)所以要翻过来
        frac = (float(scale) - lo) / (hi - lo)
        frac = min(max(frac, 0.0), 0.999999)
        idx = int(frac * N_OFFICIAL_LEVELS)                    # 0..4,值越大越轻
        if NTIREVAL_DIRECTION[dim] == "desc":
            idx = N_OFFICIAL_LEVELS - 1 - idx
        return idx + 1
    return m.index(scale) + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="val_labels.csv / val_hard_labels.csv")
    ap.add_argument("--images", required=True, help="解压出来的图片目录")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--split-val", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="写进 manifest 的 subset 标记,如 val / val_hard")
    args = ap.parse_args(argv)

    if TAXONOMY_NAME not in ("ntireval", "ntirehard"):
        raise SystemExit(f"当前 SQUADE_TAXONOMY={TAXONOMY_NAME!r},本脚本要 ntireval "
                         f"(val) 或 ntirehard (val_hard,29 维超集)。")

    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8-sig")))
    img_dir = Path(args.images).resolve()
    print(f"标注      : {args.labels}   {len(rows)} 行")
    print(f"图片目录  : {img_dir}")
    print(f"码本      : {TAXONOMY_NAME}  {len(DEG_DIMS)} 维 x {N_LEVELS} 档")

    level_map = build_level_maps(rows)
    print("\n档位映射(档 1 = 最轻, 档 5 = 最重):")
    for d in DEG_DIMS:
        m = level_map[d]
        if isinstance(m, tuple):
            lo, hi = m[1], m[2]
            w = (hi - lo) / N_OFFICIAL_LEVELS
            bins = [f"({lo + i * w:.2f},{lo + (i + 1) * w:.2f}]" for i in range(N_OFFICIAL_LEVELS)]
            if NTIREVAL_DIRECTION[d] == "desc":
                bins = list(reversed(bins))
            print(f"  {d:<18} 连续 -> 等宽 5 箱  {'  '.join(bins)}")
        elif m:
            print(f"  {d:<18} {' -> '.join(str(round(v, 4) if isinstance(v, float) else v) for v in m)}")
        else:
            print(f"  {d:<18} (本 split 里没出现)")

    out_rows, stats, lvl_stats, missing = [], Counter(), Counter(), 0
    n_dist_stats = Counter()
    for r in rows:
        name = r["image_name"]
        p = img_dir / name
        if not p.is_file():
            missing += 1
            continue
        ds = ast.literal_eval(r["distortions"])
        sc = ast.literal_eval(r["distortion_scales"])
        code = [0] * len(DEG_DIMS)
        for d, s in zip(ds, sc):
            lv = to_level(d, s, level_map)
            code[DEG_DIMS.index(d)] = lv
            stats[d] += 1
            lvl_stats[(d, lv)] += 1
        n_dist_stats[len(ds)] += 1
        stem = Path(name).stem
        h = int(hashlib.sha256(f"split|{args.seed}|{stem}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        d_ = {"path": str(p), "image_id": stem, "group": stem, "label": int(r["label"]),
              "subset": args.tag or Path(args.labels).stem}
        d_.update({f"d{i}": code[i] for i in range(len(DEG_DIMS))})
        d_["split"] = "val" if h < args.split_val else "train"
        d_["is_distorted"] = int(r.get("is_distorted", int(bool(ds))))
        d_["deg_names"] = "|".join(ds)
        d_["deg_scales"] = "|".join(f"{s:g}" for s in sc)
        out_rows.append(d_)

    if missing:
        print(f"\n[警告] {missing} 张标注里的图在 {img_dir} 找不到")
    if not out_rows:
        raise SystemExit("一行都没产出 —— 检查 --images 路径")

    print(f"\n每图退化个数: {dict(sorted(n_dist_stats.items()))}")
    print(f"\n{'畸变':<20}{'图数':>7}   各档图数 1..5")
    for d in DEG_DIMS:
        if not stats[d]:
            continue
        print(f"  {d:<18}{stats[d]:>7}   " + " ".join(f"{lvl_stats[(d, l)]:>4}" for l in range(1, N_LEVELS)))

    # 配对检查:NTIRE 的干净图与退化图应该是不同的图,不存在同源配对
    per_group = Counter(x["group"] for x in out_rows)
    dup = sum(1 for v in per_group.values() if v > 1)
    lbl = Counter(x["label"] for x in out_rows)
    dis = Counter(x["is_distorted"] for x in out_rows)
    sp = Counter(x["split"] for x in out_rows)
    n_clean = sum(1 for x in out_rows if sum(x[f"d{i}"] for i in range(len(DEG_DIMS))) == 0)
    print(f"\n配对检查  : 重复 image_id 的组 {dup} 个 (应为 0 —— NTIRE 的干净图与退化图是不同的图,"
          f"没有同源配对,所以按图名划分不会泄漏)")
    print(f"标签分布  : 真={lbl[0]} 假={lbl[1]}")
    print(f"退化标记  : 官方 is_distorted 0={dis[0]} 1={dis[1]};  由码字推出的全零行 {n_clean}"
          f"  {'✅ 一致' if dis[0] == n_clean else '❌ 不一致 —— 码字翻译有问题'}")
    print(f"划分      : train={sp['train']} val={sp['val']}")

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cols = (["path", "image_id", "group", "label", "subset"]
            + [f"d{i}" for i in range(len(DEG_DIMS))]
            + ["split", "is_distorted", "deg_names", "deg_scales"])
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n{len(out_rows)} 行 -> {outp}")
    print(f"列序 d0..d{len(DEG_DIMS) - 1} = {DEG_DIMS}")
    print(f"下一步: SQUADE_TAXONOMY=ntireval python cache_features.py "
          f"--manifest {outp} --out cache/probe_val --layers all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

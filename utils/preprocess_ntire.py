"""NTIRE 2026 训练集 -> 退化数据 + manifest(码本 = NTIRE 官方退化分组)。

    export SQUADE_TAXONOMY=ntire
    python utils/preprocess_ntire.py --shards data/ntire/shard_5 \
           --out data/ntire_deg --out-csv data/manifest_ntire.csv --limit 500 --dry-run
    python utils/preprocess_ntire.py --shards data/ntire/shard_5 \
           --out data/ntire_deg --out-csv data/manifest_ntire.csv --limit 500

`utils/preprocess.py` 的 NTIRE 版。差别只有两处,其余纪律逐条照搬:

  * **退化算子来自官方管线**,不是我们自造的谱。`utils/ntire_aug/` 是竞赛页
    "Transformations Script" 的逐字节 vendored 副本;这里只负责调度和记账。
  * **码字维度 = 官方的 7 个退化组**(+ 可选的第 8 维 geometric),而不是 6 个自造维。
    推导与 geometric 那一维的诚实边界见 `utils/deg_taxonomy.py`。

---------------------------------------------------------------------------
关于源数据的两个事实,直接决定了这个脚本存在的必要性

1. **NTIRE 训练集的图是"干净"的,退化要自己施加。** 竞赛 Data 页的表格里,Train 行的
   Transformations 一栏写的是 "(Provided as distortion pipeline)" —— 官方给的是脚本,
   不是已经退化好的图。实测佐证:shard_5 里随机抽 400 张,**JPEG 量化表完全相同**
   (标准 IJG 表,q≈90+),没有任何逐图变化的压缩强度。若管线已被施加过,jpeg 档
   q=43..4 会让量化表五花八门。

2. **官方训练变换里一个几何退化都没有。** 12 个变换全是光度/信号域的。
   这意味着 README 的 shattered 一侧在官方训练码本下**测不到**,
   故有 `geometric` 第 8 维,见 deg_taxonomy.py。

---------------------------------------------------------------------------
纪律(与 utils/preprocess.py 同源,编号对齐)

1. **一律存 PNG**  jpeg 档在内存里编解码一遍再存 PNG,保留块效应像素但不引入
   第二次、**没有记录在码字里**的压缩。

2. **确定性**  每张图的退化分配、以及官方算子内部用到的三个全局随机源
   (`random` / `np.random` / `torch`)都由 (--seed, 图片名) 派生的种子重置。
   官方算子里有真随机(impulsenoise 的坐标、colorshift 的方向、jitter 的位移场),
   不重置的话多进程下重跑结果不同,缓存与 manifest 对不上。

3. **跳过已产出的图**  输出写在独立的 clean/ 与 deg/ 目录,重跑时按文件存在跳过,
   不可能在干扰图上再叠一层。

4. **归一化在退化之前**  ① 中心裁正方 + LANCZOS -> 512   ② 施加退化   ③ 尺寸变了再归一化。
   反过来做会把 JPEG 的 8x8 块网格重采样掉、把噪声平均掉、把模糊的有效 σ 改掉。
   第 ③ 步只有 geometric 会触发。

5. **只收原生边长 >= --min-side 的源图**  NTIRE 的图从 512x288 到 3024x4032 都有。
   短边小于 512 的图在第 ① 步会被**放大**,那是一次没记录在码字里的插值退化,
   而且它与「真/假」可能相关(不同生成器的出图尺寸不同)—— 典型的伪线索。
   实测 shard_5 有 ~87% 的图短边 >= 512,丢掉那 13% 比留着这个混杂划算。

6. **train/val 按源图分组划分**  同一张源图的干净版与退化版必须落同一侧。

7. **默认按真假均衡采样**  AUC 本身对类别不平衡不敏感,但**每个退化桶内**的正负样本
   数决定了该桶 AUC 的标准误。NTIRE 训练集是 1:1.77,均衡后每桶的有效样本数更大。
   `--no-balance` 关掉。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from utils.ntire_compat import apply_numpy2_patch  # noqa: E402

apply_numpy2_patch()

from cache_features import DEG_CODE_VALUES, DEG_DIMS, N_LEVELS, TAXONOMY_NAME  # noqa: E402
from utils.deg_taxonomy import NTIRE_GEOMETRIC_DIM, NTIRE_GROUP_ALIAS  # noqa: E402
from utils.ntire_aug.utils_data import (  # noqa: E402
    distortion_functions,
    distortion_groups,
    distortion_range,
)

N_OFFICIAL_LEVELS = 5          # 官方每个畸变都是 5 档;码字里是 1..5,0 留给"未施加"


# --------------------------------------------------------------------------- geometric(第 8 维)

# 算子名取自 NTIRE val/test 的变换表(Random Crop / Random Aspect Crop / Downscale),
# **强度参数是我们定的** —— 官方只公开了训练管线的脚本。报告里必须这样写。
# 五档按严重度递增,与官方各维的排法一致。
GEOMETRIC_RANGE = {
    "randomcrop": [0.90, 0.80, 0.70, 0.60, 0.50],    # 保留的边长比例
    "aspectcrop": [0.90, 0.80, 0.70, 0.60, 0.50],    # 保留的面积比例,长宽比另抽
    "downscale": [0.75, 0.50, 0.35, 0.25, 0.15],     # 下采样倍率,之后放回原尺寸
}


def _geo_randomcrop(x: torch.Tensor, keep: float) -> torch.Tensor:
    _, h, w = x.shape
    nh, nw = max(16, int(h * keep)), max(16, int(w * keep))
    top = int(np.random.randint(0, h - nh + 1))
    left = int(np.random.randint(0, w - nw + 1))
    return x[:, top : top + nh, left : left + nw]


def _geo_aspectcrop(x: torch.Tensor, area: float) -> torch.Tensor:
    """Random Aspect Crop:面积固定,长宽比在 [3/4, 4/3] 里随机 —— 会改变长宽比。

    注意归一化第 ③ 步会把它裁回正方再缩放,所以"长宽比异常"这条线索留不下来,
    留下的是视野缺失 + 上采样插值痕迹。这与 utils/preprocess.py 对 crop 的处理一致
    (那里记为纪律 4 的已知代价)。
    """
    _, h, w = x.shape
    ar = float(np.random.uniform(3 / 4, 4 / 3))
    nh = max(16, min(h, int(round((area * h * w / ar) ** 0.5))))
    nw = max(16, min(w, int(round(nh * ar))))
    top = int(np.random.randint(0, h - nh + 1))
    left = int(np.random.randint(0, w - nw + 1))
    return x[:, top : top + nh, left : left + nw]


def _geo_downscale(x: torch.Tensor, scale: float) -> torch.Tensor:
    import torch.nn.functional as F

    _, h, w = x.shape
    small = F.interpolate(x.unsqueeze(0), size=(max(8, int(h * scale)), max(8, int(w * scale))),
                          mode="bicubic", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)


GEOMETRIC_FUNCTIONS = {
    "randomcrop": _geo_randomcrop,
    "aspectcrop": _geo_aspectcrop,
    "downscale": _geo_downscale,
}


# --------------------------------------------------------------------------- 码本 <-> 官方管线

def build_dim_variants() -> dict[str, list[str]]:
    """码字的每一维 -> 该维下可选的畸变变体名。"""
    out: dict[str, list[str]] = {}
    for d in DEG_DIMS:
        if d in NTIRE_GROUP_ALIAS:
            out[d] = list(distortion_groups[NTIRE_GROUP_ALIAS[d]])
        elif d == NTIRE_GEOMETRIC_DIM:
            out[d] = list(GEOMETRIC_FUNCTIONS)
        else:
            raise SystemExit(
                f"码本维度 {d!r} 在 NTIRE 管线里没有对应的退化组。"
                f"当前 SQUADE_TAXONOMY={TAXONOMY_NAME!r};本脚本只支持 ntire / ntire7。")
    return out


def variant_value(variant: str, level: int) -> float:
    """码字档位 (1..5) -> 该变体的实际参数。"""
    table = distortion_range.get(variant) or GEOMETRIC_RANGE[variant]
    return table[level - 1]


def apply_variant(x: torch.Tensor, variant: str, level: int) -> torch.Tensor:
    fn = distortion_functions.get(variant) or GEOMETRIC_FUNCTIONS[variant]
    y = fn(x.clone(), variant_value(variant, level))
    return torch.clip(y.to(torch.float32), 0, 1)


def level_probs(mode: str) -> np.ndarray:
    """档位抽样分布。

    `ntire` 复刻官方 get_distortions_composition 的高斯权重(MEAN=0, STD=2.5),
    重心压在最轻的一档;`uniform` 各档等概率。

    探针默认用 uniform:热力图按 `--bucket-mode dim` 把 5 档并成一桶时两者差别不大,
    但一旦要按 level 细分桶,官方分布下最重的那档样本数只有最轻档的 60%,
    正是最需要样本的地方最缺样本。改了分布不改算子,仍然是官方的退化。
    """
    if mode == "uniform":
        return np.full(N_OFFICIAL_LEVELS, 1.0 / N_OFFICIAL_LEVELS)
    std = 2.5
    p = np.array([np.exp(-(i ** 2) / (2 * std ** 2)) for i in range(N_OFFICIAL_LEVELS)])
    return p / p.sum()


# --------------------------------------------------------------------------- 图像工具

def normalize(img: Image.Image, size: int) -> Image.Image:
    """中心裁正方 + LANCZOS -> size x size。已合规则原样返回,不做无谓重采样。

    先裁正方再缩放,而不是直接 resize —— 后者会拉伸长宽比,把"非正方"这个
    与退化无关的属性变成可被利用的伪线索。
    """
    if img.size == (size, size):
        return img
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    return img.resize((size, size), Image.LANCZOS)


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1).float() / 255.0


def to_pil(x: torch.Tensor) -> Image.Image:
    a = (x.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(a)


def seed_everything(seed: int, key: str) -> np.random.Generator:
    """纪律 2:重置全部三个全局随机源 + 返回一个本地 Generator 供调度用。

    官方算子内部直接用 `np.random.*` / `torch.randn` / `random.*`,拦不住,
    只能在每张图前把全局种子按 (seed, 图片名) 定死。
    """
    h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    s = int.from_bytes(h[:8], "big")
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s % (2 ** 63))
    return np.random.default_rng(s)


# --------------------------------------------------------------------------- 单图任务

def process_one(task: dict) -> dict | None:
    """一张源图 -> 干净图 + 退化图 + 两行 manifest 记录。多进程 worker。"""
    torch.set_num_threads(1)
    src = Path(task["src"])
    size, seed = task["size"], task["seed"]
    clean_path = Path(task["clean_path"])
    deg_path = Path(task["deg_path"])

    rng = seed_everything(seed, task["image_id"])

    dim_variants = task["dim_variants"]
    probs = np.asarray(task["level_probs"])

    # ---- 抽退化:无放回抽 n 个**维**(= 官方的组),每维再抽变体与档位 ----
    n_dist = 1 if task["max_distortions"] == 1 else int(rng.integers(1, task["max_distortions"] + 1))
    dims = list(rng.choice(DEG_DIMS, size=n_dist, replace=False))
    code = [0] * len(DEG_DIMS)
    picked = []
    for d in dims:
        variant = dim_variants[d][int(rng.integers(len(dim_variants[d])))]
        level = int(rng.choice(np.arange(1, N_OFFICIAL_LEVELS + 1), p=probs))
        code[DEG_DIMS.index(d)] = level
        picked.append((d, variant, level))

    rec = {
        "image_id": task["image_id"],
        "group": task["image_id"],
        "label": task["label"],
        "shard": task["shard"],
        "code": code,
        "deg_groups": "|".join(d for d, _, _ in picked),
        "deg_variants": "|".join(v for _, v, _ in picked),
        "deg_levels": "|".join(str(l) for _, _, l in picked),
        "deg_values": "|".join(f"{variant_value(v, l):g}" for _, v, l in picked),
        "clean_path": str(clean_path),
        "deg_path": str(deg_path),
    }
    if task["dry_run"]:
        return rec
    if clean_path.exists() and deg_path.exists() and not task["overwrite"]:
        rec["skipped"] = True
        return rec

    try:
        with Image.open(src) as im:
            base = normalize(im.convert("RGB"), size)          # ① 归一化
        x = to_tensor(base)
        for _, variant, level in picked:                       # ② 施加退化
            x = apply_variant(x, variant, level)
        out = to_pil(x)
        if out.size != (size, size):                           # ③ 只有 geometric 会触发
            out = normalize(out, size)
        base.save(clean_path, "PNG")                           # 纪律 1
        out.save(deg_path, "PNG")
    except Exception as e:                                     # 损坏的源文件
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# --------------------------------------------------------------------------- 源数据

def read_shard(shard_dir: Path) -> list[dict]:
    """读一个 NTIRE shard 的 labels.csv -> [{path, image_name, label, shard}]"""
    csv_path = shard_dir / "labels.csv"
    img_dir = shard_dir / "images"
    if not csv_path.is_file():
        raise SystemExit(f"{csv_path} 不存在 —— --shards 要指向解压后的 shard_i 目录")
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "src": str(img_dir / r["image_name"]),
                "image_name": r["image_name"],
                "label": int(r["label"]),
                "shard": shard_dir.name,
            })
    return rows


# --------------------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="解压后的 shard_i 目录,可给多个")
    ap.add_argument("--out", required=True, help="退化图输出目录(会建 clean/ 与 deg/)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 张**源图**(均衡后)")
    ap.add_argument("--min-side", type=int, default=512,
                    help="源图短边下限,低于此值会在归一化时被放大 —— 见纪律 5")
    ap.add_argument("--max-distortions", type=int, default=1,
                    help="每图施加几种退化。1 = 单退化隔离(探针默认,归因最干净);"
                         "3 = 官方 get_distortions_composition 的口径(1~3 组)")
    ap.add_argument("--level-dist", default="uniform", choices=["uniform", "ntire"])
    ap.add_argument("--split-val", type=float, default=0.25, help="按源图分组划验证集")
    ap.add_argument("--no-balance", action="store_true", help="不按真假均衡采样")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if TAXONOMY_NAME not in ("ntire", "ntire7"):
        raise SystemExit(
            f"当前 SQUADE_TAXONOMY={TAXONOMY_NAME!r}。本脚本产出的是 NTIRE 码本的 manifest,"
            f"先 `export SQUADE_TAXONOMY=ntire`(或 ntire7)。")
    for d in DEG_DIMS:
        if DEG_CODE_VALUES[d] != list(range(N_LEVELS)):
            raise SystemExit(f"码本 {d} 维的合法取值 {DEG_CODE_VALUES[d]} 不是齐整五档,与本脚本不符")

    dim_variants = build_dim_variants()
    probs = level_probs(args.level_dist)

    # ---- 收集源图 ----
    src_rows: list[dict] = []
    for sd in args.shards:
        src_rows += read_shard(Path(sd).resolve())
    print(f"shard     : {args.shards}")
    print(f"源图      : {len(src_rows)} 张(labels.csv 全量)")

    # 纪律 5:短边过滤。读尺寸要开一遍文件,只在需要时做。
    from tqdm import tqdm

    keep, too_small, unreadable = [], 0, 0
    for r in tqdm(src_rows, desc="扫尺寸", unit="img"):
        try:
            with Image.open(r["src"]) as im:
                if min(im.size) >= args.min_side:
                    keep.append(r)
                else:
                    too_small += 1
        except Exception:
            unreadable += 1
    print(f"短边 >= {args.min_side}: {len(keep)} 张   丢弃 {too_small} 张(会被放大)"
          f"   读不出 {unreadable} 张")

    # ---- 均衡 + 截断(确定性:按 image_name 哈希排序,不用随机打乱)----
    def h(name: str) -> int:
        return int(hashlib.sha256(f"pick|{args.seed}|{name}".encode()).hexdigest()[:12], 16)

    keep.sort(key=lambda r: h(r["image_name"]))
    if not args.no_balance:
        by = defaultdict(list)
        for r in keep:
            by[r["label"]].append(r)
        n = min(len(by[0]), len(by[1]))
        if args.limit:
            n = min(n, args.limit // 2)
        picked = [x for pair in zip(by[0][:n], by[1][:n]) for x in pair]
    else:
        picked = keep[: args.limit] if args.limit else keep
    if args.limit:
        picked = picked[: args.limit]
    print(f"采样      : {len(picked)} 张源图"
          f"  ({'真假均衡' if not args.no_balance else '不均衡'}"
          f", 真={sum(1 for r in picked if r['label'] == 0)}"
          f" 假={sum(1 for r in picked if r['label'] == 1)})")

    print(f"码本      : {TAXONOMY_NAME}  {len(DEG_DIMS)} 维 x {N_LEVELS} 档")
    for d in DEG_DIMS:
        vs = dim_variants[d]
        src = "官方" if d in NTIRE_GROUP_ALIAS else "★自定义参数(算子名取自 NTIRE val/test 表)"
        print(f"  {d:<11} {'/'.join(vs):<28} {src}")
        for v in vs:
            tbl = distortion_range.get(v) or GEOMETRIC_RANGE[v]
            print(f"      {v:<14} 档1..5 = {tbl}")
    print(f"每图退化数: {args.max_distortions} "
          f"({'单退化隔离' if args.max_distortions == 1 else '官方 1~%d 组复合口径' % args.max_distortions})")
    print(f"档位分布  : {args.level_dist}  {probs.round(3).tolist()}")
    print(f"模式      : {'DRY RUN(不写任何文件)' if args.dry_run else '写入 ' + args.out}\n")

    out = Path(args.out)
    clean_dir, deg_dir = out / "clean", out / "deg"
    if not args.dry_run:
        clean_dir.mkdir(parents=True, exist_ok=True)
        deg_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for r in picked:
        stem = Path(r["image_name"]).stem
        tasks.append({
            "src": r["src"], "image_id": stem, "label": r["label"], "shard": r["shard"],
            "clean_path": str(clean_dir / f"{stem}.png"), "deg_path": str(deg_dir / f"{stem}.png"),
            "size": args.size, "seed": args.seed, "dim_variants": dim_variants,
            "level_probs": probs.tolist(), "max_distortions": args.max_distortions,
            "dry_run": args.dry_run, "overwrite": args.overwrite,
        })

    if args.dry_run or args.workers <= 1:
        recs = [process_one(t) for t in tqdm(tasks, desc="退化", unit="img")]
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            recs = list(tqdm(pool.imap(process_one, tasks, chunksize=8),
                             total=len(tasks), desc="退化", unit="img"))

    # ---- 记账 ----
    errs = [r for r in recs if r.get("error")]
    skipped = sum(1 for r in recs if r.get("skipped"))
    good = [r for r in recs if not r.get("error")]

    stats = Counter()
    var_stats = Counter()
    for r in good:
        for d, v, l in zip(r["deg_groups"].split("|"), r["deg_variants"].split("|"),
                           r["deg_levels"].split("|")):
            stats[f"{d}={l}"] += 1
            var_stats[v] += 1
    print("\n退化分配(维 x 档):")
    for d in DEG_DIMS:
        line = f"  {d:<11}" + "".join(f" 档{l}={stats[f'{d}={l}']:<5}" for l in range(1, N_LEVELS))
        print(line + f"   小计={sum(stats[f'{d}={l}'] for l in range(1, N_LEVELS))}")
    print("变体分配  : " + "  ".join(f"{k}={v}" for k, v in sorted(var_stats.items())))

    rows = []
    for r in good:
        gh = int(hashlib.sha256(f"split|{args.seed}|{r['group']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "val" if gh < args.split_val else "train"
        for path, code, iid, meta in (
            (r["clean_path"], [0] * len(DEG_DIMS), r["image_id"] + "__clean", ("", "", "", "")),
            (r["deg_path"], r["code"], r["image_id"] + "__deg",
             (r["deg_groups"], r["deg_variants"], r["deg_levels"], r["deg_values"])),
        ):
            d = {"path": path, "image_id": iid, "group": r["group"], "label": r["label"],
                 "shard": r["shard"]}
            d.update({f"d{i}": code[i] for i in range(len(DEG_DIMS))})
            d["split"] = split
            d["deg_groups"], d["deg_variants"], d["deg_levels"], d["deg_values"] = meta
            rows.append(d)

    n_codes = len({tuple(x[f"d{i}"] for i in range(len(DEG_DIMS))) for x in rows})
    multi = sum(1 for x in rows if sum(x[f"d{i}"] > 0 for i in range(len(DEG_DIMS))) > 1)
    lbl = Counter(x["label"] for x in rows)
    sp = Counter(x["split"] for x in rows)
    bad_groups = {g for g, s in
                  ((g, {x["split"] for x in rows if x["group"] == g}) for g in {x["group"] for x in rows})
                  if len(s) > 1}
    print(f"\n实际出现的码字种类: {n_codes}  "
          f"(单退化上限 1 + {len(DEG_DIMS)}x{N_LEVELS - 1} = {1 + len(DEG_DIMS) * (N_LEVELS - 1)})")
    print(f"多重退化样本: {multi}  "
          f"({'应为 0 —— 每图只施加一种' if args.max_distortions == 1 else '复合口径下应 > 0'})")
    print(f"标签分布: 真={lbl[0]} 假={lbl[1]}")
    print(f"划分: train={sp['train']} val={sp['val']}   跨 split 的源图组: {len(bad_groups)} (必须为 0)")
    if skipped:
        print(f"已存在而跳过: {skipped} 张(加 --overwrite 强制重做)")
    if errs:
        print(f"失败: {len(errs)} 张。前 3 条:")
        for e in errs[:3]:
            print(f"    {e['image_id']}: {e['error']}")

    if args.dry_run:
        print("\nDRY RUN,未写任何文件。")
        return 0

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cols = (["path", "image_id", "group", "label", "shard"]
            + [f"d{i}" for i in range(len(DEG_DIMS))]
            + ["split", "deg_groups", "deg_variants", "deg_levels", "deg_values"])
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 行 -> {outp}")
    print(f"列序 d0..d{len(DEG_DIMS) - 1} = {DEG_DIMS}")
    print(f"下一步: SQUADE_TAXONOMY={TAXONOMY_NAME} python cache_features.py "
          f"--manifest {outp} --out cache/probe_ntire --layers all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

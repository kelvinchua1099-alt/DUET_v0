"""进阶退化生成 —— 19 种 NTIRE 官方畸变,每图随机 1~3 种,保留干净图。

    export SQUADE_TAXONOMY=ntireval
    python utils/advanced_degrade.py --data /workspace/data/jamlai_raw \
           --out-csv /workspace/data/manifest_adv.csv --dry-run
    python utils/advanced_degrade.py --data /workspace/data/jamlai_raw \
           --out-csv /workspace/data/manifest_adv.csv --max-deg 3 --workers 12

和 `utils/preprocess.py` 的关系:**流程逐条照搬,只换退化谱**。

    preprocess.py         6 种自造退化(jpeg/blur/resize/noise/jitter/crop)
    advanced_degrade.py   19 种 NTIRE 官方 val 畸变

为什么值得换:官方 val 用 19 种、val_hard 用 19 种(另一套)、test 用 22~24 种,而我们
只有 6 种。实测这个域差距值 **4 个 AUC 点** —— 一个在 7,482 张官方图上拟合的逻辑回归,
打赢了在 100,000 张自建图上训练的三专家 MLP。换退化谱是缩小它最直接的一招。

---------------------------------------------------------------------------
19 个畸变的来源(必须分清,报告里不能混为一谈)

**12 个来自官方管线**,`utils/ntire_aug/` 是竞赛页 "Transformations Script" 的逐字节
vendored 副本,参数表也是官方的 `distortion_range`:

    gausblur lensblur colorshift colorsat jpeg whitenoise
    impulsenoise brighten darken jitter quantization lincontrchange

**7 个是本文件实现的**,官方只公开了训练管线的脚本,没公开 val/test 那 7 个的实现:

    downscale randomcrop randomaspectcrop rgbshift pixelate multnoise motionblur

但**参数不是我们编的** —— 逐个取自 `val_labels.csv` 里观测到的官方 `distortion_scales`
(见 EXTRA_RANGE 的注释)。算子语义按名字的标准含义实现。报告里要写成
「算子名与参数取自 NTIRE val 标注,实现为我们自己的」,不能说成「官方实现」。

---------------------------------------------------------------------------
纪律(与 utils/preprocess.py 同源,编号对齐)

1. **一律存 PNG**  jpeg 档在内存里编解码一遍再存 PNG,保留块效应像素但不引入第二次、
   **没有记录在码字里**的压缩。

2. **确定性**  每张图的退化分配由 (--seed, 相对路径) 派生。官方算子内部用到三个全局
   随机源(`random` / `np.random` / `torch`),逐图重置,否则多进程下重跑结果不同。
   面积裁剪与插值内核各用**独立 RNG 流**,加不加它们都不会挪动退化分配。

3. **跳过自己的产物**  产物写在原图旁边(`__clean.png` / `__deg.png`),重跑按后缀跳过,
   不可能在干扰图上再叠一层。

4. **归一化在退化之前**  ① normalize -> 512  ② 施加退化  ③ 尺寸变了再 normalize 回来。
   反过来做会把 JPEG 的 8x8 块网格重采样掉、把噪声平均掉、把模糊的有效 σ 改掉。

5. **施加顺序固定**,不是随机的。见 PIPELINE_ORDER 的注释 —— 顺序错了码字与像素会静默
   对不上,而且不报错。

6. **train/val 按源图分组划分**  同一张源图的干净版与退化版必须落同一侧。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from cache_features import DEG_DIMS, N_LEVELS, TAXONOMY_NAME  # noqa: E402
from utils.ntire_compat import apply_numpy2_patch  # noqa: E402
from utils.preprocess import (  # noqa: E402
    LABEL_HINTS, area_crop, infer_label, normalize, pick_interp, rng_for,
)

apply_numpy2_patch()

from utils.ntire_aug.utils_data import (  # noqa: E402
    distortion_functions, distortion_range,
)

N_OFFICIAL_LEVELS = 5

# --------------------------------------------------------------------------- 官方没给实现的 7 个
#
# 参数逐个取自 val_labels.csv 里观测到的官方 distortion_scales,**不是我们编的**:
#
#   downscale         连续,3708 个互不相同的值,范围 [0.30, 0.80] -> 等宽 5 档
#   randomcrop        [0.4, 0.5, 0.6, 0.7, 0.8]   保留边长比例
#   randomaspectcrop  [0.4, 0.5, 0.6, 0.7, 0.8]   保留面积比例,长宽比另抽
#   rgbshift          [10, 20, 30, 40, 50]        通道平移像素数
#   pixelate          [0.01, 0.05, 0.1, 0.2, 0.5] 下采样倍率(越小块越大)
#   multnoise         [0.001, 0.005, 0.01, 0.015, 0.035]  乘性噪声方差
#   motionblur        [1, 2, 4, 6, 10]            运动模糊核长度(像素)
#
# 方向统一为「下标越大越坏」,与官方 12 个的 distortion_range 一致。
EXTRA_RANGE = {
    "downscale": [0.8, 0.675, 0.55, 0.425, 0.3],      # 等宽 5 档的中点,倍率越小越坏
    "randomcrop": [0.8, 0.7, 0.6, 0.5, 0.4],
    "randomaspectcrop": [0.8, 0.7, 0.6, 0.5, 0.4],
    "rgbshift": [10, 20, 30, 40, 50],
    "pixelate": [0.5, 0.2, 0.1, 0.05, 0.01],
    "multnoise": [0.001, 0.005, 0.01, 0.015, 0.035],
    "motionblur": [1, 2, 4, 6, 10],
}


def _downscale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """缩小再放回原尺寸 —— 留下重采样痕迹,尺寸不变。"""
    import torch.nn.functional as F
    _, h, w = x.shape
    small = F.interpolate(x.unsqueeze(0), size=(max(8, int(h * scale)), max(8, int(w * scale))),
                          mode="bicubic", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)


def _randomcrop(x: torch.Tensor, keep: float) -> torch.Tensor:
    """随机位置裁出 keep x keep 的正方 —— **改尺寸**,由调用方归一化回来。"""
    _, h, w = x.shape
    nh, nw = max(16, int(round(h * keep))), max(16, int(round(w * keep)))
    t = int(np.random.randint(0, h - nh + 1)); l = int(np.random.randint(0, w - nw + 1))
    return x[:, t : t + nh, l : l + nw]


def _randomaspectcrop(x: torch.Tensor, area: float) -> torch.Tensor:
    """按面积比裁,长宽比在 3/4~4/3 随机 —— **改尺寸且改长宽比**。"""
    _, h, w = x.shape
    ar = float(np.random.uniform(3 / 4, 4 / 3))
    nh = max(16, min(h, int(round((area * h * w / ar) ** 0.5))))
    nw = max(16, min(w, int(round(nh * ar))))
    t = int(np.random.randint(0, h - nh + 1)); l = int(np.random.randint(0, w - nw + 1))
    return x[:, t : t + nh, l : l + nw]


def _rgbshift(x: torch.Tensor, px: float) -> torch.Tensor:
    """三个通道各自随机平移几个像素 —— 制造彩色边缘,尺寸不变(循环移位)。"""
    n = int(round(px))
    out = x.clone()
    for c in range(min(3, x.shape[0])):
        dy = int(np.random.randint(-n, n + 1)); dx = int(np.random.randint(-n, n + 1))
        out[c] = torch.roll(x[c], shifts=(dy, dx), dims=(0, 1))
    return out


def _pixelate(x: torch.Tensor, scale: float) -> torch.Tensor:
    """最近邻下采样再最近邻放回 —— 方块化,尺寸不变。用 NEAREST 才有硬块边。"""
    import torch.nn.functional as F
    _, h, w = x.shape
    small = F.interpolate(x.unsqueeze(0), size=(max(2, int(h * scale)), max(2, int(w * scale))),
                          mode="nearest")
    return F.interpolate(small, size=(h, w), mode="nearest").squeeze(0)


def _multnoise(x: torch.Tensor, var: float) -> torch.Tensor:
    """乘性(斑点)噪声 x * (1 + n),n ~ N(0, var) —— 亮处噪声更强,与加性噪声不同。"""
    n = torch.from_numpy(np.random.normal(0.0, var ** 0.5, size=tuple(x.shape)).astype(np.float32))
    return (x * (1.0 + n)).clamp(0, 1)


def _motionblur(x: torch.Tensor, length: float) -> torch.Tensor:
    """随机方向的线性运动模糊。核是一条经过中心的线段,长度 = length 像素。"""
    import torch.nn.functional as F
    k = max(3, int(round(length)) | 1)                       # 奇数核
    ker = np.zeros((k, k), np.float32)
    ang = float(np.random.uniform(0, np.pi))
    c = (k - 1) / 2.0
    for i in range(k):
        t = i - c
        y = int(round(c + t * np.sin(ang))); xx = int(round(c + t * np.cos(ang)))
        if 0 <= y < k and 0 <= xx < k:
            ker[y, xx] = 1.0
    ker /= max(ker.sum(), 1.0)
    w = torch.from_numpy(ker)[None, None].repeat(x.shape[0], 1, 1, 1)
    return F.conv2d(F.pad(x.unsqueeze(0), (k // 2,) * 4, mode="reflect"),
                    w, groups=x.shape[0]).squeeze(0).clamp(0, 1)


EXTRA_FUNCTIONS = {
    "downscale": _downscale, "randomcrop": _randomcrop,
    "randomaspectcrop": _randomaspectcrop, "rgbshift": _rgbshift,
    "pixelate": _pixelate, "multnoise": _multnoise, "motionblur": _motionblur,
}

# --------------------------------------------------------------------------- 施加顺序
#
# **不是随机顺序。** 每一段的位置都有物理理由,换了会让码字与像素静默对不上:
#
#   ① 几何/重采样最先 —— randomcrop/aspectcrop 改尺寸,做完立刻归一化回 512;
#      downscale/pixelate 虽然保尺寸,但也是重采样,放在后面会把后续算子的痕迹一起重采样掉。
#   ② 光度 —— 逐像素点变换,不动空间结构,放哪都行,统一排这里便于复现。
#   ③ 模糊在噪声**之前** —— 真实成像链里模糊是光学的(传感器之前),噪声是传感器的。
#      反过来做等于用模糊把自己刚加的噪声抹掉,noise 那几维的证据会静默消失。
#   ④ 量化与编码最后,jpeg **永远最末** —— 它是"传输编码",8x8 块网格必须活到存盘那一刻。
PIPELINE_ORDER = [
    "randomcrop", "randomaspectcrop", "downscale", "pixelate",
    "lincontrchange", "brighten", "darken", "colorsat", "colorshift", "rgbshift", "jitter",
    "gausblur", "lensblur", "motionblur",
    "whitenoise", "impulsenoise", "multnoise",
    "quantization", "jpeg",
]
SIZE_CHANGING = {"randomcrop", "randomaspectcrop"}


def value_of(dim: str, level: int) -> float:
    table = distortion_range.get(dim) or EXTRA_RANGE[dim]
    return table[level - 1]


def apply_one(x: torch.Tensor, dim: str, level: int) -> torch.Tensor:
    fn = distortion_functions.get(dim) or EXTRA_FUNCTIONS[dim]
    return torch.clip(fn(x.clone(), value_of(dim, level)).to(torch.float32), 0, 1)


def seed_globals(seed: int, rel: str) -> None:
    """官方算子内部用 random / np.random / torch 的全局源,逐图重置(纪律 2)。"""
    h = int(hashlib.sha256(f"{seed}|{rel}|globals".encode()).hexdigest()[:8], 16)
    random.seed(h); np.random.seed(h); torch.manual_seed(h)


def to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, np.uint8).transpose(2, 0, 1).astype(np.float32) / 255.0)


def to_pil(x: torch.Tensor) -> Image.Image:
    a = (x.clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(a, "RGB")


# --------------------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="数据集根目录,递归查找图片")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--suffix", default="__deg", help="干扰图的文件名后缀")
    ap.add_argument("--clean-suffix", default="__clean")
    ap.add_argument("--also-skip", default="", metavar="S1,S2",
                    help="额外要跳过的文件名片段(逗号分隔)。注意 --suffix __deg2 并不会"
                         "排除 __deg —— 子串陷阱,踩过一次")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-deg", type=int, default=3, metavar="K",
                    help="每图随机施加 1..K 种畸变(维度不重复)。默认 3")
    ap.add_argument("--label", type=int, default=None, choices=[0, 1],
                    help="无法从目录名推断真假时统一指定")
    ap.add_argument("--no-clean-rows", action="store_true", help="不写干净行(默认写)")
    ap.add_argument("--split-val", type=float, default=0.15)
    ap.add_argument("--area-prob", type=float, default=0.0,
                    help="以此概率先随机裁一块(占原面积 area-lo~area-hi)再缩到 size。"
                         "务必对所有来源用同一个值,否则'有没有重采样痕迹'会编码标签")
    ap.add_argument("--area-lo", type=float, default=0.30)
    ap.add_argument("--area-hi", type=float, default=0.90)
    ap.add_argument("--fit", default="crop", choices=["resize", "crop"])
    ap.add_argument("--interp", default="random", choices=["fixed", "random"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if TAXONOMY_NAME not in ("ntireval", "synthetic6"):
        raise SystemExit(
            f"当前 SQUADE_TAXONOMY={TAXONOMY_NAME!r}。本脚本支持:\n"
            f"  ntireval    19 维官方 val 畸变\n"
            f"  synthetic6  6 维(我们原来那六类)+ 官方算子与强度")
    missing = [d for d in DEG_DIMS if d not in distortion_range and d not in EXTRA_RANGE]
    if missing:
        raise SystemExit(f"码本里这些维没有参数表: {missing}")
    # 顺序表是 19 维的全集;码本可以只取子集(如 synthetic6 的 6 维),
    # 但码本里的每一维都必须在顺序表里有位置,否则施加顺序无定义
    uncovered = sorted(set(DEG_DIMS) - set(PIPELINE_ORDER))
    if uncovered:
        raise SystemExit(f"码本这些维不在 PIPELINE_ORDER 里,施加顺序无定义: {uncovered}")
    if not 1 <= args.max_deg <= len(DEG_DIMS):
        raise SystemExit(f"--max-deg 要在 1..{len(DEG_DIMS)}")

    root = Path(args.data).resolve()
    skip_frags = [args.suffix, args.clean_suffix] + \
                 [s for s in args.also_skip.split(",") if s]
    srcs = sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                  and not any(f in p.name for f in skip_frags))
    if args.limit:
        srcs = srcs[: args.limit]
    if not srcs:
        raise SystemExit(f"{root} 下没有找到图片(已排除含 {skip_frags} 的文件)")

    print(f"源目录    : {root}")
    print(f"源图      : {len(srcs)} 张")
    print(f"码本      : {TAXONOMY_NAME}  {len(DEG_DIMS)} 维 x {N_LEVELS} 档")
    print(f"策略      : 每图随机 1..{args.max_deg} 种畸变(维度不重复) x 各自 1 个强度,"
          f"{'不含' if args.no_clean_rows else '含'}干净行")
    print(f"归一化    : --fit {args.fit}  --interp {args.interp}"
          + (f"  面积裁剪 {args.area_prob:.0%} @ {args.area_lo}~{args.area_hi}"
             if args.area_prob > 0 else ""))
    print(f"施加顺序  : {' -> '.join(PIPELINE_ORDER)}")
    print("畸变来源  :")
    for d in DEG_DIMS:
        tag = "官方管线" if d in distortion_range else "★本文件实现(参数取自 val 标注)"
        print(f"  {d:<18} {value_of(d, 1)} -> {value_of(d, N_OFFICIAL_LEVELS)}   {tag}")
    print(f"模式      : {'DRY RUN(不写文件)' if args.dry_run else '写入'}\n")

    rows, jobs = [], []
    stats, lvl_stats, n_stats = Counter(), Counter(), Counter()
    for p in srcs:
        rel = p.relative_to(root).as_posix()
        label = infer_label(p, root, args.label)
        r = rng_for(args.seed, rel)
        k = 1 if args.max_deg == 1 else int(r.integers(1, args.max_deg + 1))
        picked_dims = [DEG_DIMS[i] for i in r.choice(len(DEG_DIMS), size=k, replace=False)]
        code = [0] * len(DEG_DIMS)
        for d in picked_dims:
            lv = int(r.integers(1, N_OFFICIAL_LEVELS + 1))
            code[DEG_DIMS.index(d)] = lv
            stats[d] += 1; lvl_stats[(d, lv)] += 1
        n_stats[k] += 1
        # 按 PIPELINE_ORDER 排,不按抽中的顺序(纪律 5)
        ordered = [(d, code[DEG_DIMS.index(d)]) for d in PIPELINE_ORDER
                   if d in DEG_DIMS and code[DEG_DIMS.index(d)]]

        deg_path = p.with_name(f"{p.stem}{args.suffix}.png")
        clean_path = p.with_name(f"{p.stem}{args.clean_suffix}.png")
        gid = hashlib.sha256(rel.encode()).hexdigest()[:16]
        base = {"group": gid, "label": label, "source": rel.split("/")[0]}
        if not args.no_clean_rows:
            d0 = dict(base, path=str(clean_path), image_id=f"{gid}_clean")
            d0.update({f"d{i}": 0 for i in range(len(DEG_DIMS))})
            d0.update(deg_names="", deg_values="")
            rows.append(d0)
        d1 = dict(base, path=str(deg_path), image_id=f"{gid}_deg")
        d1.update({f"d{i}": code[i] for i in range(len(DEG_DIMS))})
        d1.update(deg_names="|".join(d for d, _ in ordered),
                  deg_values="|".join(f"{value_of(d, l):g}" for d, l in ordered))
        rows.append(d1)

        rs = pick_interp(args.seed, rel) if args.interp == "random" else None
        jobs.append((p, clean_path, deg_path, ordered, rs, rel))

    print(f"每图畸变个数: {dict(sorted(n_stats.items()))}")
    print(f"\n{'畸变':<20}{'图数':>7}   各档图数 1..5")
    for d in DEG_DIMS:
        if stats[d]:
            print(f"  {d:<18}{stats[d]:>7}   "
                  + " ".join(f"{lvl_stats[(d, l)]:>5}" for l in range(1, N_OFFICIAL_LEVELS + 1)))

    if not args.dry_run:
        failed = write_all(jobs, args)
        if failed:
            dead = {str(x) for x in failed}
            rows = [r for r in rows if r["path"] not in dead]
            print(f"[警告] {len(failed)} 张写失败,已从 manifest 剔除")

    # ---- 按源图组划分(纪律 6) ----
    groups = sorted({r["group"] for r in rows})
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{args.seed}|{g}".encode()).hexdigest())
    va = set(order[: int(round(len(order) * args.split_val))])
    for r in rows:
        r["split"] = "val" if r["group"] in va else "train"

    cols = (["path", "image_id", "group", "label", "source"]
            + [f"d{i}" for i in range(len(DEG_DIMS))]
            + ["deg_names", "deg_values", "split"])
    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        with open(outp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    by = defaultdict(set)
    for r in rows:
        by[r["group"]].add(r["split"])
    bad = sum(1 for v in by.values() if len(v) > 1)
    print(f"\n{len(rows)} 行 -> {outp}")
    print(f"  源图组 {len(by)}   跨 split 的组 {bad}  (必须为 0)")
    print(f"  split  {dict(Counter(r['split'] for r in rows))}")
    print(f"  label  {dict(Counter(r['label'] for r in rows))}")
    print(f"  列序 d0..d{len(DEG_DIMS)-1} = {DEG_DIMS}")
    return 1 if bad else 0


def write_all(jobs, args):
    """并行写图。每图的随机源由 rel 派生,并行**不改变**任何一张图的像素(纪律 2)。"""
    def one(job):
        p, clean_path, deg_path, ordered, rs, rel = job
        if deg_path.exists() and clean_path.exists() and not args.overwrite:
            return None
        try:
            with Image.open(p) as im:
                img, frac = area_crop(im.convert("RGB"), args.seed, rel,
                                      args.area_lo, args.area_hi, args.area_prob)
                # 被面积裁剪过的必须走 resize 分支,否则 fit=crop 会直接中心裁、等于没做
                fit_i = "resize" if frac is not None else args.fit
                base = normalize(img, args.size, rs, fit_i)          # ① 归一化
                if args.overwrite or not clean_path.exists():
                    base.save(clean_path, "PNG")                     # 纪律 1
                seed_globals(args.seed, rel)                         # 纪律 2
                x = to_tensor(base)
                for d, lv in ordered:                                # ② 按固定顺序施加
                    x = apply_one(x, d, lv)
                    if d in SIZE_CHANGING:                           # ③ 改了尺寸就地归一化
                        x = to_tensor(normalize(to_pil(x), args.size, rs, fit_i))
                to_pil(x).save(deg_path, "PNG")
        except Exception as e:
            print(f"\n[跳过] {p}: {type(e).__name__}: {e}")
            return deg_path
        return None

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x
    failed = []
    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in tqdm(ex.map(one, jobs), total=len(jobs), desc="写退化图", unit="img"):
                if res is not None:
                    failed.append(res)
    else:
        for j in tqdm(jobs, desc="写退化图", unit="img"):
            res = one(j)
            if res is not None:
                failed.append(res)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())

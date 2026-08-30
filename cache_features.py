"""把整个数据集过一遍冻结的 DINOv3,把逐层特征落盘。

用法:
    python cache_features.py --manifest data/manifest.csv --out cache/probe          # 全 33 层
    python cache_features.py --manifest data/manifest.csv --out cache/train --layers 20,24,28
    python cache_features.py --manifest data/manifest.csv --out cache/smoke --limit 32

manifest 至少要有三样东西,列名自动识别(见 COLUMN_ALIASES):
    路径     path / filepath / file / image / filename
    真假标签 label / is_fake / target / y            (0=真, 1=假)
    退化码字 d0..d5 / deg_jpeg,deg_blur,...          6 个整数列, 取值见 LEVELS_PER_DIM
             或单列 deg_code = "0,1,0,2,0,0"
可选: generator / split / image_id

输出:
    cache/xxx/features.npy   (M, L, D) fp16   L=层数, D=池化维度(cls+mean+std -> 3*1280)
            /prenorm.npy    (M, L, 2) fp32   归一化前的 patch 范数 均值/标准差
            /meta.csv       M 行, 行号即 features 的第一维下标
            /config.json    出处指纹, 防止不同配置的缓存被混用
            /.done.npy      (M,) bool, 断点续传用

设计要点:
  * 预分配 memmap 而非 append —— 五万条样本是小时级任务,中途挂了要能原地续跑。
  * config.json 记录 backbone / dtype / layers / pool / 预处理协议。续跑或被下游读取时
    先比对指纹,配置不一致直接拒绝,避免把两次不同设置的特征混进同一份缓存。
  * NaN/Inf 守卫。bf16 本身没有溢出风险(见 models/dinov3.py 纪律 5),但损坏的图片文件
    仍可能产出非有限值。缓存一旦写进 NaN 就是静默污染,所以逐样本检查并记录到 meta。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.dinov3 import DEFAULT_MODEL, DINOv3Backbone  # noqa: E402

# 码本(维度 / 档数 / 每维合法取值)现在由 utils/deg_taxonomy.py 统一定义,用环境变量
# SQUADE_TAXONOMY 在 synthetic(历史默认,6 维)与 ntire(NTIRE 官方退化分组,8 维)之间切换。
# 抽出去的原因见那个文件的 docstring;这里保留同名再导出,下游六个文件的
# `from cache_features import DEG_DIMS, N_LEVELS, DEG_CODE_VALUES` 一行都不用改。
#
# 注意 N_LEVELS 是**统一上界**,各维实际用到的档数可以不同(synthetic 的 blur 只有
# 0/1/2/4,留有空洞)。不要为了省参数改成逐维不等长 —— 那会让 CoralHead 的输出变成
# 不规则张量,decode/loss/stack 全要改成 list 推导。
from utils.deg_taxonomy import (  # noqa: E402
    DEG_CODE_VALUES,
    DEG_DIMS,
    FAMILIES,
    N_LEVELS,
    TAXONOMY_NAME,
)

EXPECT_SIZE = 512     # 数据集已统一到 512x512

# 按设备的默认 batch。MPS 上实测 batch 越大越慢(1.19 / 1.25 / 1.44 / 2.09 s/图,
# 对应 batch 1/4/8/16):output_hidden_states=True 要把 33 层全留在内存,batch=16 时
# 光 hidden_states 就 ~1.4 GB,在统一内存上直接压出带宽瓶颈。独显没这个问题,可调大。
DEFAULT_BATCH = {"mps": 1, "cuda": 16, "cpu": 4}

COLUMN_ALIASES = {
    "path": ["path", "filepath", "file", "image", "image_path", "filename"],
    "label": ["label", "is_fake", "target", "y", "fake"],
    "generator": ["generator", "gen", "model", "source"],
    "split": ["split", "subset", "fold"],
    "image_id": ["image_id", "id", "uid"],
}


# --------------------------------------------------------------------------- manifest

def _resolve(fieldnames: list[str], key: str) -> str | None:
    lower = {f.lower().strip(): f for f in fieldnames}
    for alias in COLUMN_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


def _resolve_deg_columns(fieldnames: list[str]) -> list[str] | str:
    """返回 6 个码字列名,或单个 'deg_code' 风格的列名。"""
    lower = {f.lower().strip(): f for f in fieldnames}
    for pattern in (
        [f"d{i}" for i in range(len(DEG_DIMS))],
        [f"deg_{d}" for d in DEG_DIMS],
        list(DEG_DIMS),
    ):
        if all(p in lower for p in pattern):
            return [lower[p] for p in pattern]
    for single in ("deg_code", "degradation", "code", "deg"):
        if single in lower:
            return lower[single]
    raise SystemExit(
        f"manifest 里找不到退化码字。需要 {len(DEG_DIMS)} 个整数列"
        f"(d0..d{len(DEG_DIMS) - 1} / deg_* / "
        f"{','.join(DEG_DIMS)}) 或单列 deg_code=\"0,1,0,2,0,0\"。当前列: {fieldnames}"
    )


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[警告] {msg}")


def _parse_code(row: dict, deg_cols: list[str] | str, lineno: int) -> list[int]:
    if isinstance(deg_cols, str):
        raw = [t for t in str(row[deg_cols]).replace(";", ",").replace(" ", ",").split(",") if t]
    else:
        raw = [row[c] for c in deg_cols]
    if len(raw) != len(DEG_DIMS):
        raise SystemExit(
            f"manifest 第 {lineno} 行退化码字是 {len(raw)} 维,当前码本 "
            f"({TAXONOMY_NAME}) 要 {len(DEG_DIMS)} 维 {DEG_DIMS}: {raw}\n"
            f"多半是 SQUADE_TAXONOMY 没设对 —— manifest 与码本必须来自同一套。")
    code = []
    for dim, v in zip(DEG_DIMS, raw):
        try:
            iv = int(float(v))
        except (TypeError, ValueError):
            raise SystemExit(f"manifest 第 {lineno} 行 {dim} 档位不是整数: {v!r}")
        if not 0 <= iv < N_LEVELS:
            raise SystemExit(
                f"manifest 第 {lineno} 行 {dim}={iv} 越界。码字取值上界是 {N_LEVELS} 档 "
                f"(0..{N_LEVELS - 1})。"
            )
        if iv not in DEG_CODE_VALUES.get(dim, range(N_LEVELS)):
            _warn_once(f"{dim}={iv} 不在该维的合法取值 {DEG_CODE_VALUES[dim]} 内。"
                       f"码字范围本身没越界,但退化谱里没有这一档 —— "
                       f"多半是数据生成时对错了列,或退化谱改过而 manifest 是旧的。")
        code.append(iv)
    return code


def read_manifest(path: Path, root: Path, limit: int | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit(f"{path} 是空的或不是合法 CSV")
        col_path = _resolve(reader.fieldnames, "path")
        col_label = _resolve(reader.fieldnames, "label")
        if col_path is None or col_label is None:
            raise SystemExit(f"manifest 缺少 路径 或 标签 列。当前列: {reader.fieldnames}")
        deg_cols = _resolve_deg_columns(reader.fieldnames)
        col_gen = _resolve(reader.fieldnames, "generator")
        col_split = _resolve(reader.fieldnames, "split")
        col_id = _resolve(reader.fieldnames, "image_id")

        rows = []
        for lineno, row in enumerate(reader, start=2):
            p = Path(str(row[col_path]).strip())
            if not p.is_absolute():
                p = root / p
            label = int(float(row[col_label]))
            if label not in (0, 1):
                raise SystemExit(f"manifest 第 {lineno} 行 label={label},只接受 0(真)/1(假)")
            code = _parse_code(row, deg_cols, lineno)
            rows.append(
                {
                    "path": str(p),
                    "image_id": (row.get(col_id) or p.stem) if col_id else p.stem,
                    "label": label,
                    "generator": (row.get(col_gen) or "") if col_gen else "",
                    "split": (row.get(col_split) or "") if col_split else "",
                    "code": code,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"{path} 里一条样本都没有")
    return rows


# --------------------------------------------------------------------------- 缓存

def build_config(args, bb: DINOv3Backbone, layers: list[int], dim: int, n: int) -> dict:
    return {
        "backbone": bb.name,
        "dtype": str(bb.dtype),
        "n_layers_total": bb.n_layers,
        "hidden_size": bb.hidden_size,
        "n_registers": bb.n_registers,
        "layers": layers,
        "pool": args.pool,
        "feature_dim": dim,
        "n_samples": n,
        # 预处理协议:下游据此确认特征是在什么输入上算出来的
        "crop_size": getattr(args, "crop_size", None) or EXPECT_SIZE,
        "crop_mode": "center",
        "do_resize": False,
        "layer_norm": "model.norm applied to every extracted layer",
        "taxonomy": TAXONOMY_NAME,
        "deg_dims": DEG_DIMS,
        "deg_levels": N_LEVELS,
        "deg_code_values": DEG_CODE_VALUES,
        "manifest_sha1": hashlib.sha1(Path(args.manifest).read_bytes()).hexdigest(),
    }


def fingerprint(cfg: dict) -> str:
    """比对续跑/复用时用的指纹。刻意排除 n_samples 之外的易变字段之外的一切。"""
    keys = ["backbone", "dtype", "layers", "pool", "feature_dim", "n_samples",
            "crop_size", "crop_mode", "do_resize", "manifest_sha1", "taxonomy"]
    d = {k: cfg[k] for k in keys}
    # delta 必须进指纹:同样的层、同样的 manifest,累积特征和层间增量是**两份不同的数据**,
    # 不区分的话续跑会把增量缓存当成累积缓存接着写,而且不报错。
    # 只在 delta=True 时才塞进字典 —— 无条件加 "delta": False 会改变 JSON、
    # 让所有既有缓存的指纹作废(试过一次,三份缓存全部对不上)。
    if cfg.get("delta"):
        d["delta"] = True
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def write_meta(out: Path, rows: list[dict]) -> None:
    with open(out / "meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "image_id", "path", "label", "generator", "split"]
                   + [f"deg_{d}" for d in DEG_DIMS] + ["nonfinite"])
        for i, r in enumerate(rows):
            w.writerow([i, r["image_id"], r["path"], r["label"], r["generator"], r["split"]]
                       + r["code"] + [0])


def mark_nonfinite(out: Path, bad: set[int]) -> None:
    """回填 meta.csv 的 nonfinite 列。坏样本保留在表里但标记出来,下游自行过滤。"""
    if not bad:
        return
    p = out / "meta.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    header, body = lines[0], lines[1:]
    for i in bad:
        cells = next(csv.reader([body[i]]))
        cells[-1] = "1"
        body[i] = ",".join(f'"{c}"' if "," in c else c for c in cells)
    p.write_text("\n".join([header] + body) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help=f"CSV: 路径 + 真假标签 + {len(DEG_DIMS)} 维退化码字")
    ap.add_argument("--out", required=True, help="缓存输出目录")
    ap.add_argument("--root", default=".", help="manifest 里相对路径的基准目录")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default="all",
                    help='"all" 或逗号分隔。训练阶段用探针定下的三层 "20,24,28" '
                         '(见 models/dinov3.py 的 PROBE_BANDS)')
    ap.add_argument("--crop-size", type=int, default=None,
                    help=f"中心裁剪的边长,须为 backbone 的 patch_size 整数倍。"
                         f"默认 {'{}'} —— 但 DINOv2 是 patch 14,512 不整除,要传 504")
    ap.add_argument("--pool", default="cls+mean+std", choices=["cls", "cls+mean", "cls+mean+std", "mean+std"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="默认按设备决定(见 DEFAULT_BATCH)。MPS 上实测 batch=1 最快")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条,冒烟测试用")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true", help="配置指纹不符时重建而非报错")
    ap.add_argument("--delta", action="store_true",
                    help="缓存**层间残差增量**而非累积特征:band0=norm(h_a), "
                         "band1=norm(h_b-h_a), band2=norm(h_c-h_b)。差分在归一化前做"
                         "(残差流的可加性只在那里成立)。层数必须 >=2 且不能是 all")
    ap.add_argument("--io-workers", type=int, default=8,
                    help="后台读盘线程数。0 = 关闭预取,退回串行(历史行为)")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="预取多少个 batch。太大只会多占内存,读盘跟得上就够了")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(Path(args.manifest), Path(args.root), args.limit)
    n = len(rows)

    bb = DINOv3Backbone(name=args.model, device=args.device)
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH.get(bb.device_.type, 4)
    crop_size = args.crop_size or EXPECT_SIZE
    if crop_size % bb.patch_size:
        raise SystemExit(f"--crop-size {crop_size} 不是 {args.model} 的 patch_size="
                         f"{bb.patch_size} 的整数倍;DINOv2(patch 14)请用 504")
    pre = bb.make_preprocessor(crop_size=crop_size, crop="center")
    layers = list(range(bb.n_layers + 1)) if args.layers == "all" else [int(t) for t in args.layers.split(",")]
    for li in layers:
        if not 0 <= li <= bb.n_layers:
            raise SystemExit(f"层索引 {li} 越界,合法范围 0..{bb.n_layers}")
    mult = {"cls": 1, "cls+mean": 2, "cls+mean+std": 3, "mean+std": 2}[args.pool]
    dim = mult * bb.hidden_size

    if args.delta:
        if args.layers == "all" or len(layers) < 2:
            raise SystemExit("--delta 需要显式给 >=2 层,不能用 all")
    cfg = build_config(args, bb, layers, dim, n)
    if args.delta:
        # 写进指纹的一部分:差分缓存与同层的累积缓存**不是**一回事,
        # 不区分的话会在同一个 --out 里互相覆盖而不报错
        cfg["delta"] = True
        cfg["layer_norm"] = "差分在 pre-norm token 上做,之后统一 model.norm"
    fp = fingerprint(cfg)
    cfg_path = out / "config.json"
    fresh = True
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old.get("_fingerprint") == fp:
            fresh = False
            print(f"[续跑] 指纹一致,复用 {out}")
        elif not args.overwrite:
            raise SystemExit(
                f"{out} 已有一份配置不同的缓存(指纹 {old.get('_fingerprint', '?')[:8]} "
                f"!= {fp[:8]})。换个 --out,或加 --overwrite 重建。"
            )
    cfg["_fingerprint"] = fp
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    L = len(layers)
    feat_path, pren_path, done_path = out / "features.npy", out / "prenorm.npy", out / ".done.npy"
    mode = "r+" if (not fresh and feat_path.exists()) else "w+"
    feats = np.lib.format.open_memmap(feat_path, mode=mode, dtype=np.float16, shape=(n, L, dim))
    prens = np.lib.format.open_memmap(pren_path, mode=mode, dtype=np.float32, shape=(n, L, 2))
    done = (np.load(done_path) if (not fresh and done_path.exists())
            else np.zeros(n, dtype=bool))
    if fresh:
        write_meta(out, rows)

    todo = np.flatnonzero(~done).tolist()
    print(f"backbone : {bb.name}  ({bb.device_}/{bb.dtype})")
    print(f"层        : {L} 层 {layers if L <= 6 else f'{layers[0]}..{layers[-1]} (全)'}")
    print(f"池化      : {args.pool} -> {dim} 维")
    print(f"batch     : {args.batch_size}")
    print(f"样本      : {n} 条,待处理 {len(todo)} 条")
    print(f"预计体积  : {n * L * dim * 2 / 2**30:.2f} GB (features) "
          f"+ {n * L * 2 * 4 / 2**20:.1f} MB (prenorm)")
    if not todo:
        print("全部已完成。")
        return 0

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):  # noqa: ANN001
            return x

    bad: set[int] = set()

    def load_batch(idxs):
        """后台线程干的活:读盘 + 预处理。**不碰任何共享状态** ——
        bad/done 的更新和打印留给主线程,否则会和周期性的 flush 打架。
        pre() 的随机性由 image_id 派生,与线程无关,所以预取不破坏确定性。
        """
        batch, keep, bad_local, notes = [], [], [], []
        for i in idxs:
            try:
                img = Image.open(rows[i]["path"]).convert("RGB")
            except Exception as e:                      # 文件缺失/损坏
                notes.append(f"[跳过] {rows[i]['path']}: {type(e).__name__}: {e}")
                bad_local.append(i)
                continue
            if img.size != (crop_size, crop_size):
                notes.append(f"[注意] {rows[i]['path']} 是 {img.size},非 "
                             f"{crop_size}x{crop_size};走 center crop 兜底(不 resize)")
            batch.append(pre(img, image_id=rows[i]["image_id"]))
            keep.append(i)
        return batch, keep, bad_local, notes

    # 串行版里读盘和前向永不重叠:读 16 张时 GPU 空转,前向时磁盘空转。
    # 实测网络盘上 GPU 利用率在 0%~100% 之间来回跳,平均约 50% —— 一半墙钟在等 I/O。
    # 这里用后台线程预取。future 按提交顺序消费,写入顺序与串行版完全一致。
    chunks = [todo[b : b + args.batch_size] for b in range(0, len(todo), args.batch_size)]
    pbar = tqdm(chunks, unit="batch")

    if args.io_workers > 0:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=args.io_workers)
        pending, it = deque(), iter(chunks)
        for _ in range(max(1, args.prefetch)):
            nxt = next(it, None)
            if nxt is None:
                break
            pending.append(pool.submit(load_batch, nxt))

        def next_batch():
            if not pending:
                return None
            fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(load_batch, nxt))
            return fut.result()
    else:
        pool, _serial = None, iter(chunks)

        def next_batch():
            nxt = next(_serial, None)
            return None if nxt is None else load_batch(nxt)

    for bi, _chunk in enumerate(pbar):
        got = next_batch()
        if got is None:
            break
        batch, keep, bad_local, notes = got
        for msg in notes:
            print("\n" + msg)
        for i in bad_local:
            bad.add(i)
            done[i] = True
        if not batch:
            continue

        shapes = {tuple(t.shape[-2:]) for t in batch}
        if len(shapes) > 1:                             # 尺寸不一无法 stack,退化成逐张
            groups = [([t], [i]) for t, i in zip(batch, keep)]
        else:
            groups = [(batch, keep)]

        for tensors, ids in groups:
            x = torch.cat(tensors, dim=0)
            f = (bb.forward_deltas(x, layers=layers, return_norm_stats=True)
                 if args.delta else
                 bb.forward_layers(x, layers=layers, return_norm_stats=True))
            pooled = torch.stack([bb.pool(f[li], args.pool) for li in layers], dim=1)   # (B,L,D)
            pren = torch.stack([f[li]["prenorm_stats"] for li in layers], dim=1)        # (B,L,2)
            pooled_np = pooled.float().cpu().numpy()
            pren_np = pren.float().cpu().numpy()
            for k, i in enumerate(ids):
                if not (np.isfinite(pooled_np[k]).all() and np.isfinite(pren_np[k]).all()):
                    print(f"\n[非有限] idx={i} {rows[i]['path']} —— 已标记,下游请过滤")
                    bad.add(i)
                feats[i] = pooled_np[k].astype(np.float16)
                prens[i] = pren_np[k]
                done[i] = True

        if bi % 50 == 0:
            feats.flush(); prens.flush(); np.save(done_path, done)
            if bb.device_.type == "mps":
                torch.mps.empty_cache()
        pbar.set_postfix(done=int(done.sum()), bad=len(bad))

    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    feats.flush(); prens.flush(); np.save(done_path, done)
    mark_nonfinite(out, bad)
    print(f"\n完成: {int(done.sum())}/{n} 条写入 {out}")
    if bad:
        print(f"其中 {len(bad)} 条异常(缺失/损坏/非有限),已在 meta.csv 的 nonfinite 列标记为 1")
    return 0


# --------------------------------------------------------------------------- 下游读取

def load_cache(cache_dir: str | Path, drop_bad: bool = True):
    """给 probe_layers.py / train.py 用的读取入口。

    Returns: (features (M,L,D) fp16 memmap, prenorm (M,L,2) fp32, meta list[dict], config dict)
    """
    d = Path(cache_dir)
    cfg = json.loads((d / "config.json").read_text())
    feats = np.load(d / "features.npy", mmap_mode="r")
    prens = np.load(d / "prenorm.npy", mmap_mode="r")
    with open(d / "meta.csv", newline="", encoding="utf-8") as fh:
        meta = list(csv.DictReader(fh))
    for m in meta:
        m["label"] = int(m["label"])
        m["code"] = [int(m[f"deg_{x}"]) for x in DEG_DIMS]
        m["nonfinite"] = int(m.get("nonfinite", 0))
    if drop_bad:
        ok = np.array([m["nonfinite"] == 0 for m in meta])
        if not ok.all():
            idx = np.flatnonzero(ok)
            feats, prens = feats[idx], prens[idx]
            meta = [meta[i] for i in idx]
    return feats, prens, meta, cfg


if __name__ == "__main__":
    raise SystemExit(main())

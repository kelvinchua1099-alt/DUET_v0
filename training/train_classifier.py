"""Stage 0 —— 退化估计器独立预训练(免费监督)。

    python training/train_classifier.py train   --manifest data/manifest.csv --out runs/deg
    python training/train_classifier.py eval    --manifest data/manifest.csv --ckpt runs/deg/best.pt
    python training/train_classifier.py predict --manifest data/manifest.csv --ckpt runs/deg/best.pt \
                                                --pred-out cache/pred_codes.csv

「免费监督」:退化标签随数据给定,零标注成本。「独立」:本阶段的梯度只来自 CORAL 损失,
永远不接触真假分类的损失 —— 梯度隔离是可归因性的前提,路由信号必须是退化状态的函数,
不能被端到端训练吸收成「看图内容」。

predict 不是可选步骤。Stage 2 训练权重 MLP 必须喂 **预测** 码字:测试时拿不到真值,
用真值训会造成 train/test 失配,A3 的数字会虚高。真值码字只留给 A3-o(oracle 上界)。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import DEG_DIMS, N_LEVELS, read_manifest          # noqa: E402
from models.classifier import DegradationCNN, JPEG_GRID, load_estimator  # noqa: E402


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- 数据

class DegradationDataset(Dataset):
    """读原图 + manifest 里的 6 维码字。

    增广只允许保退化统计量的操作:8 对齐的随机裁剪、翻转、90 度旋转。
    **禁止** resize / 重压缩 / 调色 / 模糊 / 加噪 —— 那些会直接改变标签。
    别让后来的人往这里加 RandomResizedCrop。
    """

    def __init__(self, rows: list[dict], crop: int = 256, train: bool = True):
        self.rows, self.crop, self.train = rows, crop, train
        if crop % JPEG_GRID:
            raise ValueError(f"crop={crop} 应为 {JPEG_GRID} 的倍数,否则 JPEG 网格对不齐")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0

        c = min(self.crop, x.shape[-2] // JPEG_GRID * JPEG_GRID, x.shape[-1] // JPEG_GRID * JPEG_GRID)
        if c == 0:
            raise ValueError(f"{r['path']} 尺寸 {tuple(x.shape[-2:])} 小于一个 JPEG 块")
        max_t, max_l = x.shape[-2] - c, x.shape[-1] - c

        if self.train:
            # 用全局 RNG 而非按 image_id 派生的固定种子:后者会让每个 epoch 看到完全相同的
            # 裁剪与翻转,增广等于没开。整个 run 由 --seed 保证可复现就够了 ——
            # 确定性是缓存(cache_features.py)的要求,不是训练增广的要求。
            top = int(torch.randint(max_t // JPEG_GRID + 1, (1,))) * JPEG_GRID
            left = int(torch.randint(max_l // JPEG_GRID + 1, (1,))) * JPEG_GRID
            x = x[..., top : top + c, left : left + c]
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [-1])
            if torch.rand(1) < 0.5:
                x = torch.flip(x, [-2])
            k = int(torch.randint(4, (1,)))
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
        else:
            # 偏移同样对齐到 8:否则中心裁剪会把块网格切错相位,
            # 训练(对齐)与评估(不对齐)之间产生系统性失配
            t0 = max_t // 2 // JPEG_GRID * JPEG_GRID
            l0 = max_l // 2 // JPEG_GRID * JPEG_GRID
            x = x[..., t0 : t0 + c, l0 : l0 + c]

        # 刻意不做 ImageNet 标准化:那是逐通道的仿射,会改变噪声/对比度的绝对尺度,
        # 而绝对尺度正是 noise / jitter 档位的证据。GroupNorm 在网络内部处理归一化。
        return x, torch.tensor(r["code"], dtype=torch.long)


def split_rows(rows: list[dict], val_frac: float = 0.15) -> tuple[list[dict], list[dict]]:
    """优先用 manifest 的 split 列;没有就按 image_id 哈希确定性划分。"""
    if any(r["split"] for r in rows):
        tr = [r for r in rows if r["split"].lower() not in ("val", "valid", "test")]
        va = [r for r in rows if r["split"].lower() in ("val", "valid", "test")]
        if tr and va:
            return tr, va
    tr, va = [], []
    for r in rows:
        h = int(hashlib.sha256(r["image_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (va if h < val_frac else tr).append(r)
    return tr, va


# --------------------------------------------------------------------------- 评估

@torch.no_grad()
def evaluate(model: DegradationCNN, loader: DataLoader, device) -> dict:
    model.eval()
    pred, true, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += model.loss(logits, y).item() * len(x)
        n += len(x)
        pred.append(model.decode(logits).cpu())
        true.append(y.cpu())
    p, t = torch.cat(pred), torch.cat(true)
    return {
        "loss": loss_sum / max(n, 1),
        "per_dim_acc": {d: (p[:, i] == t[:, i]).float().mean().item() for i, d in enumerate(DEG_DIMS)},
        "per_dim_mae": {d: (p[:, i] - t[:, i]).abs().float().mean().item() for i, d in enumerate(DEG_DIMS)},
        "acc_mean": (p == t).float().mean().item(),
        "exact_match": (p == t).all(-1).float().mean().item(),   # 6 维全对
        "within_one": ((p - t).abs() <= 1).all(-1).float().mean().item(),
    }


def fmt(m: dict) -> str:
    per = " ".join(f"{d}={m['per_dim_acc'][d]:.3f}" for d in DEG_DIMS)
    return (f"loss={m['loss']:.4f} 均准={m['acc_mean']:.3f} 全对={m['exact_match']:.3f} "
            f"容错1档={m['within_one']:.3f} | {per}")


# --------------------------------------------------------------------------- 训练

def run_train(args) -> int:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    rows = read_manifest(Path(args.manifest), Path(args.root), args.limit)
    tr_rows, va_rows = split_rows(rows, args.val_frac)
    tr = DataLoader(DegradationDataset(tr_rows, args.crop, True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, drop_last=len(tr_rows) > args.batch_size)
    va = DataLoader(DegradationDataset(va_rows, args.crop, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers)

    model = DegradationCNN(dropout=args.dropout, blockiness=not args.no_blockiness).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[Stage 0] 退化估计器预训练(免费监督,梯度与分类完全隔离)")
    print(f"设备      : {device}")
    print(f"参数量    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"样本      : 训练 {len(tr_rows)} / 验证 {len(va_rows)}  crop={args.crop}")
    print(f"标签空间  : {len(DEG_DIMS)} 维 x {N_LEVELS} 档 -> 每维 {N_LEVELS - 1} 个序数 logit")
    print(f"块效应特征: {'关闭(消融)' if args.no_blockiness else '开启'}")

    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = model.loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * len(x); seen += len(x)
        m = evaluate(model, va, device)
        hist.append({"epoch": ep, "train_loss": run / max(seen, 1), **m})
        star = ""
        if m["exact_match"] > best:
            best = m["exact_match"]
            torch.save({"model": model.state_dict(),
                        # 架构参数必须完整存下:换了 stem_depth / scale_bypass / blockiness /
                        # widths 之后,用默认值重建会 state_dict 形状不匹配而加载失败
                        "cfg": {"n_dims": model.n_dims, "n_levels": model.n_levels,
                                "widths": model.widths, "stem_depth": len(model.stem_convs),
                                "scale_bypass": model.scale_bypass, "blockiness": model.blockiness,
                                "crop": args.crop},
                        "metrics": m}, out / "best.pt")
            star = " *"
        print(f"[{ep:>3}/{args.epochs}] train={run / max(seen, 1):.4f} {fmt(m)} "
              f"({time.time() - t0:.0f}s){star}")

    (out / "history.json").write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n最佳全对率 {best:.3f},权重存于 {out / 'best.pt'}")
    print(f"下一步: python training/train_classifier.py predict --ckpt {out / 'best.pt'} "
          f"--manifest {args.manifest} --pred-out cache/pred_codes.csv")
    return 0


# --------------------------------------------------------------------------- 导出 / 评估

def run_predict(args) -> int:
    device = pick_device(args.device)
    rows = read_manifest(Path(args.manifest), Path(args.root), args.limit)
    model = load_estimator(args.ckpt, device)
    loader = DataLoader(DegradationDataset(rows, args.crop, False),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    preds, softs = [], []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            preds.append(model.decode(logits).cpu())
            softs.append(model.soft_code(logits).cpu())
    p, s = torch.cat(preds).numpy(), torch.cat(softs).numpy()

    outp = Path(args.pred_out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "image_id"] + [f"pred_{d}" for d in DEG_DIMS]
                   + [f"soft_{d}" for d in DEG_DIMS] + [f"true_{d}" for d in DEG_DIMS])
        for i, r in enumerate(rows):
            w.writerow([i, r["image_id"]] + p[i].tolist()
                       + [round(float(v), 4) for v in s[i]] + r["code"])
    t = np.array([r["code"] for r in rows])
    print(f"写出 {len(rows)} 条 -> {outp}")
    print(f"全对率 {float((p == t).all(1).mean()):.3f}  均准 {float((p == t).mean()):.3f}")
    print("行序与 manifest 一致,即与 cache_features 的 meta.csv 行号一一对应。")
    return 0


def run_eval(args) -> int:
    device = pick_device(args.device)
    rows = read_manifest(Path(args.manifest), Path(args.root), args.limit)
    _, va = split_rows(rows, args.val_frac)
    model = load_estimator(args.ckpt, device)
    m = evaluate(model, DataLoader(DegradationDataset(va, args.crop, False),
                                   batch_size=args.batch_size, num_workers=args.workers), device)
    print(fmt(m))
    print("每维 MAE:", {d: round(v, 3) for d, v in m["per_dim_mae"].items()})
    return 0


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--manifest", required=True)
        p.add_argument("--root", default=".")
        p.add_argument("--crop", type=int, default=256, help=f"须为 {JPEG_GRID} 的倍数")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--val-frac", type=float, default=0.15)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--device", default=None)

    p = sub.add_parser("train"); common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-blockiness", action="store_true",
                   help="关闭块边界统计量(纪律 7),用于消融对照")

    p = sub.add_parser("eval"); common(p)
    p.add_argument("--ckpt", required=True)

    p = sub.add_parser("predict"); common(p)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--pred-out", required=True)

    args = ap.parse_args(argv)
    return {"train": run_train, "eval": run_eval, "predict": run_predict}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

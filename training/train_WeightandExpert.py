"""Stage 1 & 2 —— 三专家训练,然后权重 MLP 训练。

    # Stage 1:三专家,权重固定均匀(防饿死)
    python training/train_WeightandExpert.py stage1 --cache cache/train --out runs/squade

    # Stage 2:权重 MLP,专家冻结,温度 2.0 -> 1.0 退火
    python training/train_WeightandExpert.py stage2 --cache cache/train --out runs/squade \
           --pred-codes cache/pred_codes.csv

    # 评测:A2(均匀) / A3(学习权重) / A3-o(oracle 上界) 三行并排 + oracle gap
    python training/train_WeightandExpert.py eval --cache cache/train --out runs/squade \
           --pred-codes cache/pred_codes.csv

分两阶段而不是一次训完,是 README 的纪律:

* **Stage 1 权重固定均匀**  三个专家在同等曝光下各自学到能学的东西。若一开始就让权重
  可学,梯度会迅速把预算集中到最快下降的那个专家,其余两个饿死 —— 之后再谈"路由"就没有
  可路由的对象了。
* **Stage 2 专家冻结**  此时唯一可变的是 899 个权重 MLP 参数。A2 -> A3 那一格增益因此
  **只可能**来自退化路由,不可能来自专家自身变强。这是 oracle gap 归因成立的前提。
  冻结用 ExpertBank.freeze_experts(),它同时关梯度和 dropout —— 只关梯度的话专家输出
  带随机性,权重 MLP 学的是噪声上的平均。

核心指标不是原始 AUC,而是 **关闭 oracle gap 的百分比**:
    gap_closed = (A3 - A2) / (A3o - A2)
A3-o 用真值码字逐码字查最优权重给出理论上界,学习版逼近它的程度量化了"退化路由"这
一个自由度的真实贡献。
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import DEG_DIMS, N_LEVELS, load_cache          # noqa: E402
from models.experts_mlp import BANDS, ExpertBank, fit_normalization  # noqa: E402
from models.weights_mlp import TAU_END, TAU_START, WeightMLP        # noqa: E402


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """秩法 AUC,避免引入 sklearn 依赖。"""
    if labels.min() == labels.max():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列:同分取平均秩
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- 数据

class Bundle:
    """缓存 + 码字,已按训练/验证切好。特征保持 fp16 常驻内存,按 batch 转 fp32。"""

    def __init__(self, cache_dir: str, pred_codes: str | None, val_frac: float, oracle: bool):
        feats, prenorm, meta, cfg = load_cache(cache_dir)
        if feats.shape[1] != len(BANDS):
            raise SystemExit(
                f"缓存里有 {feats.shape[1]} 层,但专家组需要恰好 {len(BANDS)} 层"
                f"(浅/中/深)。请用 --layers 只缓存 probe_layers 选出的三层。")
        self.feats = torch.from_numpy(np.ascontiguousarray(feats).copy())       # (M,3,D) fp16
        self.prenorm = torch.from_numpy(np.ascontiguousarray(prenorm).copy())   # (M,3,2) fp32
        self.labels = torch.tensor([m["label"] for m in meta], dtype=torch.float32)
        self.cfg, self.meta = cfg, meta

        true_codes = torch.tensor([m["code"] for m in meta], dtype=torch.long)
        if oracle or pred_codes is None:
            if not oracle:
                print("[注意] 未提供 --pred-codes,退回真值码字。这会造成 train/test 失配,"
                      "A3 数字虚高 —— 仅供冒烟测试。")
            self.codes, self.soft = true_codes, true_codes.float()
        else:
            self.codes, self.soft = self._read_pred(pred_codes, len(meta), meta)
        self.true_codes = true_codes

        self.tr, self.va = self._split(meta, val_frac)

    @staticmethod
    def _read_pred(path: str, n: int, meta: list[dict]):
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        if len(rows) != n:
            raise SystemExit(f"{path} 有 {len(rows)} 行,缓存有 {n} 条,对不上")
        by_id = {r["image_id"]: r for r in rows}
        if all(m["image_id"] in by_id for m in meta):
            rows = [by_id[m["image_id"]] for m in meta]      # 按 image_id 对齐更稳
        codes = torch.tensor([[int(r[f"pred_{d}"]) for d in DEG_DIMS] for r in rows], dtype=torch.long)
        soft = torch.tensor([[float(r[f"soft_{d}"]) for d in DEG_DIMS] for r in rows], dtype=torch.float32)
        return codes, soft

    @staticmethod
    def _split(meta: list[dict], val_frac: float):
        if any(m.get("split") for m in meta):
            tr = [i for i, m in enumerate(meta) if m["split"].lower() not in ("val", "valid", "test")]
            va = [i for i, m in enumerate(meta) if m["split"].lower() in ("val", "valid", "test")]
            if tr and va:
                return torch.tensor(tr), torch.tensor(va)
        tr, va = [], []
        for i, m in enumerate(meta):
            h = int(hashlib.sha256(m["image_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            (va if h < val_frac else tr).append(i)
        return torch.tensor(tr), torch.tensor(va)

    def batches(self, idx: torch.Tensor, bs: int, shuffle: bool, device):
        order = idx[torch.randperm(len(idx))] if shuffle else idx
        for k in range(0, len(order), bs):
            j = order[k : k + bs]
            yield (self.feats[j].to(device, torch.float32),
                   self.prenorm[j].to(device, torch.float32),
                   self.codes[j].to(device), self.soft[j].to(device),
                   self.labels[j].to(device))


# --------------------------------------------------------------------------- 评估

@torch.no_grad()
def collect_logits(bank: ExpertBank, b: Bundle, idx: torch.Tensor, device, bs: int = 512):
    """-> (N,3) 各专家 logit。专家冻结后这个是常量,A2/A3/A3-o 都复用它,不必重跑。"""
    bank.eval()
    out = []
    for k in range(0, len(idx), bs):
        j = idx[k : k + bs]
        _, parts = bank(b.feats[j].to(device, torch.float32),
                        b.prenorm[j].to(device, torch.float32), return_parts=True)
        out.append(parts["expert_logits"].cpu())
    return torch.cat(out)


def auc_se(a: float, n_pos: int, n_neg: int) -> float:
    """AUC 的标准误(Hanley-McNeil)。用来判断 oracle gap 是否落在噪声里。"""
    if not (n_pos and n_neg) or not np.isfinite(a):
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return float(np.sqrt(max(v, 0.0)))


def score(z: torch.Tensor, y: torch.Tensor) -> dict:
    s, l = z.numpy(), y.numpy()
    return {"auc": auc(s, l), "acc": float(((s > 0) == (l > 0.5)).mean()),
            "bce": float(F.binary_cross_entropy_with_logits(z, y))}


def oracle_weights(zs: torch.Tensor, y: torch.Tensor, codes: torch.Tensor, grid: int = 10) -> dict:
    """逐码字在单纯形网格上搜最优权重(A3-o)。**只在训练集上拟合**,验证集上评估。

    这样得到的才是「理论上界」而非过拟合数字:若在验证集上直接搜,A3-o 会虚高,
    oracle gap 被撑大,你的 gap_closed 反而变小 —— 两头都不对。
    """
    simplex = torch.tensor([[i, j, grid - i - j] for i in range(grid + 1)
                            for j in range(grid + 1 - i)], dtype=torch.float32) / grid
    table = {}
    keys = [tuple(c.tolist()) for c in codes]
    for key in set(keys):
        sel = torch.tensor([i for i, k in enumerate(keys) if k == key])
        if len(sel) < 8:                     # 样本太少,搜出来的是噪声
            continue
        zz, yy = zs[sel], y[sel]
        losses = torch.stack([F.binary_cross_entropy_with_logits((simplex[t] * zz).sum(-1), yy)
                              for t in range(len(simplex))])
        table[key] = simplex[int(losses.argmin())]
    return table


def apply_table(table: dict, codes: torch.Tensor, n_experts: int = 3) -> torch.Tensor:
    w = torch.full((len(codes), n_experts), 1.0 / n_experts)
    for i, c in enumerate(codes):
        k = tuple(c.tolist())
        if k in table:
            w[i] = table[k]
    return w


# --------------------------------------------------------------------------- Stage 1

def run_stage1(args) -> int:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    b = Bundle(args.cache, None, args.val_frac, oracle=True)   # Stage 1 不用码字
    hidden_size = b.feats.shape[-1] // 3                        # cls|mean|std 三块

    bank = ExpertBank(hidden_size=hidden_size, hidden=args.hidden, dropout=args.dropout)
    if getattr(args, "init", None):
        ck0 = torch.load(args.init, map_location="cpu")
        bank.load_state_dict(ck0["bank"])
        print(f"[热启动] 专家权重载自 {args.init}")
    bank = bank.to(device)
    # 决策 2:标准化统计量**只在训练集上**标定,防泄漏。
    # 热启动时也必须重标定:统计量是数据相关的校准,不是学出来的参数,
    # 搬旧数据集的统计量到新数据上会严重失配。
    fit_normalization(bank, b.feats, b.prenorm, b.tr)
    bank.to(device)

    opt = torch.optim.AdamW(bank.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, (len(b.tr) + args.batch_size - 1) // args.batch_size) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("[Stage 1] 三专家训练,权重固定均匀(防饿死)")
    print(f"设备      : {device}   hidden_size={hidden_size}")
    print(f"可训练参数: {sum(p.numel() for p in bank.parameters() if p.requires_grad):,}")
    print(f"样本      : 训练 {len(b.tr)} / 验证 {len(b.va)}")

    best, hist = (-1.0, -1e9), []          # (AUC, -BCE):AUC 打平时用 BCE 决胜
    for ep in range(1, args.epochs + 1):
        bank.train(); t0 = time.time(); run = seen = 0
        for f, pn, _, _, y in b.batches(b.tr, args.batch_size, True, device):
            z = bank(f, pn, weights=None)              # weights=None -> 严格均匀
            loss = F.binary_cross_entropy_with_logits(z, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * len(y); seen += len(y)

        zs = collect_logits(bank, b, b.va, device)
        m = score(zs.mean(-1), b.labels[b.va])         # 均匀权重 = 三者平均
        per = {n: score(zs[:, i], b.labels[b.va])["auc"] for i, n in enumerate(BANDS)}
        hist.append({"epoch": ep, "train_loss": run / max(seen, 1), **m, "per_expert_auc": per})
        star = ""
        key = (m["auc"], -m["bce"])
        if key > best:
            best = key
            torch.save({"bank": bank.state_dict(),
                        "cfg": {"hidden_size": hidden_size, "hidden": args.hidden,
                                "dropout": args.dropout, "cache_cfg": b.cfg}}, out / "stage1.pt")
            star = " *"
        print(f"[{ep:>3}/{args.epochs}] train={run / max(seen, 1):.4f} "
              f"AUC={m['auc']:.4f} acc={m['acc']:.3f} | "
              + " ".join(f"{n}={per[n]:.3f}" for n in BANDS) + f" ({time.time() - t0:.0f}s){star}")

    (out / "stage1_history.json").write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n最佳 AUC {best[0]:.4f} (BCE {-best[1]:.4f}) -> {out / 'stage1.pt'}")
    print("三个专家的单独 AUC 若高度接近,说明它们在做同一件事;"
          "用 ExpertBank.collect_hidden() 做 t-SNE 可进一步确认。")
    return 0


# --------------------------------------------------------------------------- Stage 2

def run_stage2(args) -> int:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    b = Bundle(args.cache, args.pred_codes, args.val_frac, oracle=args.oracle)

    ck = torch.load(Path(args.out) / "stage1.pt", map_location="cpu")
    bank = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                      dropout=ck["cfg"]["dropout"])
    bank.load_state_dict(ck["bank"])
    bank.to(device).freeze_experts()          # 关梯度 **且** 关 dropout,两者缺一不可

    mlp = WeightMLP().to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.wd)
    total = max(1, (len(b.tr) + args.batch_size - 1) // args.batch_size) * args.epochs

    print("[Stage 2] 权重 MLP 训练,专家冻结,温度退火")
    print(f"设备      : {device}")
    print(f"可训练参数: {sum(p.numel() for p in mlp.parameters()):,}  "
          f"(专家侧 {sum(p.numel() for p in bank.parameters() if p.requires_grad)})")
    print(f"码字来源  : {'真值(oracle 模式)' if args.oracle else args.pred_codes}")
    print(f"温度      : {TAU_START} -> {TAU_END} ({args.tau_schedule})")

    step, best, hist = 0, (-1.0, -1e9), []   # 同上:AUC 打平时用 BCE 决胜
    for ep in range(1, args.epochs + 1):
        mlp.train(); t0 = time.time(); run = seen = 0
        for f, pn, code, soft, y in b.batches(b.tr, args.batch_size, True, device):
            tau = mlp.anneal(step / max(total - 1, 1), args.tau_schedule); step += 1
            w = mlp(code, soft)
            z = bank(f, pn, weights=w)
            loss = F.binary_cross_entropy_with_logits(z, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(y); seen += len(y)

        mlp.eval()
        zs = collect_logits(bank, b, b.va, device)
        with torch.no_grad():
            w = mlp(b.codes[b.va].to(device), b.soft[b.va].to(device)).cpu()
        m = score((w * zs).sum(-1), b.labels[b.va])
        wm = w.mean(0)
        hist.append({"epoch": ep, "train_loss": run / max(seen, 1), "tau": tau,
                     "mean_weights": wm.tolist(), **m})
        star = ""
        key = (m["auc"], -m["bce"])
        if key > best:
            best = key
            torch.save({"mlp": mlp.state_dict(), "stage1": str(Path(args.out) / "stage1.pt")},
                       Path(args.out) / "stage2.pt")
            star = " *"
        print(f"[{ep:>3}/{args.epochs}] train={run / max(seen, 1):.4f} AUC={m['auc']:.4f} "
              f"tau={tau:.2f} 平均权重=[{wm[0]:.2f} {wm[1]:.2f} {wm[2]:.2f}]"
              f" ({time.time() - t0:.0f}s){star}")

    (Path(args.out) / "stage2_history.json").write_text(
        json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n最佳 AUC {best[0]:.4f} (BCE {-best[1]:.4f}) -> {Path(args.out) / 'stage2.pt'}")
    # 诊断:权重是否真的是退化状态的函数。若各码字给出的权重几乎相同,
    # 那么「路由」只是个恒定的重加权,A2->A3 的增益与退化无关 —— 这一点从
    # 平均权重上看不出来,必须逐码字看。
    mlp.load_state_dict(torch.load(Path(args.out) / "stage2.pt", map_location="cpu")["mlp"])
    tbl = mlp.cpu().route_table()
    spread = max(max(abs(w[k] - v[0][k]) for w in v for k in range(3)) for v in tbl.values())
    print(f"\n路由表(固定其余维为 0,单独扫每一维):")
    for name, ws in tbl.items():
        print(f"  {name:<8} " + "  ".join(f"档{i}=[{w[0]:.2f} {w[1]:.2f} {w[2]:.2f}]" for i, w in enumerate(ws)))
    print(f"权重随码字的最大变动 = {spread:.4f}")
    if spread < 0.02:
        print("  ⚠️ 权重几乎不随退化码字变化 —— 名义上有路由,实质是恒定重加权。"
              "A2->A3 若有增益也与退化无关,不能作为 novelty 证据。")
    return 0


# --------------------------------------------------------------------------- 评测

def run_eval(args) -> int:
    device = pick_device(args.device)
    b = Bundle(args.cache, args.pred_codes, args.val_frac, oracle=False)
    ck = torch.load(Path(args.out) / "stage1.pt", map_location="cpu")
    bank = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                      dropout=ck["cfg"]["dropout"])
    bank.load_state_dict(ck["bank"]); bank.to(device).freeze_experts()

    z_tr = collect_logits(bank, b, b.tr, device)
    z_va = collect_logits(bank, b, b.va, device)
    y_va = b.labels[b.va]

    rows = [("A2  三专家 + 均匀权重", score(z_va.mean(-1), y_va))]

    s2 = Path(args.out) / "stage2.pt"
    if s2.exists():
        mlp = WeightMLP().to(device); mlp.load_state_dict(torch.load(s2, map_location="cpu")["mlp"])
        mlp.eval()
        with torch.no_grad():
            w = mlp(b.codes[b.va].to(device), b.soft[b.va].to(device)).cpu()
        rows.append(("A3  + 学习权重 MLP", score((w * z_va).sum(-1), y_va)))

    # A3-o:用**真值**码字在训练集上逐码字搜最优权重,验证集上评估
    table = oracle_weights(z_tr, b.labels[b.tr], b.true_codes[b.tr], args.oracle_grid)
    w_o = apply_table(table, b.true_codes[b.va])
    rows.append((f"A3-o + oracle 权重({len(table)} 个码字)", score((w_o * z_va).sum(-1), y_va)))

    print(f"验证集 {len(b.va)} 条\n")
    print(f"{'消融':<34}{'AUC':>9}{'acc':>8}{'BCE':>9}")
    print("-" * 60)
    for n, m in rows:
        print(f"{n:<34}{m['auc']:>9.4f}{m['acc']:>8.3f}{m['bce']:>9.4f}")

    a2 = rows[0][1]["auc"]; a3o = rows[-1][1]["auc"]
    if len(rows) == 3:
        a3 = rows[1][1]["auc"]
        gap = a3o - a2
        n_pos, n_neg = int(y_va.sum()), int((1 - y_va).sum())
        se = auc_se(a2, n_pos, n_neg)
        print(f"\noracle gap = A3o - A2 = {gap:+.4f}   (AUC 标准误 ±{se:.4f}, "
              f"n={len(y_va)} 正{n_pos}/负{n_neg})")

        # 守卫:gap 若小于 AUC 的标准误,这个比值就是在放大噪声。它可以轻易超过 100%
        # 或变成负数,那不是「路由超过了理论上界」,只是验证集太小。
        if not np.isfinite(se) or abs(gap) <= se:
            print("  ⚠️ oracle gap 未超出 AUC 的标准误 —— 下面的百分比是噪声的放大,不要报告。"
                  f"\n     要让这个指标可信,验证集大致需要 {int(4 * (se / max(abs(gap), 1e-6)) ** 2 * len(y_va))} 条以上。")
        if abs(gap) > 1e-6:
            pct = (a3 - a2) / gap * 100
            print(f"**关闭 oracle gap 的百分比 = (A3-A2)/(A3o-A2) = {pct:.1f}%**")
            if pct > 100 or pct < 0:
                print("  ⚠️ 百分比越界(<0 或 >100)。A3 不可能真的超过 oracle 上界 —— "
                      "oracle 权重在训练集拟合、验证集评估,样本不足时会被学习版偶然超过。"
                      "扩大验证集,或调大 --oracle-grid。")
            print("这才是核心指标 —— 它量化「退化路由」这一个自由度的真实贡献,而非原始 AUC。")
        else:
            print("gap 近似为 0:oracle 权重也没有超过均匀权重,说明在当前专家上"
                  "「按退化路由」这个自由度本身没有价值,需要先回头检查抽头层的选择。")

    # 分退化档报告(README 要求)
    print("\n按退化强度分档(真值码字之和):")
    sev = b.true_codes[b.va].sum(-1)
    for lo, hi, tag in [(0, 0, "干净      "), (1, 2, "轻度(1-2) "), (3, 4, "中度(3-4) "), (5, 99, "重度(5+)  ")]:
        sel = (sev >= lo) & (sev <= hi)
        if sel.sum() < 10:
            continue
        line = f"  {tag} n={int(sel.sum()):>5}"
        for n, _ in rows:
            zz = {"A2": z_va.mean(-1), "A3": (w * z_va).sum(-1) if len(rows) == 3 else None,
                  "A3-o": (w_o * z_va).sum(-1)}[n.split()[0]]
            line += f"   {n.split()[0]}={auc(zz[sel].numpy(), y_va[sel].numpy()):.4f}"
        print(line)
    return 0


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", required=True, help="cache_features.py 的输出目录(须恰好 3 层)")
        p.add_argument("--out", required=True, help="checkpoint 目录")
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--val-frac", type=float, default=0.15)
        p.add_argument("--device", default=None)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("stage1"); common(p)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--init", default=None,
                   help="热启动:从另一个 stage1.pt 载入专家权重再继续训。"
                        "**标准化统计量不搬** —— 那是数据相关的校准,换数据集必须重标定")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    p = sub.add_parser("stage2"); common(p)
    p.add_argument("--pred-codes", default=None, help="train_classifier.py predict 的输出")
    p.add_argument("--oracle", action="store_true", help="A3-o:用真值码字训(仅作上界参考)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--tau-schedule", default="cosine", choices=["cosine", "linear"])

    p = sub.add_parser("eval"); common(p)
    p.add_argument("--pred-codes", default=None)
    p.add_argument("--oracle-grid", type=int, default=10, help="单纯形网格分辨率")

    args = ap.parse_args(argv)
    return {"stage1": run_stage1, "stage2": run_stage2, "eval": run_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

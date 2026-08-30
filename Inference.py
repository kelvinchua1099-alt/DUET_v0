"""SQuaDE 推理 —— 门控两组专家,干净图在 L27 提前退出。

    python Inference.py --dir /path/to/images \
        --shallow runs/mix_shallow --deep runs/mix_deep --gate runs/mix_gate.pt \
        --out preds.csv

    # 单图
    python Inference.py --image a.jpg --shallow ... --deep ... --gate ...

Output CSV: image, verdict(FAKE/REAL), confidence(0-100), score, label(0/1),
            route(shallow/deep), depth_used, gate_logit
The verdict comes from a **logit-space** threshold (--threshold, default -8.7).
`confidence` is an uncalibrated 0-100 reliability index, NOT a probability:

    confidence = 100 * tanh(|z - threshold| / 8) * exp(-(max(e) - min(e)) / 60)

where z is the fused logit and e the three per-expert logits. It falls when the
decision sits near the threshold **or** when the three experts disagree — the
latter matters because the bank averages logits uniformly, so a single saturated
expert can outvote the other two while `score` still reads 1.000000.

---------------------------------------------------------------------------
两组专家 + 门

    输入 → DINOv2 ViT-g(冻结)
             ├─ 跑到 L27,取 L14/L21/L27 → 浅组三专家均匀平均 → z_shallow
             │                          → 门 g(读同样的浅层特征)
             │        g 判"干净" → 直接输出 z_shallow,**在这里停止**,省掉 L28~L37
             └─ g 判"退化" → 继续跑到 L37,取 L26/L33/L37 → 深组 → z_deep

门读的是**浅组的特征**,这一点是架构上的关键:它必须在提前退出点之前就能算出来。
门若改读深层特征,就必须先跑完全深度,提前退出的收益整个消失。

实测(28 万模型,官方 val):省 19.7% 前向深度,robust AUC 不降反升。
但在**原生低分辨率**来源上门会失效(把 82% 的图判成退化,省下的算力掉到 5.8%),
因为它学到的实际是"图糊不糊"而不是"有没有施加过退化算子"。

---------------------------------------------------------------------------
预处理必须与训练完全一致,否则等于给测试集加一层没记录的域偏移

    短边 >= 512   **中心裁 512x512,一次重采样都不做**
                  —— 保住原生高频(生成器指纹住在那里),也消除"缩放倍率"这条捷径
    短边 <  512   先裁成正方,再用**随机挑的一个内核**上采样到 512
                  内核池 [BILINEAR, BICUBIC, BOX, LANCZOS],按图确定性挑
                  —— 固定一个内核的话,"这张图带哪种插值痕迹"会变成一条与内容无关的线索

然后喂给 backbone 时 **crop 到 504**(DINOv2 patch=14 的整数倍),同样不 resize。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.dinov3 import DINOv3Backbone                        # noqa: E402
from models.experts_mlp import ExpertBank                       # noqa: E402
from utils.preprocess import INTERP_POOL, normalize, rng_for    # noqa: E402

CONF_MARGIN_SCALE = 8.0    # logit 单位;|margin| 到这个量级 confidence 才接近饱和
CONF_SPREAD_SCALE = 60.0   # 三专家 logit 极差到这个量级,置信度衰减到 1/e


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


def confidence(z: float, expert_logits, threshold: float) -> float:
    """0-100 可靠度指数(不是概率)。两个来源都会把它压低:
    离阈值近(margin 小)、三个专家意见分裂(极差大)。后者是关键 ——
    均匀 logit 平均下,单个饱和专家能压过另外两票,而 score 仍然显示 1.000000。"""
    margin = float(z) - float(threshold)
    spread = float(np.max(expert_logits) - np.min(expert_logits))
    return 100.0 * math.tanh(abs(margin) / CONF_MARGIN_SCALE) * math.exp(-spread / CONF_SPREAD_SCALE)


SHALLOW_LAYERS = [14, 21, 27]
DEEP_LAYERS = [26, 33, 37]
FIELDS = ["image", "verdict", "confidence", "score", "label", "route", "experts",
          "votes", "vote_spread", "depth_used", "gate_logit", "ms_per_img"]
EXIT_DEPTH, FULL_DEPTH = 27, 37        # 提前退出点 / 完整深度(总 40 个 block)


def pick_kernel(name: str):
    """按图名确定性地挑一个上采样内核 —— 与训练时 --interp random 同一套逻辑。"""
    r = rng_for(0, name + "|interp")
    return INTERP_POOL[int(r.integers(len(INTERP_POOL)))]


def load_image(p: Path, size: int = 512) -> Image.Image:
    """短边 >=size 中心裁不重采样;<size 裁正方后随机内核上采样。"""
    with Image.open(p) as im:
        im = im.convert("RGB")
        kernel = pick_kernel(p.name) if min(im.size) < size else None
        return normalize(im, size, kernel, fit="crop")


class SQuaDE:
    """门控两组专家 + 真正的提前退出。"""

    def __init__(self, shallow_run, deep_run, gate_path, model, crop=504, device="cuda",
                 allow_mismatch=False):
        self.dev, self.crop = device, crop
        self.bb = DINOv3Backbone(name=model, device=device)
        self._cache_cfgs = []
        self.shallow = self._bank(shallow_run)
        self.deep = self._bank(deep_run)
        self._check_runtime(allow_mismatch)
        g = torch.load(gate_path, map_location="cpu")
        self.gate = nn.Sequential(nn.Linear(g["D"], g["hidden"]), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(g["hidden"], 1))
        self.gate.load_state_dict(g["net"])
        self.gate.to(device).eval()
        self.g_mu, self.g_sd = g["mu"].to(device), g["sd"].to(device)
        self.prep = self.bb.make_preprocessor(crop_size=crop)

    def _bank(self, run):
        ck = torch.load(Path(run) / "stage1.pt", map_location="cpu")
        b = ExpertBank(hidden_size=ck["cfg"]["hidden_size"], hidden=ck["cfg"]["hidden"],
                       dropout=ck["cfg"]["dropout"])
        b.load_state_dict(ck["bank"])
        b.to(self.dev).freeze_experts()
        # cache_cfg 记录了特征缓存是用什么骨干/精度/窗口算出来的 —— 专家的权重与
        # 标准化 buffer 都在那个分布上标定,运行时对不上就是分布外输入。
        self._cache_cfgs.append((str(run), ck["cfg"].get("cache_cfg") or {}))
        return b

    def _check_runtime(self, allow_mismatch=False):
        """比对 checkpoint 记录的缓存配置与当前运行时。不一致会静默给出错误结论:
        实测同一张图 bf16 融合 logit -0.10(FAKE)、fp32 -29.53(REAL),不报任何错。"""
        want = {"dtype": str(self.bb.dtype), "backbone": self.bb.name,
                "crop_size": self.crop}
        bad = []
        for run, cfg in self._cache_cfgs:
            for k, got in want.items():
                exp = cfg.get(k)
                if exp is not None and exp != got:
                    bad.append(f"    {run}\n      {k:<9} trained on {exp}   running {got}")
        if not bad:
            return
        msg = ("checkpoint/runtime mismatch -- results would be silently invalid\n"
               + "\n".join(bad) + "\n\n"
               "  The expert heads and their standardisation buffers were fitted on cached\n"
               "  features produced with the values on the left. Feeding them anything else\n"
               "  is out-of-distribution input and the verdict can flip with no error raised.\n"
               "  On Apple silicon pass --device mps: CPU forces the backbone to fp32.\n"
               "  Pass --allow-mismatch to run anyway.")
        if allow_mismatch:
            print(f"[WARNING] {msg}\n", flush=True)
        else:
            raise RuntimeError(msg)

    @staticmethod
    def _pack(feats, layers, dev):
        """{layer: entry} -> (B,3,3H) 与 (B,3,2),顺序按 layers 给定。"""
        f = torch.stack([DINOv3Backbone.pool(feats[l]) for l in layers], 1).to(dev, torch.float32)
        p = torch.stack([feats[l]["prenorm_stats"] for l in layers], 1).to(dev, torch.float32)
        return f, p

    @torch.no_grad()
    def predict(self, imgs, names):
        # 预处理器每张返回 (1,3,H,W),要 cat 不是 stack —— stack 会多出一维
        x = torch.cat([self.prep(im, image_id=n) for im, n in zip(imgs, names)])

        # ① 只跑到 L27。门与浅组都读这一段的特征,所以判"干净"时后面 13 个 block 根本不跑。
        f1, hid = self.bb.forward_blocks(x, layers=SHALLOW_LAYERS + [DEEP_LAYERS[0]],
                                         max_block=EXIT_DEPTH)
        fs, ps = self._pack(f1, SHALLOW_LAYERS, self.dev)
        zs, ps_parts = self.shallow(fs, ps, return_parts=True)
        votes = ps_parts["expert_logits"].clone()          # (B,3) 组内三个专家各自的 logit
        gz = self.gate((fs.reshape(len(imgs), -1) - self.g_mu) / self.g_sd).squeeze(-1)
        clean = gz <= 0

        z = zs.clone()
        n_deep = int((~clean).sum())
        if n_deep:
            # ② 只有被判为退化的图才继续跑 L28~L37,而且从 L27 的 hidden 接着跑,不重跑前面
            sel = (~clean).nonzero(as_tuple=True)[0]
            f2, _ = self.bb.forward_blocks(None, layers=DEEP_LAYERS[1:], max_block=FULL_DEPTH,
                                           resume=hid[sel], start_block=EXIT_DEPTH)
            f2[DEEP_LAYERS[0]] = {k: (v[sel] if torch.is_tensor(v) else v)
                                  for k, v in f1[DEEP_LAYERS[0]].items()}
            fd, pd = self._pack(f2, DEEP_LAYERS, self.dev)
            zd, pd_parts = self.deep(fd, pd, return_parts=True)
            z[sel] = zd
            votes[sel] = pd_parts["expert_logits"]
        # 一律返回裸 logit:阈值与 confidence 都必须在 logit 空间算,
        # sigmoid 在两端饱和,过早转换会把判据本身丢掉。
        return (z.cpu().numpy(), clean.cpu().numpy(),
                gz.cpu().numpy(), votes.cpu().numpy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="image directory (searched recursively)")
    src.add_argument("--image", help="a single image")
    ap.add_argument("--shallow", required=True, help="shallow-bank run dir (must contain stage1.pt)")
    ap.add_argument("--deep", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--model", default="facebook/dinov2-giant")
    ap.add_argument("--out", default=None, help="write CSV here; prints to stdout if omitted")
    ap.add_argument("--resume", action="store_true",
                    help="skip images already present in --out and append to it. "
                         "Makes a killed run restartable with the same command")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process only shard I of N (0-based), e.g. 0/4. "
                         "Run N processes with different I to split a corpus")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="run even if the checkpoint's cached-feature config "
                         "(backbone/dtype/crop_size) differs from the runtime. "
                         "Downgrades the error to a warning; results may be invalid")
    ap.add_argument("--threshold", type=float, default=-8.7,
                    help="decision threshold in **logit space** "
                         "(fused logit > threshold -> FAKE). Default -8.7; "
                         "pass 0.0 for the original score>0.5 behaviour")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop-size", type=int, default=504)
    ap.add_argument("--no-early-exit", action="store_true",
                    help="disable early exit: run both banks, then pick by the gate. "
                         "Same accuracy, no compute saved; use it to verify that "
                         "early exit does not change results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    paths = ([Path(a.image)] if a.image else
             sorted(p for p in Path(a.dir).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    if not paths:
        raise SystemExit("no images found")
    total_found = len(paths)

    if a.shard:                                    # 分片必须在过滤之前,且 paths 已排序
        i, nsh = (int(x) for x in a.shard.split("/"))
        if not 0 <= i < nsh:
            raise SystemExit(f"--shard {a.shard}: need 0 <= I < N")
        paths = paths[i::nsh]
        print(f"shard {i}/{nsh}: {len(paths)} of {total_found} images", flush=True)

    done = set()
    if a.resume and a.out and Path(a.out).exists():
        with open(a.out, newline="", encoding="utf-8") as fh:
            done = {r["image"] for r in csv.DictReader(fh)}
        paths = [p for p in paths if str(p) not in done]
        print(f"resume: {len(done)} row(s) already in {a.out}", flush=True)
    if not paths:
        print("nothing left to do")
        return 0
    print(f"{len(paths)} image(s)   shallow bank {SHALLOW_LAYERS}   "
          f"deep bank {DEEP_LAYERS}", flush=True)

    net = SQuaDE(a.shallow, a.deep, a.gate, a.model, a.crop_size, a.device,
                 allow_mismatch=a.allow_mismatch)

    # 流式写盘:大规模跑时中途被杀也不会丢已算完的部分。追加模式仅在续传时使用。
    fh = writer = None
    if a.out:
        append = bool(done)
        fh = open(a.out, "a" if append else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not append:
            writer.writeheader()

    import time
    # 只累计汇总所需的量,不把全部行留在内存 —— 百万张时 rows 会吃掉几百 MB
    head, confs, n_exit, n_fake, n_done, t_total = [], [], 0, 0, 0, 0.0
    failed = []
    t_start = time.perf_counter()
    for k in range(0, len(paths), a.batch_size):
        raw = paths[k : k + a.batch_size]
        imgs, chunk = [], []
        for p in raw:                              # 逐图容错:坏文件跳过,不拖垮整跑
            try:
                imgs.append(load_image(p)); chunk.append(p)
            except Exception as e:
                failed.append((p, f"{type(e).__name__}: {e}"))
                print(f"  [skip] {p}: {type(e).__name__}: {e}", flush=True)
        if not imgs:
            continue
        t0 = time.perf_counter()
        zlog, cl, gz, vlog = net.predict(imgs, [p.name for p in chunk])
        dt = time.perf_counter() - t0
        t_total += dt
        n_exit += int(cl.sum())
        for q, zl, c, g, v in zip(chunk, zlog, cl, gz, vlog):
            grp = SHALLOW_LAYERS if c else DEEP_LAYERS
            is_fake = bool(zl > a.threshold)
            vp = _sigmoid(v)                       # 仅用于展示的每专家概率
            row = ({"image": str(q),
                         "verdict": "FAKE" if is_fake else "REAL",
                         "confidence": f"{confidence(zl, v, a.threshold):.1f}",
                         "score": f"{float(_sigmoid(zl)):.6f}",
                         "label": int(is_fake),
                         "route": "shallow" if c else "deep",
                         "experts": "/".join(f"L{l}" for l in grp),
                         # 每个专家各自的概率 —— 三票分歧会把 confidence 拉下来
                         "votes": "|".join(f"{x:.3f}" for x in vp),
                         "vote_spread": f"{float(vp.max() - vp.min()):.3f}",
                         "depth_used": EXIT_DEPTH if c else FULL_DEPTH,
                         "gate_logit": f"{g:.4f}",
                         "ms_per_img": f"{dt / len(chunk) * 1000:.1f}"})
            n_done += 1
            n_fake += int(is_fake)
            confs.append(float(row["confidence"]))
            if len(head) < 20:
                head.append(row)
            if writer:
                writer.writerow(row)
        if writer:
            fh.flush()                             # 每批落盘,被杀也只丢当前批
        if (k // a.batch_size) % 20 == 0:
            seen = min(k + a.batch_size, len(paths))
            el = time.perf_counter() - t_start
            rate = seen / el if el > 0 else 0.0
            eta = (len(paths) - seen) / rate if rate > 0 else 0.0
            print(f"  {seen}/{len(paths)}  {rate:.2f} img/s  "
                  f"ETA {eta / 60:.1f} min", flush=True)

    import statistics as _st
    if not n_done:
        print(f"\nno image could be read ({len(failed)} failed)")
        if fh: fh.close()
        return 1
    saved = n_exit / n_done * (1 - EXIT_DEPTH / 40)
    print(f"\nDecision threshold: fused logit > {a.threshold:+g} (logit space)")
    print(f"Routed shallow (early exit taken): {n_exit}/{n_done} = "
          f"{n_exit / n_done * 100:.1f}%")
    print(f"Forward depth saved by early exit: {saved * 100:.1f}%  "
          f"(clean images stop at L{EXIT_DEPTH}, skipping "
          f"L{EXIT_DEPTH + 1}-L{FULL_DEPTH}, 13/40 blocks)")
    print(f"Verdict: FAKE {n_fake}/{n_done}   REAL {n_done - n_fake}/{n_done}")
    print(f"\nTime: {t_total:.2f} s total, {t_total / n_done * 1000:.1f} ms/image, "
          f"{n_done / t_total:.1f} images/s")
    print(f"Confidence: median {_st.median(confs):.1f}  min {min(confs):.1f}  "
          f"max {max(confs):.1f}")
    print("  Low confidence means a small margin to the threshold and/or the three "
          "experts disagree.")
    print("  Those are the samples worth reviewing by hand -- note that `score` "
          "saturates to 1.000000")
    print("  in exactly those cases, so it cannot be used to spot them.")

    if failed:
        print(f"\nSkipped {len(failed)} unreadable image(s)")
        if a.out:
            fpath = Path(str(a.out) + ".failed")
            with open(fpath, "a", encoding="utf-8") as ffh:
                for q, err in failed:
                    ffh.write(f"{q}\t{err}\n")
            print(f"  -> {fpath}")
        else:
            for q, err in failed[:10]:
                print(f"  {q}: {err}")

    if a.out:
        fh.close()
        print(f"-> {a.out}  ({n_done} row(s) written this run)")
    else:
        print(f"\n  {'image':<34} {'verdict':<8} {'conf':>5}  {'route':<8} "
              f"{'experts':<14} votes")
        for r in head:
            print(f"  {Path(r['image']).name:<34} {r['verdict']:<8} "
                  f"{r['confidence']:>5}  {r['route']:<8} {r['experts']:<14} {r['votes']}")
        if n_done > len(head):
            print(f"  ... {n_done} rows total; use --out to write a CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

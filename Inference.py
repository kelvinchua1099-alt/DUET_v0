"""SQuaDE 推理 —— 门控两组专家,干净图在 L27 提前退出。
SQuaDE inference -- two gated expert banks with early exit at L27 for clean inputs.

    python Inference.py --dir /path/to/images \
        --shallow runs/mix_shallow --deep runs/mix_deep --gate runs/mix_gate.pt \
        --json preds.json --out preds.csv

    # 单图 / Single image
    python Inference.py --image a.jpg --shallow ... --deep ... --gate ...

--json 是**提交格式**:[{"image_path": str, "pred": float in [0,1]}, ...],pred 越大越像
AI 生成。它不是 sigmoid(z) 而是以阈值居中、除以温度的 sigmoid —— 理由见 pred_score 的
docstring(一句话:裸 sigmoid 在 |z|>40 上并列成 1.000000,把排序信息丢光,AUC 白掉)。
--json writes the submission format: a list of {"image_path", "pred"}, pred in [0,1] and
higher means more likely AI-generated. See pred_score for why it is not plain sigmoid(z).

Output CSV: image, pred, verdict(FAKE/REAL), confidence(0-100), score, label(0/1),
            route(shallow/deep), depth_used, gate_logit
The verdict comes from a **logit-space** threshold (--threshold, default -8.7).
`confidence` is an uncalibrated 0-100 reliability index, NOT a probability:

    confidence = 100 * tanh(|z - threshold| / 8) * exp(-(max(e) - min(e)) / 60)

where z is the fused logit and e the three per-expert logits. It falls when the
decision sits near the threshold **or** when the three experts disagree — the
latter matters because the bank averages logits uniformly, so a single saturated
expert can outvote the other two while `score` still reads 1.000000.

---------------------------------------------------------------------------
两组专家 + 门 / Two expert banks + gate

    输入 → DINOv2 ViT-g(冻结)
             ├─ 跑到 L27,取 L14/L21/L27 → 浅组三专家均匀平均 → z_shallow
             │                          → 门 g(读同样的浅层特征)
             │        g 判"干净" → 直接输出 z_shallow,**在这里停止**,省掉 L28~L37
             └─ g 判"退化" → 继续跑到 L37,取 L26/L33/L37 → 深组 → z_deep

    Input → frozen DINOv2 ViT-g
             ├─ run to L27; tap L14/L21/L27 → shallow experts → z_shallow
             │                                → gate g from the same shallow features
             │        g predicts "clean" → return z_shallow and stop at L27
             └─ g predicts "degraded" → resume to L37; tap L26/L33/L37 → z_deep

门读的是**浅组的特征**,这一点是架构上的关键:它必须在提前退出点之前就能算出来。
门若改读深层特征,就必须先跑完全深度,提前退出的收益整个消失。
The gate must read shallow features so it is available before the exit point. A gate that
depends on deep features would require full-depth execution and eliminate the saving.

早退样本相对 L37 路线少跑 10 个 block;程序会按实际路由率报告平均节省。
但在**原生低分辨率**来源上门会失效(把 82% 的图判成退化),
因为它学到的实际是"图糊不糊"而不是"有没有施加过退化算子"。
Each early-exited sample skips 10 blocks relative to the L37 route; the runtime summary
reports the average saving from the observed routing rate. The gate is less reliable on
native low-resolution sources because it can learn blur rather than applied degradation.

---------------------------------------------------------------------------
预处理必须与训练完全一致,否则等于给测试集加一层没记录的域偏移。
Preprocessing must exactly match training; otherwise inference introduces an undocumented
domain shift.

    短边 >= 512   **中心裁 512x512,一次重采样都不做**
                  —— 保住原生高频(生成器指纹住在那里),也消除"缩放倍率"这条捷径
    短边 <  512   先裁成正方,再用**随机挑的一个内核**上采样到 512
                  内核池 [BILINEAR, BICUBIC, BOX, LANCZOS],按图确定性挑
                  —— 固定一个内核的话,"这张图带哪种插值痕迹"会变成一条与内容无关的线索

    short side >= 512  center-crop to 512x512 without resampling
    short side <  512  square-crop, then upsample to 512 using a deterministic per-image
                       choice from [BILINEAR, BICUBIC, BOX, LANCZOS]

然后喂给 backbone 时 **crop 到 504**(DINOv2 patch=14 的整数倍),同样不 resize。
Before the backbone, crop to 504 (a multiple of DINOv2 patch size 14) without resizing.
"""
from __future__ import annotations

import argparse
import csv
import json
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

CONF_MARGIN_SCALE = 8.0    # logit 单位 / logit units; confidence nears saturation here
CONF_SPREAD_SCALE = 60.0   # 三专家极差 / expert-logit range giving 1/e decay
PRED_TEMPERATURE = 20.0    # `pred` 的温度 / temperature for `pred`; see pred_score


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


def confidence(z: float, expert_logits, threshold: float) -> float:
    """0-100 可靠度指数(不是概率)。两个来源都会把它压低:
    离阈值近(margin 小)、三个专家意见分裂(极差大)。后者是关键 ——
    均匀 logit 平均下,单个饱和专家能压过另外两票,而 score 仍然显示 1.000000。

    Uncalibrated 0-100 reliability index, not a probability. It decreases near the
    decision threshold or when the three expert logits disagree."""
    margin = float(z) - float(threshold)
    spread = float(np.max(expert_logits) - np.min(expert_logits))
    return 100.0 * math.tanh(abs(margin) / CONF_MARGIN_SCALE) * math.exp(-spread / CONF_SPREAD_SCALE)


def pred_score(z: float, threshold: float) -> float:
    """0-1 的「这张图是 AI 生成的」分数 —— 提交用的那一列。
    0-1 likelihood that the image is AI-generated -- the submission column.

    不是 sigmoid(z),而是 **以阈值为中心、加温度** 的 sigmoid:
    Not sigmoid(z), but a sigmoid centred on the threshold and divided by a temperature:

        pred = sigmoid((z - threshold) / PRED_TEMPERATURE)

    两个理由,都只关乎可用性,不改变判决:
      * 单调于 z,所以 AUC 与直接用 z 排序完全相同,且 pred > 0.5 恰好等价于
        verdict == FAKE(裸 sigmoid 的 0.5 对应 z=0,与 -8.7 的阈值对不上);
      * 裸 sigmoid(z) 在 |z| > 40 时数值上饱和成 0.0 / 1.0 —— 本模型的 logit
        常到 ±100,于是绝大多数行并列在 1.000000,排序信息全丢,AUC 被并列拖垮。
        除以温度把它拉回可分辨的区间。

    Both reasons concern usability only; neither changes the verdict. It is monotone in z,
    so AUC matches ranking by the raw logit, and pred > 0.5 is exactly verdict == FAKE.
    Plain sigmoid(z) saturates numerically to 0.0/1.0 beyond |z| > 40, and this model's
    logits routinely reach +/-100, so most rows would tie at 1.000000 and the ranking
    information -- hence the AUC -- would be lost.
    """
    return float(_sigmoid((float(z) - float(threshold)) / PRED_TEMPERATURE))


SHALLOW_LAYERS = [14, 21, 27]
DEEP_LAYERS = [26, 33, 37]
FIELDS = ["image", "pred", "verdict", "confidence", "score", "label", "route", "experts",
          "votes", "vote_spread", "depth_used", "gate_logit", "ms_per_img"]
EXIT_DEPTH, FULL_DEPTH = 27, 37        # 早退/部署深度 / exit/deployed full depth


def default_device() -> str:
    """自动选择运行设备 / Select the best available runtime device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_kernel(name: str):
    """按图名确定性地挑一个上采样内核;
    deterministically select the same per-image kernel policy used in training."""
    r = rng_for(0, name + "|interp")
    return INTERP_POOL[int(r.integers(len(INTERP_POOL)))]


def load_image(p: Path, size: int = 512) -> Image.Image:
    """短边 >=size 中心裁不重采样;<size 裁正方后按图选内核上采样。
    Center-crop large images without resampling; square-crop and upsample small ones."""
    with Image.open(p) as im:
        im = im.convert("RGB")
        kernel = pick_kernel(p.name) if min(im.size) < size else None
        return normalize(im, size, kernel, fit="crop")


class SQuaDE:
    """门控两组专家 + 真正的提前退出。
    Two gated expert banks with true staged early-exit execution."""

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
        # cache_cfg 记录骨干/精度/窗口;对不上会给专家喂入分布外特征。
        # cache_cfg records backbone/precision/window; a mismatch feeds OOD features to heads.
        self._cache_cfgs.append((str(run), ck["cfg"].get("cache_cfg") or {}))
        return b

    def _check_runtime(self, allow_mismatch=False):
        """比对 checkpoint 记录的缓存配置与当前运行时。不一致会静默给出错误结论:
        实测同一张图 bf16 融合 logit -0.10(FAKE)、fp32 -29.53(REAL),不报任何错。

        Compare checkpoint cache metadata with runtime settings; a mismatch can silently
        flip predictions because the expert heads receive out-of-distribution features."""
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
        """{layer: entry} -> (B,3,3H) 与 (B,3,2),按 layers 排序。
        Pack layer entries in the requested order."""
        f = torch.stack([DINOv3Backbone.pool(feats[l]) for l in layers], 1).to(dev, torch.float32)
        p = torch.stack([feats[l]["prenorm_stats"] for l in layers], 1).to(dev, torch.float32)
        return f, p

    @torch.no_grad()
    def predict(self, imgs, names, no_early_exit=False):
        # 预处理器每张返回 (1,3,H,W),要 cat 不是 stack —— stack 会多出一维。
        # Each preprocessed image has shape (1,3,H,W); concatenate rather than stack it.
        x = torch.cat([self.prep(im, image_id=n) for im, n in zip(imgs, names)])

        # ① 只跑到 L27。门与浅组都读这一段的特征,所以判"干净"时可在这里停止。
        # ① Run only to L27. Both the gate and shallow bank use these features, so clean
        # samples can stop here.
        f1, hid = self.bb.forward_blocks(x, layers=SHALLOW_LAYERS + [DEEP_LAYERS[0]],
                                         max_block=EXIT_DEPTH)
        fs, ps = self._pack(f1, SHALLOW_LAYERS, self.dev)
        zs, ps_parts = self.shallow(fs, ps, return_parts=True)
        # (B,3) 组内三个专家各自的 logit / logits from the three experts.
        votes = ps_parts["expert_logits"].clone()
        gz = self.gate((fs.reshape(len(imgs), -1) - self.g_mu) / self.g_sd).squeeze(-1)
        clean = gz <= 0

        z = zs.clone()
        deep_sel = (~clean).nonzero(as_tuple=True)[0]

        # 默认只让 deep 路由样本继续运行;验证模式下让整个 batch 都运行到 L37。
        # By default only deep-routed samples continue. Verification mode runs every sample
        # to L37 while preserving the gate-selected final output.
        run_sel = (torch.arange(len(imgs), device=clean.device)
                   if no_early_exit else deep_sel)

        if len(run_sel):
            # ② 从 L27 的 hidden state 继续跑 L28~L37,不重复计算前 27 层。
            # ② Resume from the L27 hidden state and run L28-L37 without recomputing L1-L27.
            f2, _ = self.bb.forward_blocks(None, layers=DEEP_LAYERS[1:], max_block=FULL_DEPTH,
                                           resume=hid[run_sel], start_block=EXIT_DEPTH)
            f2[DEEP_LAYERS[0]] = {k: (v[run_sel] if torch.is_tensor(v) else v)
                                  for k, v in f1[DEEP_LAYERS[0]].items()}
            fd, pd = self._pack(f2, DEEP_LAYERS, self.dev)
            zd, pd_parts = self.deep(fd, pd, return_parts=True)

            # zd 使用 run_sel 的局部行号;只有真正路由到 deep 的样本采用 deep 输出。
            # zd uses run_sel-local indices; only genuinely deep-routed samples adopt it.
            use_local = (~clean[run_sel]).nonzero(as_tuple=True)[0]
            use_global = run_sel[use_local]
            z[use_global] = zd[use_local]
            votes[use_global] = pd_parts["expert_logits"][use_local]

        # 一律返回裸 logit:阈值与 confidence 都必须在 logit 空间计算。
        # Return raw logits because thresholding and confidence must stay in logit space;
        # sigmoid saturation would otherwise discard the decision margin.
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
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write the submission JSON here: a list of "
                         '{\"image_path\": str, \"pred\": float in [0,1]}. '
                         "Written once at the end of the run; with --resume it is "
                         "rebuilt from the whole --out CSV, not just this run's rows")
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
    ap.add_argument(
        "--device",
        default=default_device(),
        help="runtime device: cuda, mps, or cpu; defaults to the best available device",
    )
    a = ap.parse_args(argv)

    paths = ([Path(a.image)] if a.image else
             sorted(p for p in Path(a.dir).rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")))
    if not paths:
        raise SystemExit("no images found")
    total_found = len(paths)

    # 分片必须在过滤之前,且 paths 已排序 / Shard the sorted list before filtering.
    if a.shard:
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

    # 流式写盘以保留已完成批次;只有续传时才追加。
    # Stream completed batches to disk; append only when resuming an existing output.
    fh = writer = None
    if a.out:
        append = bool(done)
        fh = open(a.out, "a" if append else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not append:
            writer.writeheader()

    import time
    # 只累计汇总所需的量,避免百万行结果常驻内存。
    # Retain only summary state instead of keeping millions of output rows in memory.
    head, confs, n_exit, n_fake, n_done, t_total = [], [], 0, 0, 0, 0.0
    json_rows = []                                 # 仅 --json 且没有 --out 时才用得上
                                                   # only used when --json without --out
    failed = []
    t_start = time.perf_counter()
    for k in range(0, len(paths), a.batch_size):
        raw = paths[k : k + a.batch_size]
        imgs, chunk = [], []
        # 逐图容错:坏文件跳过,不拖垮整跑 / Skip unreadable files individually.
        for p in raw:
            try:
                imgs.append(load_image(p)); chunk.append(p)
            except Exception as e:
                failed.append((p, f"{type(e).__name__}: {e}"))
                print(f"  [skip] {p}: {type(e).__name__}: {e}", flush=True)
        if not imgs:
            continue
        t0 = time.perf_counter()
        zlog, cl, gz, vlog = net.predict(
            imgs, [p.name for p in chunk], no_early_exit=a.no_early_exit
        )
        dt = time.perf_counter() - t0
        t_total += dt
        n_exit += int(cl.sum())
        for q, zl, c, g, v in zip(chunk, zlog, cl, gz, vlog):
            grp = SHALLOW_LAYERS if c else DEEP_LAYERS
            is_fake = bool(zl > a.threshold)
            vp = _sigmoid(v)  # 仅用于展示的专家概率 / display-only expert probabilities
            row = ({"image": str(q),
                         # 提交列:阈值居中 + 温度的概率,单调于 z,不并列
                         # submission column: threshold-centred, tempered, no ties
                         "pred": f"{pred_score(zl, a.threshold):.6f}",
                         "verdict": "FAKE" if is_fake else "REAL",
                         "confidence": f"{confidence(zl, v, a.threshold):.1f}",
                         "score": f"{float(_sigmoid(zl)):.6f}",
                         "label": int(is_fake),
                         "route": "shallow" if c else "deep",
                         "experts": "/".join(f"L{l}" for l in grp),
                         # 三票分歧会降低 confidence / expert disagreement lowers confidence.
                         "votes": "|".join(f"{x:.3f}" for x in vp),
                         "vote_spread": f"{float(vp.max() - vp.min()):.3f}",
                         "depth_used": (FULL_DEPTH if a.no_early_exit else
                                        (EXIT_DEPTH if c else FULL_DEPTH)),
                         "gate_logit": f"{g:.4f}",
                         "ms_per_img": f"{dt / len(chunk) * 1000:.1f}"})
            n_done += 1
            n_fake += int(is_fake)
            confs.append(float(row["confidence"]))
            if len(head) < 20:
                head.append(row)
            # 有 CSV 时统一从 CSV 重建,避免两份真相 / rebuild from the CSV when there is one
            if a.json and not a.out:
                json_rows.append({"image_path": row["image"],
                                  "pred": float(row["pred"])})
            if writer:
                writer.writerow(row)
        if writer:
            fh.flush()  # 每批落盘,中断时只丢当前批 / flush every batch
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
    # 相对于部署时完整的 L37 路线计算额外早退节省。
    # Report incremental early-exit savings relative to the deployed full L37 route.
    saved = (0.0 if a.no_early_exit else
             n_exit / n_done * (FULL_DEPTH - EXIT_DEPTH) / FULL_DEPTH)
    print(f"\nDecision threshold: fused logit > {a.threshold:+g} (logit space)")
    print(f"Gate selected shallow route: {n_exit}/{n_done} = "
          f"{n_exit / n_done * 100:.1f}%")
    if a.no_early_exit:
        print(f"Forward depth saved by early exit: 0.0%  "
              f"(--no-early-exit: every image ran to L{FULL_DEPTH})")
    else:
        print(f"Forward depth saved by early exit: {saved * 100:.1f}%  "
              f"(shallow-routed images stop at L{EXIT_DEPTH} instead of L{FULL_DEPTH}, "
              f"saving {FULL_DEPTH - EXIT_DEPTH}/{FULL_DEPTH} blocks on those images)")
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

    if a.json:
        # 从 CSV 重建(若有):这样 --resume / 崩溃续跑之后,JSON 覆盖的是全部行,
        # 而不只是本次进程算出来的那部分。
        # Rebuilding from the CSV keeps the JSON complete across --resume and crashes.
        if a.out and Path(a.out).exists():
            with open(a.out, newline="", encoding="utf-8") as jfh:
                # `pred` 是后加的列;续跑到一份旧 CSV 上时退回 score,别整跑崩掉
                # `pred` is a newer column; fall back to `score` on a pre-existing CSV
                json_rows = [{"image_path": r["image"],
                              "pred": float(r.get("pred") or r["score"])}
                             for r in csv.DictReader(jfh)]
        # 读不出来的文件也要占一行,否则 JSON 的条数与图片目录对不上。0.5 = 不表态。
        # Unreadable files still get a row (0.5 = abstain) so the count matches the directory.
        json_rows += [{"image_path": str(q), "pred": 0.5} for q, _ in failed]
        with open(a.json, "w", encoding="utf-8") as jfh:
            json.dump(json_rows, jfh, indent=2, ensure_ascii=False)
        print(f"-> {a.json}  ({len(json_rows)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

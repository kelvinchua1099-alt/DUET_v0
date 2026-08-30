# DUET — Depth-Uniform Ensemble with early Termination

AI-generated image detection that stays reliable after images are compressed, cropped,
blurred, or resized — and that skips 10 of the 37 transformer blocks it would otherwise
run, whenever it doesn't need them.

A frozen **DINOv2 ViT-g/14** (1.1B parameters) plus **7M trainable parameters**: six MLP
expert heads reading six intermediate depths, and one binary gate.

```
input → DINOv2 ViT-g (frozen, 40 blocks)
         │
         ├─ run to L27, tap L14/L21/L27 → shallow bank, uniform mean in logit space
         │                              → gate (reads the same shallow features)
         │       "clean" → output here; L28–L37 never run
         │
         └─ "degraded" → resume to L37, tap L26/L33/L37 → deep bank
```

---

## Results

All numbers are **held-out**. 70% of the official val set and 70% of val_hard were used in
training; the split files ship with the weights (`v3/val_split_70_30.json`,
`v3/valhard_split_70_30.json`) so the split can be verified independently.

| Data | Scheme | Overall | Clean | Degraded |
|---|---|---|---|---|
| Official val held-out (3,000)<br>±SE 0.23 | shallow-only | **0.9922** | **0.9978** | **0.9833** |
| | gated route | 0.9911 | 0.9977 | 0.9809 |
| Official val_hard held-out (750)<br>±SE 1.8 | shallow-only | 0.9430 | **0.9904** | 0.8674 |
| | gated route | 0.9423 | 0.9857 | **0.8781** |

"Degraded" is the competition's headline metric.

**Uniform fusion beats the best single expert** by +3.39 / +1.67 points (shallow / deep) on
official val, and +5.89 / +1.74 on the hard split. The experts are not making identical
errors — averaging their logits reduces variance.

**Early exit costs nothing.** Staged inference is numerically identical to a full forward
pass (max absolute difference = 0 across all six taps). On the official val set the gate
marks ~60% of images clean, for a measured **19.7% reduction in forward depth**. ⚠️

**We did not hand-pick the tap layers.** A linear probe was fitted to every block, candidate
triples were scored by rank-averaged AUC, and a permutation test checked them against
randomly constructed alternatives.

---

## Quick start

```bash
git clone https://github.com/kelvinchua1099-alt/DUET_v0.git
cd DUET_v0

hf download TechJam2026-Jamlai-Bench/squade-vitg \
  --include "v3/*" --local-dir ckpt
```

DINOv2 weights download automatically on first run.

**Single directory:**

```bash
python Inference.py --dir /path/to/images \
  --shallow ckpt/v3/mix2_shallow \
  --deep    ckpt/v3/mix2_deep \
  --gate    ckpt/v3/mix2_gate.pt \
  --out     preds.csv \
  --json    preds.json
```

**Large corpus (past a few thousand images):**

```bash
python Inference.py --dir /corpus \
  --shallow ckpt/v3/mix2_shallow \
  --deep    ckpt/v3/mix2_deep \
  --gate    ckpt/v3/mix2_gate.pt \
  --device cuda --batch-size 16 \
  --out preds.csv --json preds.json --resume
```

`--resume` reads what is already in `--out`, skips those images, and appends. Rows are
flushed after every batch, so a crash, an OOM or a Ctrl-C loses nothing — re-run the
identical command. Unreadable files are logged to `<out>.failed` and never abort the run.

`--batch-size 16` fits in 24 GB. On MPS leave it at the default; batch 1 is fastest there.

> **Precision matters.** The expert heads were calibrated on bf16 features, so runtime
> precision must match. CUDA is correct by default. **On a Mac, pass `--device mps`** — the
> fallback to CPU silently switches to fp32, which is out-of-distribution for the heads.

Splitting across multiple GPUs: see [USAGE.md](USAGE.md).

---

## Output

### JSON (challenge submission format)

`--json` writes one entry per image:

```json
[
  {"image_path": "/corpus/002d7df53e3ae55af5.jpg", "pred": 0.836464},
  {"image_path": "/corpus/0053097bfa680600.jpg",   "pred": 0.605878}
]
```

`pred` ∈ [0, 1] is the likelihood that the image is AI-generated, and `pred > 0.5` is exactly
the `FAKE` verdict. It is **not** the plain `sigmoid` of the fused logit — that is the CSV's
`score` column, which saturates to `1.000000` for most rows because this model's logits
routinely reach ±100, and ties like that cost AUC for nothing. `pred` centres the sigmoid on
the decision threshold and divides by a temperature instead:

```
pred = sigmoid((z - threshold) / 20)
```

Monotone in `z`, so ranking — and therefore AUC — is identical to ranking by the raw logit,
but the values stay distinguishable. `pred_score()` in [Inference.py](Inference.py) is the
one place this is defined.

Unreadable files (truncated downloads, mislabelled extensions) are skipped, logged to
`<out>.failed`, and still get a JSON row with `pred: 0.5` — abstention — so the entry count
always matches the input directory.

### CSV (analysis)

```
image                    pred      verdict  confidence  score     route    votes
002d7df53e3ae55af5.jpg   0.836464  FAKE     84.7        1.000000  deep     1.000|1.000|1.000
0053097bfa680600.jpg     0.605878  FAKE      1.3        0.474883  shallow  0.000|0.000|1.000
```

| Column | Meaning |
|---|---|
| `pred` | Same value as `pred` in the JSON. **The submission column** |
| `verdict` | `FAKE` / `REAL`, thresholded in logit space (`--threshold`, default −8.7) |
| `confidence` | 0–100 reliability index. **Not a probability** |
| `score` | Raw `sigmoid(z)`, kept for continuity with earlier runs. Saturates — **do not rank by it** |
| `route` | Which bank the image went through (`shallow` = early exit) |
| `votes` | The three experts' individual probabilities |
| `depth_used` | Blocks actually run (27 or 37 of 40) |
| `gate_logit` | ≤ 0 routes to the shallow bank |

**Sort by `confidence`, not `score`.** Confidence drops when the margin to the threshold is
small **or when the three experts disagree** — and disagreement is the case `score` hides.
Row 2 above is exactly that: the experts return (−98.9, −50.5, +149.1), the mean of −0.10
barely clears the threshold, but `score` still reads 1.000.

```bash
# 200 least reliable verdicts, for manual review
head -1 preds.csv > review.csv
tail -n +2 preds.csv | sort -t, -k4 -g | head -200 >> review.csv
```

Full column list, every flag, and the `pred` / `confidence` formulas: [USAGE.md](USAGE.md).
Steps that reproduce the numbers above, from data download to evaluation, are in the same
file: [Reproducing our results](USAGE.md#reproducing-our-results).

---

## Input handling

Inference and training follow **the same rules**. A mismatch introduces an undocumented
domain shift that nothing will warn you about.

```
short side ≥ 512   center-crop 512×512 — no resampling, so native high frequencies
                   survive (that is where generator fingerprints live)

short side < 512   crop square, then upsample to 512 with a kernel chosen
                   deterministically from the filename among
                   [BILINEAR, BICUBIC, BOX, LANCZOS]
                   — a fixed kernel would make "which interpolation trace does this
                     image carry" a content-independent shortcut
```

Then center-crop to **504** (a multiple of patch=14). `Inference.py` handles all of this.

---

## Known limitations

**1. The gate learned "how blurry", not "was an operator applied".** On natively
low-resolution sources (tested with images upsampled from 200×200), 82% were routed as
degraded and the compute saving fell to 5.8%. Accuracy was unaffected — only the saving.

**2. Real images in training skew towards professional photography** (stock libraries,
OpenImages). Casual phone photos that messaging apps have re-compressed are not represented,
and in practice such images are frequently misclassified as fake. **Performance on
low-quality real photographs is unverified.**

**3. Held-out sets are small.** With 750 val_hard images the standard error is ±1.8 points,
so differences below 3–4 points cannot be read. Tap layers were also selected on the
official val sets. The only fully clean evaluation would be the official test set.

---

## Team member contributions

| Member | Contribution |
|---|---|
| **Cai Yizhong** | Algorithm architecture design and model training — the two-bank / early-exit design, tap-layer probing, expert-head and gate training |
| **Zhang Heng** | Algorithm architecture design and model training — degradation-aware routing, evaluation protocol, checkpoint iteration v1 → v3 |
| **Wang Yunxiang** | Dataset design — degradation taxonomy and codebook, training-mix composition, manifest and split construction |
| **Yin Lichen** | Dataset design — data collection and cleaning, official-label alignment, group-wise train/val splitting and leakage checks |
| **Wang Yajie** | Frontend and backend development, and web design — the demo webapp: FastAPI scoring service, the contact-sheet UI, and its visual design (kept in a separate repository) |

---

## License

MIT. DINOv2 weights are subject to their own license.

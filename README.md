# DUET - Depth-Uniform Ensemble with early eTerminate


A frozen **DINOv2 ViT-g** plus two banks of three expert heads, detecting AI-generated
images. Clean images exit early at block 27, saving roughly 20% of the forward depth.
Trained for NTIRE 2026 Robust AIGI Detection.

```
input → DINOv2 ViT-g (frozen, 40 blocks)
         │
         ├─ run to L27, tap L14/L21/L27 → shallow bank, 3 experts, uniform mean → score
         │                              → gate (reads the same shallow features)
         │       gate says "clean" → output here; the remaining 13 blocks never run
         │
         └─ gate says "degraded" → resume from L27's hidden state to L37,
                                   tap L26/L33/L37 → deep bank
```

---

## Quick start

```bash
git clone https://github.com/kelvinchua1099-alt/DUET_v0.git
cd DUET_v0
```

The DINOv2 weights download automatically on first run; no access request needed.

```bash
# Fetch the v3 expert heads and gate
hf download TechJam2026-Jamlai-Bench/squade-vitg \
  --include "v3/*" \
  --local-dir ckpt

# Run over a directory
python Inference.py --dir /path/to/images \
  --shallow ckpt/v3/mix2_shallow \
  --deep    ckpt/v3/mix2_deep \
  --gate    ckpt/v3/mix2_gate.pt \
  --device  mps \
  --out preds.csv

# Single image
python Inference.py --image a.jpg --shallow ckpt/v3/mix2_shallow \
  --deep ckpt/v3/mix2_deep --gate ckpt/v3/mix2_gate.pt --device mps
```

**v3 is the currently recommended version.** v1 and v2 are kept for comparison; drop
`--include` to fetch them too. Their paths are `ckpt/v1/vg_*_full` + `ckpt/v1/gate_full.pt`
and `ckpt/v2/mix_*` + `ckpt/v2/mix_gate.pt`. All three versions share the same tap layers,
backbone, precision and input window, so they are drop-in replacements for each other.
[Inference.py](Inference.py) compares the checkpoint's `cache_cfg` against the runtime at
startup and raises an error on any mismatch.

**On a Mac you must pass `--device mps` explicitly.** The default is
`"cuda" if torch.cuda.is_available() else "cpu"`, which falls through to CPU on a Mac, and
CPU silently switches the backbone from bf16 to fp32. The expert heads were calibrated on
bf16 cached features, so fp32 is out-of-distribution input: **the verdict flips and nothing
is reported**. Measured on one image: bf16 gives a fused logit of −0.10 (FAKE), fp32 gives
−29.53 (REAL). CUDA machines need no flag; the default is already correct.

## Output

```
image                    verdict  confidence  score     label  route    experts       votes
002d7df53e3ae55af5.jpg   FAKE     84.7        1.000000  1      deep     L26/L33/L37   1.000|1.000|1.000
0053097bfa680600.jpg     FAKE      1.3        0.474883  1      shallow  L14/L21/L27   0.000|0.000|1.000
```

| Column | Meaning |
|---|---|
| `verdict` | `FAKE` / `REAL`, decided by a threshold in logit space (`--threshold`, default −8.7) |
| `confidence` | 0–100 reliability index. **Not a probability** — see below |
| `score` | 0–1, higher means more likely AI-generated. Saturates at both ends; do not use it as the decision criterion |
| `label` | 1 when `verdict == FAKE` |
| `route` | Which bank of experts the image went through |
| `experts` | That bank's three tap layers |
| `votes` | Each of the three experts' own probabilities |
| `vote_spread` | Largest gap among the three votes. **Large = the image sits on the decision boundary**, most worth a manual look |
| `depth_used` | How many blocks actually ran (27 or 37, out of 40) |
| `gate_logit` | The gate's output; ≤ 0 means "clean" and routes to the shallow bank |
| `ms_per_img` | Time per image |

### How confidence is computed

```
confidence = 100 * tanh(|z - threshold| / 8) * exp(-(max(e) - min(e)) / 60)
```

`z` is the fused logit and `e` the three raw per-expert logits. Two things push it down:
**a small margin to the threshold**, or **disagreement among the three experts**. The second
is the important one — the bank averages logits uniformly, so a single saturated expert can
outvote the other two while `score` still reads 1.000000 and `vote_spread` still reads 1.000.
Neither of those columns exposes the problem.

The second row above is exactly that case: the three experts return (−98.9, −50.5, +149.1),
the mean of −0.10 barely clears the threshold, and confidence is only 1.3.
**Low-confidence samples need a manual review.**

The two scale constants live in `Inference.py` as `CONF_MARGIN_SCALE` and
`CONF_SPREAD_SCALE`. They were taken from the observed logit magnitudes of this checkpoint
and have never been calibrated on a labelled validation set.

## Common flags

```bash
--device mps          # required on Mac; optional on CUDA machines. **Do not use cpu** — see above
--threshold -8.7      # fake threshold, in logit space. Pass 0.0 for the old score>0.5 behaviour
--batch-size 8        # can go to 16 on 24 GB of VRAM
--no-early-exit       # run both banks, then pick by the gate. Same accuracy; use it to verify
                      # that early exit does not change results
--crop-size 504       # default; a multiple of DINOv2's patch=14. Do not change
```

---

## Running at scale

For anything past a few thousand images, use these three flags. They exist because a long
run will otherwise lose everything to a single bad file or a killed process.

```bash
python Inference.py --dir /corpus \
  --shallow ckpt/v3/mix2_shallow \
  --deep    ckpt/v3/mix2_deep \
  --gate    ckpt/v3/mix2_gate.pt \
  --device  cuda --batch-size 16 \
  --out preds.csv --resume
```

| Flag | What it buys you |
|---|---|
| `--resume` | Reads the images already in `--out`, skips them, and appends. Re-run the identical command after a crash, an OOM or a Ctrl-C and it picks up where it stopped |
| `--shard I/N` | Processes only shard `I` of `N` (0-based). Run N processes to split a corpus across GPUs or machines |
| `--batch-size` | 16 fits in 24 GB. On MPS leave it at the default: batch 1 is fastest there |

Rows are written and flushed **after every batch**, so a killed run keeps everything
computed so far. Unreadable files (truncated downloads, mislabelled extensions, non-images)
are skipped with a `[skip]` line and appended to `<out>.failed`; they never abort the run.
Progress prints a running rate and ETA.

### Splitting across 4 GPUs

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python Inference.py --dir /corpus \
    --shallow ckpt/v3/mix2_shallow --deep ckpt/v3/mix2_deep \
    --gate ckpt/v3/mix2_gate.pt --device cuda --batch-size 16 \
    --out preds.$i.csv --shard $i/4 --resume &
done
wait

# Merge: keep one header, concatenate the rest
head -1 preds.0.csv > preds.csv
tail -q -n +2 preds.0.csv preds.1.csv preds.2.csv preds.3.csv >> preds.csv
```

The file list is sorted before sharding, so the shards are disjoint and their union is the
full corpus regardless of how many processes finish. `--resume` is per-shard, so any single
shard can be restarted on its own.

### How long it takes

| Hardware | Throughput | 100k images |
|---|---|---|
| 24 GB GPU, batch 16 | ~1.5 img/s | ~18 h |
| Apple M-series, MPS | ~0.6 img/s | ~46 h |

The gate's early exit only helps on corpora with clean images in them; a corpus that is
entirely re-compressed photographs routes everything to the deep bank and saves nothing.
The run summary reports the actual saving.

### Reading the output

`score` saturates to `1.000000` on most rows and is not useful for triage. Sort by
`confidence` instead — the low end is where the three experts disagree:

```bash
# 200 least reliable verdicts, for manual review
head -1 preds.csv > review.csv
tail -n +2 preds.csv | sort -t, -k3 -g | head -200 >> review.csv
```

---

## How input size is handled

Inference and training follow **the same rules**. A mismatch adds an undocumented domain
shift to your test set, and nothing raises an error.

```
short side >= 512   center-crop 512x512, not a single resampling step
                    — preserves native high frequencies (that is where generator
                      fingerprints live) and removes the "scale factor" shortcut
short side <  512   crop to a square first, then upsample to 512 with a randomly
                    chosen kernel from [BILINEAR, BICUBIC, BOX, LANCZOS],
                    picked deterministically from the file name
                    — with a fixed kernel, "which interpolation trace does this
                      image carry" becomes a content-independent spurious cue
```

The image is then center-cropped to **504** (a multiple of patch=14) before the backbone,
again without scaling. `Inference.py` does all of this automatically; no manual
preprocessing is required.

---

## Speed

24 GB of VRAM, batch size 8, 504×504: about **1.5 images/s** (684 ms per image).

Clean images stop at L27, skipping L28–L37, 13 of 40 blocks. On the official val set the
gate marks about 60% of images clean, for a measured **19.7%** saving in forward depth.

⚠️ The gate fails on **natively low-resolution** sources. What it actually learned is "how
blurry is this image", not "has a degradation operator been applied". Measured on images
upsampled from 200×200, 82% were judged degraded and the compute saving fell to 5.8%.
Accuracy is unaffected.

---

## Results

**Held-out numbers only.** 70% of the official val set and 70% of val_hard went into v3's
training. The split files (`v3/val_split_70_30.json`, `v3/valhard_split_70_30.json`) ship
with the weights so the split can be verified independently.

### v3 (current)

| Data | Scheme | Overall | Clean | Degraded |
|---|---|---|---|---|
| Official val held-out (3,000) | shallow-only | **0.9922** | **0.9978** | **0.9833** |
| ±SE 0.23 | gated route | 0.9911 | 0.9977 | 0.9809 |
| Official val_hard held-out (750) | shallow-only | 0.9430 | **0.9904** | 0.8674 |
| ±SE 1.8 | gated route | 0.9423 | 0.9857 | **0.8781** |

### Three versions compared

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| val_hard held-out, degraded | — | 0.8068 | **0.8674** |
| val_hard held-out, overall | — | 0.9111 | **0.9430** |
| val held-out, degraded | 0.9629 | 0.9830 | **0.9833** |
| Gate AUC (val_hard) | 0.7684 | 0.9072 | **0.9464** |

*val_hard had not been split yet when v1 was trained, so it has no held-out numbers.*

- **v1 → v2**: training-set mix (NTIRE raised from 11.4% to 37.9%), degradation strength
  (our own levels blended with the official `distortion_range`), and 70% of the official val
  set folded in. The three contributions were not separated.
- **v2 → v3**: 70% of official val_hard folded in (1,750 rows), warm-started from v2 for
  8 epochs at lr 1e-4. val_hard held-out degraded **+6.06**, with val held-out unchanged
  (+0.03).

"Degraded" = computed on degraded images only, which is the competition's headline metric.

### Known limitations

1. Both official evaluation sets were partly used for training, leaving only 3,000 + 750
   held-out images. With 750 images the standard error is ±1.8 points, so differences below
   3–4 points cannot be read. The only fully clean set is the official test set.
2. The tap layers were selected on the official val and val_hard sets.
3. **The gate fails on natively low-resolution sources.** What it learned leans towards "how
   blurry is this image" rather than "has a degradation operator been applied".
4. **The training set's real images do not cover casual phone photos that have been
   re-compressed by messaging apps.** The real images come from stock libraries and
   OpenImages and skew towards professional photography. **Performance on low-quality real
   photographs is unverified**, and in practice such images are frequently misclassified
   as fake.
5. Part of the training data's "clean" tier is not truly clean — the source images are
   themselves JPEGs, carrying undocumented compression.

---

## License

MIT. The DINOv2 weights are subject to their own license terms.

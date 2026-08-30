# DUET — Depth-Uniform Ensemble with early eTerminate

**TechJam 2026, Track 5 — Robust Detection of AI-Generated Images Under Real-World
Transformations.** Team **Jamlai**.

A frozen **DINOv2 ViT-g** plus two banks of three expert heads. Clean images exit early at
block 27, saving roughly 20% of the forward depth. Only the heads and the gate are trained
(≈280k parameters, 35 MB); the backbone never moves.

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

## 1. Project overview

### The problem

A detector that scores 0.99 on pristine generator output is worth little in the wild, where
every image has been through JPEG, a resize, a screenshot, a messaging app. The competition
metric is AUC **on degraded images**, and that is where naive detectors collapse.

### The claim we build on

**Different degradations kill forensic evidence at different depths of the network.** JPEG
and noise destroy the high-frequency, patch-local traces that live in *shallow* ViT blocks;
they leave *deep* semantic-composition cues intact. Heavy blur and downscaling do the
reverse. So a single fixed read-out depth is a compromise that is wrong for most images.

We verified this before building anything. `probe_layers.py` fits one linear probe per layer
per degradation bucket and runs a **falsification gate**: if every bucket's best layer lands
in the same place, depth routing has no value and the whole design should be abandoned. On
our own synthetic degradations the gate **failed** (max tie = 5, needs ≤3) — the degradations
were too weak to kill anything. On NTIRE's official per-image degradation ground truth it
**passed** on all three criteria (span 9, 10 distinct values, max tie 1). That negative
result is documented rather than hidden; it is why we train on official-degradation data.

### What we actually shipped

The honest outcome of that investigation is narrower than the original ambition, and the
architecture reflects it:

* **Per-image soft routing over depth did not survive cross-fitting** (out-of-fold net
  +0.117 AUC, p = 0.32). We dropped it. Each bank instead **averages its three experts'
  logits uniformly** — no learned weights, nothing to overfit.
* **A binary route did survive.** A gate decides only *clean vs degraded* and picks the
  shallow or the deep bank. It reads shallow features, which is the architectural crux: it
  must be computable *before* the early-exit point, or the compute saving disappears.
* The gate therefore buys **compute**, not accuracy: 19.7% of forward depth on the official
  val set, with AUC unchanged (0.9922 → 0.9911 overall, and *up* on val_hard degraded).

### Repository map

| Path | What it is |
|---|---|
| [Inference.py](Inference.py) | **The submission script.** Directory in → JSON out |
| [cache_features.py](cache_features.py) | Freeze the backbone, dump per-layer features to disk |
| [probe_layers.py](probe_layers.py) | Per-layer linear probes + the falsification gate |
| [training/train_WeightandExpert.py](training/train_WeightandExpert.py) | Stage 1 expert banks, Stage 2 router, eval |
| [tools/train_gate.py](tools/train_gate.py) | The clean/degraded gate |
| [utils/preprocess_ntire.py](utils/preprocess_ntire.py) | Degradation generation over NTIRE train, with manifest |
| [utils/manifest_ntire_val.py](utils/manifest_ntire_val.py) | Official degradation labels → our codeword manifest |
| [utils/ntire_aug/](utils/ntire_aug/) | Byte-for-byte vendored copy of the official transformation script |

---

## 2. Setup and installation

```bash
git clone https://github.com/kelvinchua1099-alt/DUET_v0.git
cd DUET_v0
pip install -r requirements.txt
```

`requirements-lock.txt` has the exact pinned versions we ran, if you need bit-level
reproducibility.

The DINOv2 backbone (`facebook/dinov2-giant`, 4.2 GB) downloads automatically on first run;
no access request is needed. Only the trained heads have to be fetched:

```bash
# 35 MB: expert heads and gate only, no backbone
hf download TechJam2026-Jamlai-Bench/squade-vitg \
  --include "v3/*" \
  --local-dir ckpt
```

**One pip constraint that bites:** do not upgrade `huggingface_hub` to 1.x — transformers 4.x
pins `<1.0`, and upgrading makes `import transformers` fail outright. Install the CLI as
`pip install "huggingface_hub[cli]<1.0"`.

**Hardware.** Any CUDA GPU with ≥8 GB works; 24 GB lets you raise `--batch-size` to 16.
Apple Silicon works via MPS. CPU technically runs but **must not be used** — see the warning
under §3.

---

## 3. Scoring a directory (the required script)

```bash
python Inference.py --dir /path/to/images \
  --shallow ckpt/v3/mix2_shallow \
  --deep    ckpt/v3/mix2_deep \
  --gate    ckpt/v3/mix2_gate.pt \
  --device  cuda \
  --json    preds.json
```

That writes exactly the requested submission format — a JSON list with `image_path` and
`pred` per image, `pred` ∈ [0, 1] being the likelihood the image is AI-generated:

```json
[
  { "image_path": "/data/test/0001.jpg", "pred": 0.836464 },
  { "image_path": "/data/test/0002.jpg", "pred": 0.072118 }
]
```

Add `--out preds.csv` to get the full diagnostic table alongside it. Both flags can be used
together, and with `--dir` replaced by `--image a.jpg` for a single file.

**On a Mac you must pass `--device mps` explicitly.** The default is
`"cuda" if torch.cuda.is_available() else "cpu"`, which falls through to CPU on a Mac, and
CPU silently switches the backbone from bf16 to fp32. The expert heads were calibrated on
bf16 cached features, so fp32 is out-of-distribution input: **the verdict flips and nothing
is reported**. Measured on one image: bf16 gives a fused logit of −0.10 (FAKE), fp32 gives
−29.53 (REAL). CUDA machines need no flag; the default is already correct.

### Why `pred` is not `sigmoid(z)`

```
pred = sigmoid((z - threshold) / 20)        # z = fused logit, threshold = -8.7
```

Monotone in `z`, so AUC is identical to ranking by the raw logit, and `pred > 0.5` is exactly
`verdict == FAKE`. Both properties fail for the plain sigmoid: its 0.5 sits at `z = 0`, not at
the −8.7 threshold, and this model's logits routinely reach ±100, where `sigmoid` **saturates
numerically to 1.000000**. In a 3-image smoke test all three rows had `score = 1.000000`
while `pred` read 0.836 / 0.771 / 0.763. Ties like that destroy AUC for free; the temperature
just pulls the range back to where float64 can tell rows apart.

Unreadable files (truncated downloads, mislabelled extensions) are skipped, logged to
`<out>.failed`, and emitted in the JSON with `pred: 0.5` — abstention — so the row count always
matches the input directory.

### The full CSV columns

```
image                    pred      verdict  confidence  score     label  route    experts       votes
002d7df53e3ae55af5.jpg   0.836464  FAKE     84.7        1.000000  1      deep     L26/L33/L37   1.000|1.000|1.000
0053097bfa680600.jpg     0.605878  FAKE      1.3        0.474883  1      shallow  L14/L21/L27   0.000|0.000|1.000
```

| Column | Meaning |
|---|---|
| `pred` | **0–1 likelihood of being AI-generated. This is the submission column** |
| `verdict` | `FAKE` / `REAL`, from a threshold in logit space (`--threshold`, default −8.7) |
| `confidence` | 0–100 reliability index. **Not a probability** — see below |
| `score` | Raw `sigmoid(z)`. Kept for continuity with earlier runs; **saturates, do not rank by it** |
| `label` | 1 when `verdict == FAKE` |
| `route` | Which bank of experts the image went through |
| `experts` | That bank's three tap layers |
| `votes` | Each of the three experts' own probabilities |
| `vote_spread` | Largest gap among the three votes. **Large = the image sits on the decision boundary** |
| `depth_used` | How many blocks actually ran (27 or 37, out of 40) |
| `gate_logit` | The gate's output; ≤ 0 means "clean" and routes to the shallow bank |
| `ms_per_img` | Time per image |

### How `confidence` is computed

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
**Low-confidence samples need a manual review.** Sort by it, not by `score`:

```bash
head -1 preds.csv > review.csv
tail -n +2 preds.csv | sort -t, -k4 -g | head -200 >> review.csv   # 200 least reliable
```

The two scale constants live in [Inference.py](Inference.py) as `CONF_MARGIN_SCALE` and
`CONF_SPREAD_SCALE`. They were taken from the observed logit magnitudes of this checkpoint
and **have never been calibrated on a labelled validation set**.

### Common flags

```bash
--device mps          # required on Mac; optional on CUDA. **Never cpu** — see above
--json preds.json     # submission JSON: [{image_path, pred}, ...]
--out preds.csv       # full diagnostic CSV
--threshold -8.7      # fake threshold, in logit space. Pass 0.0 for the old score>0.5 behaviour
--batch-size 8        # can go to 16 on 24 GB of VRAM
--no-early-exit       # run both banks, then pick by the gate. Same accuracy; use it to verify
                      # that early exit does not change results
--crop-size 504       # default; a multiple of DINOv2's patch=14. Do not change
```

### Checkpoint versions

**v3 is the recommended version.** v1 and v2 are kept for comparison; drop `--include` to
fetch them too. Their paths are `ckpt/v1/vg_*_full` + `ckpt/v1/gate_full.pt`, and
`ckpt/v2/mix_*` + `ckpt/v2/mix_gate.pt`. All three share the same tap layers, backbone,
precision and input window, so they are drop-in replacements for each other.
[Inference.py](Inference.py) compares the checkpoint's `cache_cfg` against the runtime at
startup and raises an error on any mismatch.

---

## 4. Running at scale

For anything past a few thousand images, use these three flags. They exist because a long
run will otherwise lose everything to a single bad file or a killed process.

```bash
python Inference.py --dir /corpus \
  --shallow ckpt/v3/mix2_shallow --deep ckpt/v3/mix2_deep --gate ckpt/v3/mix2_gate.pt \
  --device cuda --batch-size 16 \
  --out preds.csv --json preds.json --resume
```

| Flag | What it buys you |
|---|---|
| `--resume` | Reads the images already in `--out`, skips them, and appends. Re-run the identical command after a crash, an OOM or a Ctrl-C and it picks up where it stopped. The JSON is rebuilt from the whole CSV, so it stays complete |
| `--shard I/N` | Processes only shard `I` of `N` (0-based). Run N processes to split a corpus across GPUs or machines |
| `--batch-size` | 16 fits in 24 GB. On MPS leave it at the default: batch 1 is fastest there |

CSV rows are written and flushed **after every batch**, so a killed run keeps everything
computed so far. The JSON is written once at the end (from the CSV when one exists).

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

# Then turn the merged CSV into the submission JSON
python - <<'PY'
import csv, json
rows = [{"image_path": r["image"], "pred": float(r["pred"])}
        for r in csv.DictReader(open("preds.csv", newline=""))]
json.dump(rows, open("preds.json", "w"), indent=2)
PY
```

The file list is sorted before sharding, so the shards are disjoint and their union is the
full corpus regardless of how many processes finish. `--resume` is per-shard.

### Throughput

| Hardware | Throughput | 100k images |
|---|---|---|
| 24 GB GPU, batch 16 | ~1.5 img/s | ~18 h |
| Apple M-series, MPS | ~0.6 img/s | ~46 h |

Early exit only helps on corpora that contain clean images; a corpus that is entirely
re-compressed photographs routes everything to the deep bank and saves nothing. The run
summary reports the actual saving.

---

## 5. Reproducing our results

The pipeline has **hard dependencies** — steps cannot be reordered. `train_classifier` and
`probe_layers` are the only pair that can run in parallel.

```
utils/preprocess_ntire.py + utils/manifest_ntire_val.py     degradations + manifests
        │
        ├─► cache_features.py --layers all ─► probe_layers.py    ← pick the tap layers
        │                                     tools/bootstrap_oracle.py  ← the gate
        │
        └─► cache_features.py --layers <the six chosen layers>
                     │
                     ├─► training/train_WeightandExpert.py stage1   ← the two banks
                     ├─► tools/train_gate.py                        ← the clean/degraded gate
                     └─► tools/eval_two_groups.py                   ← the reported numbers
```

Everything below assumes `export SQUADE_TAXONOMY=ntireval`. The codeword taxonomy is
recorded in each cache's `config.json`, and `probe_layers.py` refuses to run against a cache
built under a different one — a mismatch would silently mix incompatible degradation codes.

> The `tools/dl_ntire_*.py` download helpers hardcode `/workspace/...` paths from the machine
> we trained on. Edit the destination at the top of each, or set `HF_HOME` and use
> `hf download` directly.

### Step 1 — data

```bash
# Official val + val_hard: images and per-image degradation ground truth
python tools/dl_ntire_val.py

# Official labels -> our codeword manifest (no pixel is modified here;
# NTIRE already applied the degradations and tells you which ones, per image)
python utils/manifest_ntire_val.py --labels data/ntire_val/_dl/val_labels.csv \
       --images data/ntire_val/val_images --out-csv data/manifest_val.csv --tag val
python utils/manifest_ntire_val.py --labels data/ntire_val/_dl/val_hard_labels.csv \
       --images data/ntire_val/val_images_hard --out-csv data/manifest_valhard.csv --tag val_hard

# NTIRE train shards are clean source images; degradations must be applied by us,
# using the vendored official transformation script
python tools/dl_ntire_train.py shard_5.zip
SQUADE_TAXONOMY=ntire python utils/preprocess_ntire.py --shards data/ntire/shard_5 \
       --out data/ntire_deg --out-csv data/manifest_ntire.csv --level-dist ntire
```

`preprocess_ntire.py` normalizes **before** degrading and splits train/val **by source
image group**, so a clean image and its degraded siblings never land on opposite sides. It
prints a cross-split group count that must read 0.

### Step 2 — tap-layer selection, and the falsification gate

```bash
python cache_features.py --manifest data/manifest_val.csv --out cache/probe_val --layers all
python probe_layers.py --cache cache/probe_val --out probe/val
python tools/bootstrap_oracle.py --cache cache/probe_val --out probe/val
python tools/pick_tap_layers.py --cache cache/probe_val --scores probe/val/probe_scores_l20.001.npy
```

**Read the falsification report before continuing.** It must show, across degradation
buckets: best-layer span ≥ 3, ≥ 3 distinct best layers, and max tie ≤ 3. Ours read 9 / 10 / 1
on official-degradation data and 7 / 5 / **5** (fail) on our own synthetic degradations. If it
fails, depth routing has no headroom and building experts is three wasted hours; the script
prints which of the three criteria failed and what to check next.

`bootstrap_oracle.py` is the statistical guard on top: it reports a bootstrap interval and a
permutation null, because a max over 33 layers produces a positive oracle gap even when no
real depth preference exists.

This is how L14/L21/L27 (shallow) and L26/L33/L37 (deep) were chosen.

### Step 3 — cache the six tap layers and train

```bash
python cache_features.py --manifest data/manifest_val.csv --out cache/val_shallow --layers 14,21,27
python cache_features.py --manifest data/manifest_val.csv --out cache/val_deep    --layers 26,33,37

# v3's training mix: our degraded NTIRE-train data + 70% of official val + 70% of val_hard.
# The 70/30 split files ship with the weights, so the held-out 30% can be verified.
python tools/concat_cache.py --out cache/mix2_shallow \
  --src cache/ntire_shallow \
  --src cache/val_shallow:train:ckpt/v3/val_split_70_30.json \
  --src cache/valhard_shallow:train:ckpt/v3/valhard_split_70_30.json
# ... same for the deep bank

python training/train_WeightandExpert.py stage1 --cache cache/mix2_shallow --out ckpt/v3/mix2_shallow
python training/train_WeightandExpert.py stage1 --cache cache/mix2_deep    --out ckpt/v3/mix2_deep
python tools/train_gate.py --cache cache/mix2_shallow --out ckpt/v3/mix2_gate.pt
```

v3 was warm-started from v2 for 8 epochs at lr 1e-4 (`stage1 --init ckpt/v2/mix_shallow/stage1.pt
--epochs 8 --lr 1e-4`) rather than trained from scratch.

Note that `stage1` is all that ships. `stage2` (the learned per-image router) exists in the
repo and can be run, but its gain did not survive cross-fitting, so the released checkpoints
use uniform averaging. If you do run it, feed it **predicted** codewords from
`train_classifier.py predict`, never ground-truth ones — ground truth is unavailable at test
time, and training on it inflates the numbers.

### Step 4 — evaluation

```bash
python tools/eval_two_groups.py \
  --shallow-run ckpt/v3/mix2_shallow --shallow-cache cache/val_shallow \
  --deep-run    ckpt/v3/mix2_deep    --deep-cache    cache/val_deep \
  --gate ckpt/v3/mix2_gate.pt --held-out ckpt/v3/val_split_70_30.json --tag "official val"
```

It reports four numbers — shallow-only, deep-only, oracle route (upper bound, not
achievable), and gated route. `tools/gate_sweep.py` sweeps the routing threshold; pick τ on
one dataset and report it on the other, never both.

Always report **per degradation tier**, with ≥3 seeds and error bars. On a 750-image split
the standard error is ±1.8 points, and we once mistook single-seed noise for a result.

### Results

**Held-out numbers only.** 70% of the official val set and 70% of val_hard went into v3's
training. The split files (`v3/val_split_70_30.json`, `v3/valhard_split_70_30.json`) ship
with the weights so the split can be verified independently.

| Data | Scheme | Overall | Clean | Degraded |
|---|---|---|---|---|
| Official val held-out (3,000) | shallow-only | **0.9922** | **0.9978** | **0.9833** |
| ±SE 0.23 | gated route | 0.9911 | 0.9977 | 0.9809 |
| Official val_hard held-out (750) | shallow-only | 0.9430 | **0.9904** | 0.8674 |
| ±SE 1.8 | gated route | 0.9423 | 0.9857 | **0.8781** |

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
  8 epochs at lr 1e-4. val_hard held-out degraded **+6.06**, val held-out unchanged (+0.03).

"Degraded" = computed on degraded images only, which is the competition's headline metric.

### Preprocessing — identical at train and test

A mismatch here adds an undocumented domain shift to your test set, and nothing raises an
error.

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
again without scaling. [Inference.py](Inference.py) does all of this automatically.

---

## 6. Limitations, and what we would improve with more time

### What is actually wrong today

1. **Both official evaluation sets were partly used for training.** Only 3,000 + 750 images
   remain held out. At 750 images the standard error is ±1.8 points, so differences below
   3–4 points are unreadable. The only fully clean set is the official test set. *Fix: stop
   folding evaluation data into training and buy the headroom back with more train shards
   instead — we used 1 of 6 available shards (27k of 277k images).*
2. **The tap layers were selected on official val and val_hard**, the same data we report on.
   *Fix: nested cross-validation, or select layers on a shard of NTIRE train.*
3. **The gate learned "how blurry is this image", not "was a degradation operator applied".**
   On natively low-resolution sources (200×200 upsampled) it calls 82% degraded and the
   compute saving falls from 19.7% to 5.8%. Accuracy is unaffected — it only wastes compute.
   *Fix: train it on native-low-res negatives, and add a soft-fusion route so a gate error
   costs a weighted blend rather than the whole bank ([tools/gate_sweep.py](tools/gate_sweep.py)
   sketches this; the hard route amplifies every gate mistake into a full bank swap).*
4. **The real images skew professional.** They come from stock libraries and OpenImages, and
   do not cover casual phone photos that a messaging app has re-compressed. Performance on
   low-quality real photographs is **unverified, and in practice such images are frequently
   misclassified as fake** — the most likely source of real-world false positives.
   *Fix: this is a data problem, not a model problem. Add a WhatsApp/WeChat-recompressed
   real-photo tier to the training mix.*
5. **Part of the "clean" tier is not clean.** NTIRE train ships as JPEG (we verified the
   luminance quantization tables are identical, standard IJG q≈93, across 360 images from
   6 shards), so every "clean" row carries undocumented compression not recorded in its
   codeword. It biases the clean baseline upward uniformly, not between buckets.
6. **`confidence` is uncalibrated.** Its two constants were read off this checkpoint's logit
   magnitudes and never fitted. Treat it as a ranking aid for manual review, nothing more.
7. **The depth-routing thesis is only half-confirmed.** Different degradations *do* prefer
   different depths (the probe gate passed decisively), but a learned per-image router over
   those depths did not beat uniform averaging out-of-fold. The ceiling was +0.37 AUC points
   with ~100 samples per bucket. *Fix: more images with degradation ground truth. We had
   15,000; the effect is real but too small to fit against at that size.*

### What we would build next, in priority order

1. **Retrain on all 6 NTIRE train shards** with the official distortion pipeline. Everything
   else on this list is downstream of having 10× the degradation-labelled data.
2. **Calibrate the threshold and `confidence`** on a proper held-out set, and report expected
   calibration error alongside AUC. A detector people act on needs a trustworthy probability.
3. **Soft routing instead of hard.** Weight the two banks by gate confidence; a gate error
   then costs proportionally rather than swapping the entire bank.
4. **Test-time augmentation over the interpolation-kernel choice**, which currently injects a
   deterministic-but-arbitrary cue for sub-512 images.
5. **Ablate the ensemble honestly** — 3 seeds with error bars per configuration. Several
   numbers in this README rest on single runs, and we know from experience that single-seed
   differences on a small val set are noise up to ±0.03.

---

## 7. Team member contributions

| Member | Contribution |
|---|---|
| **Cai Yizhong** | Algorithm architecture design and model training — the two-bank / early-exit design, tap-layer probing, expert-head and gate training |
| **Zhang Heng** | Algorithm architecture design and model training — degradation-aware routing, evaluation protocol, checkpoint iteration v1 → v3 |
| **Wang Yunxiang** | Dataset design — degradation taxonomy and codebook, training-mix composition, manifest and split construction |
| **Yin Lichen** | Dataset design — data collection and cleaning, official-label alignment, group-wise train/val splitting and leakage checks |
| **Wang Yajie** | Frontend and backend development, and web design — the demo webapp: FastAPI scoring service, the contact-sheet UI, and its visual design (kept in a separate repository) |

---

## 8. License

MIT. The DINOv2 weights are subject to their own license terms.

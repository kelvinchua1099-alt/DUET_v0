# DUET — usage reference

The detail that [README.md](README.md) links out to: every CSV column, how `confidence` and
`pred` are computed, sharding a corpus across GPUs, and the steps that reproduce the
published numbers.

---

## All command-line flags

```bash
--dir DIR / --image PATH   # a directory (searched recursively) or one file. Required
--shallow ckpt/v3/mix2_shallow     # shallow-bank run dir (must contain stage1.pt)
--deep    ckpt/v3/mix2_deep
--gate    ckpt/v3/mix2_gate.pt

--json PATH        # submission JSON: [{"image_path", "pred"}, ...]
--out PATH         # full diagnostic CSV; prints to stdout if omitted
--resume           # skip images already in --out and append; makes a killed run restartable
--shard I/N        # process only shard I of N (0-based), for splitting across machines

--device cuda       # optional; defaults to CUDA, then MPS, then CPU
--batch-size 8      # released A100 benchmark setting; reduce if memory is limited
--threshold -8.7   # decision threshold in **logit space**. 0.0 restores score>0.5 behaviour
--crop-size 504    # a multiple of DINOv2's patch=14. Do not change
--no-early-exit    # run both banks, then pick by the gate. Same accuracy, no compute saved;
                   # use it to verify that early exit does not change results
--allow-mismatch   # run even if the checkpoint's cached-feature config differs from the
                   # runtime. Downgrades the error to a warning; results may be invalid
--model            # backbone id, default facebook/dinov2-giant
```

## The full CSV column list

| Column | Meaning |
|---|---|
| `image` | Path as given to `--dir` / `--image` |
| `pred` | **Uncalibrated 0–1 ranking score for AI-generated content. The submission column** |
| `verdict` | `FAKE` / `REAL`, from a threshold in logit space (`--threshold`, default −8.7) |
| `confidence` | 0–100 reliability index. **Not a probability** — see below |
| `score` | Raw `sigmoid(z)`. Kept for continuity with earlier runs; saturates, do not rank by it |
| `label` | 1 when `verdict == FAKE` |
| `route` | Which bank of experts the image went through |
| `experts` | That bank's three tap layers |
| `votes` | Each of the three experts' own probabilities |
| `vote_spread` | Largest gap among the three votes. Large = the image sits on the boundary |
| `depth_used` | How many blocks actually ran (27 or 37) |
| `gate_logit` | The gate's output; ≤ 0 means "clean" and routes to the shallow bank |
| `ms_per_img` | Model preprocessing and forward inference time; excludes image decoding and disk I/O |

## How `pred` is computed

```
pred = sigmoid((z - threshold) / 20)        # z = fused logit, threshold = -8.7
```

Monotone in `z`, so AUC is identical to ranking by the raw logit, and `pred > 0.5` is exactly
`verdict == FAKE`. Both properties fail for the plain sigmoid: its 0.5 sits at `z = 0`, not at
the −8.7 threshold, and this model's logits routinely reach ±100, where `sigmoid` saturates
numerically to `1.000000`. In a 3-image smoke test all three rows had `score = 1.000000` while
`pred` read 0.836 / 0.771 / 0.763 — ties like that cost AUC for free. The temperature just
pulls the range back to where float64 can tell rows apart.

The constant lives in `Inference.py` as `PRED_TEMPERATURE`.

## How `confidence` is computed

```
confidence = 100 * tanh(|z - threshold| / 8) * exp(-(max(e) - min(e)) / 60)
```

`z` is the fused logit and `e` the three raw per-expert logits. Two things push it down:
**a small margin to the threshold**, or **disagreement among the three experts**. The second
is the important one — the bank averages logits uniformly, so a single saturated expert can
outvote the other two while `score` still reads 1.000000 and `vote_spread` still reads 1.000.
Neither of those columns exposes the problem.

The two scale constants live in `Inference.py` as `CONF_MARGIN_SCALE` and
`CONF_SPREAD_SCALE`. They were taken from the observed logit magnitudes of this checkpoint
and **have never been calibrated on a labelled validation set**. Treat `confidence` as a
ranking aid for manual review, not as a probability.

---

## Splitting a corpus across GPUs

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python Inference.py --dir /corpus \
    --shallow ckpt/v3/mix2_shallow --deep ckpt/v3/mix2_deep \
    --gate ckpt/v3/mix2_gate.pt --device cuda --batch-size 8 \
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
full corpus regardless of how many processes finish. `--resume` is per-shard, so any single
shard can be restarted on its own.

### Throughput

The reported benchmark uses a single NVIDIA A100, BF16, batch size 8, and
504×504 inputs. It includes model preprocessing and forward inference but
excludes image decoding and disk I/O.

| Execution path | ms/image | images/s |
|---|---:|---:|
| Shallow committee | 30.2 | 33.1 |
| DUET gated route | 33.5 | 29.9 |
| Deep committee | 40.8 | 24.5 |

Actual end-to-end throughput also depends on storage speed, image decoding,
input resolution, and the fraction of images routed to each branch.

---

## Reproducing our results

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


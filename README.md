# DUET — Depth-Uniform Ensemble with early Termination
<img width="1536" height="1024" alt="main" src="https://github.com/user-attachments/assets/3dba46aa-3b0f-456d-860d-f8d7c55bc63c" />

DUET reads a frozen DINOv2 ViT-g/14 at six intermediate blocks rather than only its final layer. Each tap is summarised into a descriptor (CLS token, patch mean, patch std) and scored by a small 2-layer MLP head. The taps form two committees — a shallow group at blocks 14/21/27 and a deep group at 26/33/37 — and within each group the three logits are simply averaged, with no learned fusion weights. A binary gate, which sees only shallow features, predicts whether the image has been degraded: if not, inference stops at block 27 and returns the shallow score; if so, the backbone continues to block 37 and returns the deep score. Nothing in the 1.1B-parameter backbone is trained — only six MLP heads and one gate, about 7M parameters in total.

```
input → DINOv2 ViT-g/14 (frozen, 40 blocks)
         │
         ├─ run to L27, tap L14/L21/L27 → shallow bank, uniform mean in logit space
         │                              → gate (reads the same shallow features)
         │       "clean" → output here; L28–L37 never run
         │
         └─ "degraded" → resume to L37, tap L26/L33/L37 → deep bank
```
The routing design is backbone-agnostic in principle, but all results reported in this repository use DINOv2 ViT-g/14. Transfer to DINOv3 or other ViT backbones has not yet been validated.

---


## Results

### Primary evaluation — NTIRE official held-out splits

We used fixed image-level 70/30 splits for both official val and val_hard. Only the 70% training partitions were used to fit the expert heads and gate; every number below was computed exclusively on the disjoint 30% held-out partitions. The exact split manifests are released with the weights and mirrored here for independent verification: [official val split](splits/v3/val_split_70_30.json) and [val_hard split](splits/v3/valhard_split_70_30.json).

`Overall` is ROC-AUC over the complete held-out partition; it is not the average of the other two columns. `Clean AUC` is computed on images without degradation, while `Robust AUC` is computed on the degraded subset and is the competition's headline metric. Approximate standard errors are reported in AUC percentage points.

| Data | Scheme | Depth | ms/img | Overall | Clean AUC | Robust AUC |
|---|---|---:|---:|---:|---:|---:|
| Official val held-out (3,000)<br>±SE 0.41 | single layer L37 + 1 MLP | 37 | 40.8 | 0.9646 | 0.9739 | 0.9539 |
| | deep committee [26,33,37] | 37 | 40.8 | 0.9883 | 0.9933 | 0.9816 |
| | single layer L27 + 1 MLP | 27 | 30.2 | 0.9887 | 0.9948 | 0.9801 |
| | **shallow committee [14,21,27]** | 27 | 30.2 | **0.9922** | **0.9978** | **0.9833** |
| | gated route — DUET | 27 or 37 | 33.5 | 0.9911 | 0.9977 | 0.9809 |
| Official val_hard held-out (750)<br>±SE 0.82 | single layer L37 + 1 MLP | 37 | 40.8 | 0.8613 | 0.8906 | 0.8319 |
| | deep committee [26,33,37] | 37 | 40.8 | 0.9163 | 0.9569 | 0.8694 |
| | single layer L27 + 1 MLP | 27 | 30.2 | 0.9206 | 0.9744 | 0.8513 |
| | **shallow committee [14,21,27]** | 27 | 30.2 | **0.9430** | **0.9904** | 0.8674 |
| | gated route — DUET | 27 or 37 | 33.5 | 0.9423 | 0.9857 | **0.8781** |

*Latency was measured on a single NVIDIA A100 GPU with batch size 8 and BF16 inference at 504×504 resolution. Timing includes model preprocessing and forward inference, but excludes image decoding and disk I/O.*

### What the ablation shows

**Depth is not monotonic.** On both held-out splits, the single-layer L37 baseline is less accurate than single-layer L27 despite being deeper and slower. Reading the final available block is therefore a design choice rather than a reliable default.

**Multi-layer fusion is more useful than depth alone.** The shallow committee wins five of the six AUC comparisons against the deep committee while reducing latency from 40.8 to 30.2 ms/image, a 26% reduction. The only exception is Robust AUC on val_hard, where the deep committee is 0.20 percentage points higher.

**The gate trades a small amount of clean-set performance for robustness.** On val_hard, compared with the shallow committee, DUET gives up 0.47 percentage points of Clean AUC and adds 3.3 ms/image, but gains 1.07 points of Robust AUC. Compared with always running the deep route, it reduces latency from 40.8 to 33.5 ms/image. On the easier official val split, the shallow committee remains marginally ahead, so the gate's main benefit appears on harder degradations.

### External sanity check — COCO vs. DALL·E 3

We additionally evaluated the delivered gated model on 4,998 real COCO photographs and 8,843 DALL·E 3 generations, for 13,841 unmodified images in total.

| Scheme | ROC-AUC | Overall accuracy | Balanced accuracy |
|---|---:|---:|---:|
| DUET | 0.9972 | 97.60% | 96.86% |

| Dataset | Correct | Accuracy | Shallow route |
|---|---:|---:|---:|
| COCO (real) | 4,708 / 4,998 | 94.20% | 18.43% |
| DALL·E 3 (fake) | 8,801 / 8,843 | 99.53% | 84.33% |

This comparison is reported only as an external sanity check. Class label is coupled to dataset source and file-processing history, so it should not be interpreted as an independent robustness benchmark. The asymmetric routing rates suggest that the gate is sensitive to processing history rather than semantic real/fake status, consistent with the limitation discussed below.

### Training cost

Training was completed on a single NVIDIA RTX PRO 4000 in approximately six hours end-to-end. This includes image preprocessing, frozen-backbone feature caching, and training the expert heads and binary gate. The 1.1B-parameter DINOv2 backbone remained frozen throughout.

## Quick start

The checkpoint repository is access-gated. First [request access on Hugging Face](https://huggingface.co/TechJam2026-Jamlai-Bench/squade-vitg), then authenticate using the account that received access.

```bash
git clone https://github.com/kelvinchua1099-alt/DUET_v0.git
cd DUET_v0
pip install -r requirements.txt
hf auth login
hf download TechJam2026-Jamlai-Bench/squade-vitg \
  --include "v3/*" --local-dir ckpt
```

The public DINOv2 backbone weights are downloaded automatically on first use.

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
  --device cuda --batch-size 8 \
  --out preds.csv --json preds.json --resume
```

For input formats, checkpoint compatibility, resume behaviour, and the complete CLI reference, see [USAGE.md](USAGE.md).

## Output

### JSON (challenge submission format)

`--json` writes one entry per image:

```json
[
  {"image_path": "/corpus/002d7df53e3ae55af5.jpg", "pred": 0.836464},
  {"image_path": "/corpus/0053097bfa680600.jpg",   "pred": 0.605878}
]
```

`pred` ∈ [0, 1] is an uncalibrated ranking score for AI-generated content, not a calibrated probability. `pred > 0.5` is exactly equivalent to the `FAKE` verdict. It is not the plain sigmoid of the fused logit—that value is stored in the CSV `score` column and often saturates to `1.000000` because the model's logits can reach ±100.

Instead, `pred` centres the sigmoid on the decision threshold and applies a temperature:

```text
pred = sigmoid((z - threshold) / 20)
```

Monotone in `z`, so ranking — and therefore AUC — is identical to ranking by the raw logit,
but the values stay distinguishable. `pred_score()` in [Inference.py](Inference.py) is the
one place this is defined.

Unreadable files (truncated downloads, mislabelled extensions) are skipped, logged to
`<out>.failed`, and still get a JSON row with `pred: 0.5` — abstention — so the entry count
always matches the input directory.


## Limitations

**1. The gate does not detect "was an operator applied" — it detects "does this image look processed".** This is a genuine limitation and we want to state it plainly. The gate routes many images to the deep branch simply because they are blurry or heavily compressed, regardless of whether our pipeline actually degraded them; its decision is entangled with the preprocessing we used to construct the training pairs. On natively low-resolution sources (tested with images upsampled from 200×200), 82% were routed as degraded and the compute saving fell to 5.8%.

Accuracy was unaffected — only the saving. We suspect this is not purely a failure: an image that is already soft or heavily recompressed is genuinely ambiguous for the shallow experts, whether or not we were the ones who degraded it, so sending it to the deep branch is arguably the right call. The gate is a cheap proxy for "is the low-frequency evidence still trustworthy", and on that reading it behaves sensibly. But we cannot claim it identifies our operators, and a deployment where most inputs are natively low-quality would see much of the efficiency benefit disappear.

**2. Real images in training skew towards professional photography** (stock libraries,
OpenImages). Casual phone photos that messaging apps have re-compressed are not represented,
and in practice such images are frequently misclassified as fake. **Performance on
low-quality real photographs is unverified.**

**3. Generalisation to newer generators remains unverified.** Our training data is dominated by earlier generator families, and the external sanity check uses DALL·E 3. We have not yet evaluated DUET on newer generator families. The DALL·E 3 result should therefore not be interpreted as evidence of equivalent performance on more recent generators; evaluating this remains future work.

## Next for DUET

Our next steps follow directly from the limitations above. The gate needs to learn "was an operator applied" rather than "does this look soft" — training it on natively low-quality but un-degraded images, ideally with a second signal such as JPEG quantisation-table estimation, would decouple the two. 


We could also run the same comparison on newer backbones — DINOv3, SigLIP2, EVA-CLIP. If the two findings hold there too — that the final block isn't the best representation, and that multi-depth fusion helps — it's a property of self-supervised ViTs, not a quirk of DINO series. If they don't hold, our conclusions are backbone-dependent. Either way we learn something.

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

DUET source code is released under the [MIT License](LICENSE). DINOv2 weights remain subject to their original license terms.

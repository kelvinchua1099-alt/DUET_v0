# DUET — Depth-Uniform Ensemble with early Termination
<img width="1536" height="1024" alt="main" src="https://github.com/user-attachments/assets/3dba46aa-3b0f-456d-860d-f8d7c55bc63c" />

DUET reads a frozen DINOv2 ViT-g/14 at six intermediate blocks rather than only its final layer. Each tap is summarised into a descriptor (CLS token, patch mean, patch std) and scored by a small 2-layer MLP head. The taps form two committees — a shallow group at blocks 14/21/27 and a deep group at 26/33/37 — and within each group the three logits are simply averaged, with no learned fusion weights. A binary gate, which sees only shallow features, predicts whether the image has been degraded: if not, inference stops at block 27 and returns the shallow score; if so, the backbone continues to block 37 and returns the deep score. Nothing in the 1.1B-parameter backbone is trained — only six MLP heads and one gate, about 7M parameters in total.

```
input → DINOv2 ViT-g (frozen, 40 blocks)--The whole mechanism can be moved to Dinov3 and probably other ViT-based encoder.(We planned to used Dinov3 at first but wasn't very sure about the permission of a non-standard open-source license)
         │
         ├─ run to L27, tap L14/L21/L27 → shallow bank, uniform mean in logit space
         │                              → gate (reads the same shallow features)
         │       "clean" → output here; L28–L37 never run
         │
         └─ "degraded" → resume to L37, tap L26/L33/L37 → deep bank
```

---

## Results

All numbers are **held-out**. 70% of the official val set and 70% of val_hard were used in training; the split files ship with the weights (`v3/val_split_70_30.json`, `v3/valhard_split_70_30.json`) so the split can be verified independently.

### Cross-dataset check — COCO vs. DALL·E 3 (13,841 images/ Clean)

We first verified that the early-exit gate does not degrade predictions on a large out-of-distribution set: 4,998 real COCO photographs and 8,843 DALL·E 3 generations, each image run under both schemes.

| Scheme | ROC-AUC | Overall accuracy | Balanced accuracy |
|---|---|---|---|
| DUET | 0.99719648 | 97.60% | 96.86% |

| Dataset | Scheme | Correct | Accuracy | Shallow route |
|---|---|---|---|---|
| COCO (real) | DUET | 4,708 / 4,998 | 94.20% | 18.43% |
| DALL·E 3 (fake) | DUET | 8,801 / 8,843 | 99.53% | 84.33% |

The routing split is informative in itself: 84% of DALL·E 3 images take the shallow branch while only 18% of COCO photographs do. This matches how the two sets were produced — generator output is delivered as clean PNG, whereas COCO photographs have already been compressed and resized. The gate is reading the processing history, which is exactly what it was trained to do.

### Primary evaluation — NTIRE official splits

We used an extra benchmark to test our model because we thought the official validation set isn't challenging enough even for an encoder that is kind of outdated.

All numbers are **held-out**. 70% of the official val set and 70% of val_hard were used in training; the split files ship with the weights (`v3/val_split_70_30.json`, `v3/valhard_split_70_30.json`) so the split can be verified independently. "Robust" — AUC on the degraded subset — is the competition's headline metric.

| Data | Scheme | ms/img | Overall | Clean AUC | Robust AUC |
|---|---|---|---|---|---|
| Official val held-out (3,000)<br>±SE 0.41 | single layer L37 + 1 MLP | 40.8 | 0.9646 | 0.9739 | 0.9539 |
| | single layer L27 + 1 MLP | 30.2 | 0.9887 | 0.9948 | 0.9801 |
| | **DUET** | 33.5 | **0.9911** | **0.9977** | **0.9809** |
| Official val_hard held-out (750)<br>±SE 0.82 | single layer L37 + 1 MLP | 40.8 | 0.8613 | 0.8906 | 0.8319 |
| | single layer L27 + 1 MLP | 30.2 | 0.9206 | 0.9744 | 0.8513 |
| | **DUET** | 33.5 | **0.9423** | **0.9857** | **0.8781** |


## Quick start

```bash
git clone https://github.com/kelvinchua1099-alt/DUET_v0.git
cd DUET_v0
pip install -r requirements.txt
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


## Limitations

**1. The gate does not detect "was an operator applied" — it detects "does this image look processed".** This is a genuine limitation and we want to state it plainly. The gate routes many images to the deep branch simply because they are blurry or heavily compressed, regardless of whether our pipeline actually degraded them; its decision is entangled with the preprocessing we used to construct the training pairs. On natively low-resolution sources (tested with images upsampled from 200×200), 82% were routed as degraded and the compute saving fell to 5.8%.

Accuracy was unaffected — only the saving. We suspect this is not purely a failure: an image that is already soft or heavily recompressed is genuinely ambiguous for the shallow experts, whether or not we were the ones who degraded it, so sending it to the deep branch is arguably the right call. The gate is a cheap proxy for "is the low-frequency evidence still trustworthy", and on that reading it behaves sensibly. But we cannot claim it identifies our operators, and a deployment where most inputs are natively low-quality would see much of the efficiency benefit disappear.

**2. Real images in training skew towards professional photography** (stock libraries,
OpenImages). Casual phone photos that messaging apps have re-compressed are not represented,
and in practice such images are frequently misclassified as fake. **Performance on
low-quality real photographs is unverified.**

**3.Generalisation to the newest generators is unproven.** Our training data is dominated by earlier generator families, and the cross-dataset check used DALL·E 3 — a 2023 model. We have not evaluated on the current frontier: FLUX, SD 3.5, Qwen-Image, Nano Banana, Seedream, or GPT-Image. This matters because detection accuracy is known to fall sharply with generator recency; independent 2026 evaluations report the strongest open detectors dropping to 20–30% accuracy on the newest commercial models, well below chance. The 99.5% we measure on DALL·E 3 should not be read as evidence that DUET would hold up on a 2026 generator, and we would expect a substantial drop.

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

MIT. DINOv2 weights are subject to their own license.

# LeafLens — Plant Disease Detection with Explainable AI

A local FastAPI web app: upload a leaf → segment with **real MedSAM2** → classify Healthy/Diseased → inspect **class-specific Grad-CAM**. No mock predictions, API keys, or external inference services.

## Run

The environment, official MedSAM2 source/weights, and trained classifiers are included locally in this workspace. From this directory:

```bash
./run.sh
```

Open http://127.0.0.1:8000. Set `PORT=8001 ./run.sh` to choose another port. The server binds only to loopback. Upload JPG, PNG, or WebP (10 MB / 20 MP maximum). Draw a box around one leaf, or use the automatic inset full-frame prompt. Keyboard box entry is also available. Full-image classification is the default because the measured automatic ROI ablation reduced accuracy. Enable “Isolate the leaf” to run MedSAM2 and classify its masked ROI. Select either classifier, analyze, inspect the four image views, and download the JSON results or binary mask.

## Fresh installation

Python 3.9–3.12 supported by the tested app (upstream recommends Python 3.12). CPU works; CUDA/MPS are selected when available. For exact tested dependency versions, use `requirements-lock.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
bash scripts/setup_medsam2.sh
.venv/bin/python scripts/train.py --data '/path/to/plant_dataset' --epochs 10
./run.sh
```

Dataset layout: `plant_dataset/Diseased/*.{jpg,png,...}` and `plant_dataset/Healthy/*`. The supplied data is at `../../ass-1/plant_dataset`. It stays outside the project; uploads are processed in memory, not saved by the application. The web UI uses Google Fonts with system-font fallbacks; inference remains local and works without Internet after setup.

## Models and interpretation of “from scratch”

- **Simple CNN:** three convolution/ReLU/max-pool blocks (3→16→32→64), flatten 64×28×28, Linear(128), ReLU, Dropout(0.3), Linear(2). Matches the Assignment 1 architecture.
- **ResNet-18:** stem, four residual stages with two BasicBlocks each (64/128/256/512), global average pooling, Dropout(0.3), Linear(2). Implemented directly in `app/models.py`, without torchvision's model constructor or pretrained classifier weights.
- Both classifiers are trained from **random initialization**, as requested. The A1 report used ImageNet-pretrained ResNet-18; this run is deliberately different, so its numbers must not be presented as reproduction of that transfer-learning experiment.
- Input: 224×224, ImageNet channel normalization. Training augmentation: flips, ±20° rotation, brightness/contrast jitter. Cross-entropy, Adam, learning rates 0.001 (CNN) and 0.0001 (ResNet).
- **MedSAM2:** official Wang Lab implementation and `MedSAM2_latest.pt`, paired with `configs/sam2.1_hiera_t512.yaml`. Its pretrained foundation-model weights are necessary; this project does not claim to train MedSAM2 from scratch. It uses its image predictor for single-image box-prompted segmentation. No color threshold or GrabCut substitute is used.
- Grad-CAM differentiates the predicted class logit with respect to the last convolution (CNN) or last residual stage (ResNet), globally averages gradients, applies ReLU, normalizes, and overlays at input-image resolution. A zero positive activation map is explicitly flagged.

## Reproducibility and evaluation

`scripts/train.py` scans actual image files, deduplicates decoded pixel content, rejects duplicates with conflicting labels, and makes a seeded stratified 70/15/15 split. Validation loss selects the checkpoint; test images are evaluated after training. `checkpoints/split.json` records exact membership. Seed 42 provides repeatability on the same environment, but GPU kernels and library versions may still vary.

Generated outputs:

- `checkpoints/{simple_cnn,resnet18}.pt`: model parameters plus architecture/class-order metadata.
- `checkpoints/*_metrics.json`: test accuracy, macro F1, per-class metrics, confusion matrix (rows=true, columns=predicted; Diseased then Healthy), training/validation curves, selected epoch.
- `artifacts/training.log`: training record.
- `artifacts/roi_evaluation.json`: paired full-image vs MedSAM2 ROI ablation, when generated.

```bash
# Tests, including malformed uploads and real architecture/Grad-CAM checks
.venv/bin/python -m pytest -q

# Held-out comparison with/without MedSAM2 (same images, same classifier weights)
.venv/bin/python scripts/evaluate_roi.py

# Optional augmentation ablation; isolated output so deployed weights are preserved
.venv/bin/python scripts/train.py --data '../../ass-1/plant_dataset' --no-augmentation --output artifacts/no_augmentation

# True mask quality requires human ground-truth masks with matching PNG filenames
.venv/bin/python scripts/evaluate_masks.py --predictions /path/to/predictions --references /path/to/reference_masks
```

No reference leaf masks were supplied, so true segmentation IoU/Dice cannot be claimed. MedSAM2's predicted mask score is not measured IoU. Near-duplicate photos are not detected by exact-content deduplication; grouping by plant/capture session would strengthen evaluation.

The main app shows current-run full-image metrics, curves, and the paired ROI ablation. One empty-mask failure is excluded from both sides of the paired comparison and recorded explicitly. Segmentation removes background but changes the classifier's input distribution; it can decrease accuracy, especially without ROI-specific training. Do not assume the SOTA component improves classification.

## API

- `GET /api/status`: checkpoint availability and compute device.
- `GET /api/metrics`: trained-model evaluation summaries.
- `GET /api/roi-metrics`: paired MedSAM2 ablation, including segmentation failures.
- `POST /api/analyze`: multipart `file`, `model` (`simple_cnn` or `resnet18`), `segment` (boolean), optional `box` (JSON `[x1,y1,x2,y2]`, normalized 0–1 coordinates).
- `GET /docs`: interactive FastAPI documentation.

Responses include class probabilities, mask coverage, timings, original/input/mask/Grad-CAM PNG data URLs, and method limitations. The model is loaded lazily; first MedSAM2 inference includes its loading cost. Inference is serialized to protect the predictor's mutable image embedding and Grad-CAM hooks. Missing checkpoints or model errors produce explicit errors, never random-weight or synthetic results.

Environment overrides: `PLANT_DEVICE`, `MEDSAM2_REPO`, `MEDSAM2_CHECKPOINT`, `MEDSAM2_CONFIG`. Restart the server after retraining to reload cached models.

## Limitations and responsible use

This is an educational research prototype. The dataset is small and does not establish performance across species, locations, lighting, or diseases. Any image receives one of two labels; there is no out-of-distribution/non-leaf detector. Confidence is an uncalibrated softmax score. Grad-CAM indicates classifier influence, not proven disease lesions. MedSAM2 was trained on medical imagery, not plant leaves; manually inspect masks and use a tight prompt for cluttered or multiple-leaf scenes. Uploaded images are not retained; use images you have permission to process. No synthetic data is generated. The app does not make treatment recommendations.

## Sources and licenses

- Assignment 1 report and notebook: supplied locally; used to identify architectures and preprocessing, not to copy historical metrics into this run.
- Assignment 2 PDF: used as project context; app implementation does not constitute the separate report/video deliverables.
- Official MedSAM2: https://github.com/bowang-lab/MedSAM2 (Apache-2.0; retain upstream notices in `vendor/MedSAM2`). Tested revision: `332f30d420f1d1b08e2a79b3ae6a602458808383`.
- Checkpoint: https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt
- MedSAM2 paper: https://arxiv.org/abs/2504.03600
- Grad-CAM: https://arxiv.org/abs/1610.02391

Large weights, the Python environment, and the downloaded upstream repo are ignored by Git. Keep the local checkpoint files with a submission bundle or recreate them using the commands above. The setup script pins the tested upstream revision.

## Measured results from this workspace

Ten training epochs per classifier, CPU. On all 41 held-out full images both models achieved 36/41 = **87.8% accuracy** (macro F1: CNN 0.8778; ResNet-18 0.8780). These are small-sample estimates.

The automatic MedSAM2 prompt produced nonempty masks on 40/41 images. On those same 40 images, full-image macro F1 was 0.8749 for both models. Masked-ROI macro F1 fell to **0.4000** for CNN and **0.5833** for ResNet-18 (accuracies 55% and 60%). This is a negative ablation result, not an improvement claim. The medical-domain segmentation model and the input distribution change need further adaptation before treating ROI classification as reliable.

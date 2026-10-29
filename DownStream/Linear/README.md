# ROI-Pan-28 reproduction

The anonymized release contains 53,755 tiles in 28 classes, with five fixed
image-level splits (seeds 2021–2025): 37,630 train / 5,375 validation / 10,750 test.
Use the released ordered manifests; UUID filename order must not regenerate the splits.

Request access to the [tile release](https://huggingface.co/datasets/Rainy-fog/Cervical-Histopathology-Tile-Classification)
and [CerS-Path backbone](https://huggingface.co/Rainy-fog/CerS-Path) separately.
Data and weights retain their Hugging Face access conditions.

## Run

Reference environment: Linux, Python 3.11, PyTorch 2.4.1, torchvision 0.19.1,
timm 1.0.22 and CUDA 12.1 (A6000/RTX4090). The archive pins reference package versions.
RTX5090 requires a compatible CUDA13/PyTorch stack; retain its compatible
torch/torchvision versions when installing the remaining requirements.

From a fresh working directory on the reference CUDA 12.1 stack:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf auth login

hf download Rainy-fog/Cervical-Histopathology-Tile-Classification \
  --repo-type dataset --revision fffd485caa01207d1b6d5f5acd6dddb49cae16ff \
  --local-dir roi28_release
hf download Rainy-fog/CerS-Path weights/CerS-Path.pth \
  --revision c246c7bbaa854e1615cc0539b5da7f27ab4ecd97 --local-dir backbone

python -m zipfile -e roi28_release/data/ROI-Pan-28_anonymized.zip .
python -m zipfile -e roi28_release/reproduction.zip .
python -m pip install -r ROI-Pan-28_reproduction/requirements.txt

python ROI-Pan-28_reproduction/prepare_run.py \
  --dataset-root ./ROI-Pan-28_anonymized \
  --backbone ./backbone/weights/CerS-Path.pth \
  --output-dir ./roi28_run --tmp-root /tmp/roi28
python -u roi28_run/parallel_resume.py --root ./roi28_run --run-id roi28
```

The launcher runs all five seeds on idle GPUs with at least 12 GB free.
Use a fresh output directory. Logs, predictions, per-seed metrics and the aggregate
report are saved under `roi28_run/`; `COMPLETE` marks successful completion.

## Fixed recipe

| Setting | Value |
|---|---|
| Model | Frozen CerS-Path ViT-L/16 encoder; train a 1024-to-28 linear head |
| Optimization | Adam, LR 1e-4, weight decay 0, cross-entropy |
| Training | Batch 128, 140 epochs, 8 workers |
| Checkpoint selection | Highest validation ACC |
| Scheduler | Historical ReduceLROnPlateau(mode=min, factor=0.1, patience=20, cooldown=10), stepped with validation ACC |
| Final test | Separate deterministic test of the selected checkpoint, batch 32 |
| Snapshots | Cumulative validation-best heads at epochs 20, 40, 60, 80, 100, 120, 140 |

Training and validation retain the recovered random augmentations. Final testing
uses RGB, Resize(224), ImageNet normalization and no random augmentation.

The released seed2021 / epoch52 head is the previously chosen demonstration after
exploratory test comparison. Across-seed selection by validation ACC chose seed2025;
report all five seeds for the standard reproduction. Per-seed and cross-device
results are in the release's `results.json`.

These historical image-level holdouts do not establish patient/slide independence.
The replay does not exactly recover the historical Source Data metrics.

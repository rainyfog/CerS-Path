# ROI-Pan-28 reproduction

Request access to the [data and reproduction pack](https://huggingface.co/datasets/Rainy-fog/Cervical-Histopathology-Tile-Classification)
and [backbone](https://huggingface.co/Rainy-fog/CerS-Path).
Requires Linux, Python 3.11, CUDA 12.1 and an authenticated `hf` CLI.

```bash
hf download Rainy-fog/Cervical-Histopathology-Tile-Classification \
  --repo-type dataset --revision fffd485caa01207d1b6d5f5acd6dddb49cae16ff --local-dir roi28
hf download Rainy-fog/CerS-Path weights/CerS-Path.pth \
  --revision c246c7bbaa854e1615cc0539b5da7f27ab4ecd97 --local-dir backbone
python -m zipfile -e roi28/data/ROI-Pan-28_anonymized.zip .
python -m zipfile -e roi28/reproduction.zip .
python -m pip install -r ROI-Pan-28_reproduction/requirements.txt
python ROI-Pan-28_reproduction/prepare_run.py \
  --dataset-root ./ROI-Pan-28_anonymized \
  --backbone ./backbone/weights/CerS-Path.pth \
  --output-dir ./roi28_run --tmp-root /tmp/roi28
python -u roi28_run/parallel_resume.py --root ./roi28_run --run-id roi28
```

Use the bundled ordered manifests and a fresh output directory. Freeze the encoder;
train the linear head with Adam, LR 1e-4, batch128, 140 epochs, seeds2021–2025.
Select checkpoints by validation ACC; final independent test uses batch32.
The requirements target CUDA12.1; RTX5090 must retain compatible torch/torchvision versions.
These image-level splits do not establish patient/slide independence.

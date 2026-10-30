# Cervical tile classification

Install a PyTorch build compatible with your device, then:

```bash
pip install -r requirements.txt
hf auth login
python train_val.py
python predict.py --input images/ --checkpoint output/classifier.pt
```

Request access to the [CerS-Path model and access instructions](https://huggingface.co/Rainy-fog/CerS-Path).
The model card provides the associated gated dataset entry point; model and dataset access
are reviewed separately. Training downloads the data and backbone automatically. Use `--data` for your own
directory with one subdirectory per class, or `--backbone` for local weights.

Defaults: frozen encoder, Adam, learning rate 1e-4, batch size 128, 140 epochs,
seed 2021, 70/10/20 split. Checkpoints are selected by validation accuracy.
All command-line parameters can be changed.

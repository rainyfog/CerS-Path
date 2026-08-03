# CerS-Path

Research code accompanying:

> **A Subspecialty Diagnostic System Powered by Self-Supervised Learning for Cervical Histopathology**

CerS-Path is a cervical subspecialty pathology system with two complementary
components:

- **CerS-V** for cervical-specific visual representation learning.
- **CerS-M** for multimodal pathology analysis based on the CerS-V visual
  encoder and Qwen2.5-VL.

The system is evaluated across whole-slide image (WSI), region-of-interest
(ROI), segmentation, zero-shot, multimodal description and prognostic tasks.

> [!IMPORTANT]
> This repository is intended for research use only and is not a clinical
> diagnostic product.

## Repository structure

| Component | Location | Description |
|---|---|---|
| Visual pretraining | [`CerS-V/`](CerS-V/) | Cervical-specific self-supervised visual learning |
| Multimodal training | [`CerS-M/`](CerS-M/) | Multimodal model preparation, training and inference |
| Image/video RAG demo | [`CerS-M/RAG-Video-Demo/`](CerS-M/RAG-Video-Demo/) | Local image/video inference with optional retrieval augmentation |
| ROI analysis | [`DownStream/Linear/`](DownStream/Linear/) | ROI-level classification and evaluation |
| WSI analysis | [`DownStream/MIL/`](DownStream/MIL/) | Weakly supervised WSI workflows |
| Segmentation | [`DownStream/Segment/`](DownStream/Segment/) | Lesion and tumour segmentation |
| Zero-shot evaluation | [`DownStream/Zero-shot/`](DownStream/Zero-shot/) | Zero-shot classification workflow |
| Multimodal description | [`DownStream/MM-description/`](DownStream/MM-description/) | Multimodal description workflow |

Installation, configuration and task-specific instructions are provided in the
README files within the corresponding directories.

## Quick start

```bash
git clone https://github.com/rainyfog/CerS-Path.git
cd CerS-Path

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The main tested environment uses Python 3.10 and PyTorch 2.5.1. A CUDA-capable
Linux environment is recommended for training and multimodal inference.

For CerS-M setup and usage, see [`CerS-M/README.md`](CerS-M/README.md). For the
local image/video interface and optional RAG workflow, see
[`CerS-M/RAG-Video-Demo/README.md`](CerS-M/RAG-Video-Demo/README.md).

## Reproducibility notes

- Configure local data and output paths before running an experiment.
- Keep all material from the same case within a single data partition.
- Hardware, package versions and dataset partitions may affect reproduced
  results.

## Citation

Please cite the associated manuscript:

> Wang, Y. *et al.* **A Subspecialty Diagnostic System Powered by
> Self-Supervised Learning for Cervical Histopathology.** Manuscript under
> revision.

A `CITATION.cff` file and a version-specific DOI will be added for the archival
release.

## License

The project code is intended to be released under the MIT License in the
archival release. Third-party code and datasets remain subject to their original
licenses and terms of use.

## Acknowledgements

This repository builds on or references:

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [CONCH](https://github.com/mahmoodlab/CONCH)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

For questions about the code, please open a GitHub issue.

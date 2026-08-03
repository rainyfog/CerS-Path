# CerS-Path

Research code accompanying:

> **A Subspecialty Diagnostic System Powered by Self-Supervised Learning for Cervical Histopathology**

CerS-Path is a cervical subspecialty pathology system developed through two complementary stages:

1. **CerS-V** — cervical-specific visual representation learning following the DINOv2 self-supervised learning framework.
2. **CerS-M** — multimodal enhancement by integrating the visual encoder with Qwen2.5-VL and fine-tuning on pathology image-text data.

The resulting representations are evaluated across whole-slide image (WSI), region-of-interest (ROI), segmentation, zero-shot, multimodal description and prognostic tasks.

> [!IMPORTANT]
> This repository is a research release rather than a clinical product. It is not intended for direct clinical diagnosis or treatment decisions.

## Repository status

This repository contains the code available for the manuscript revision. Some experiment scripts preserve the original cluster paths for traceability and must be configured before use. Patient-level data, in-house WSIs, clinical metadata and trained model checkpoints are not included.

| Component | Location | Current contents |
|---|---|---|
| Visual pretraining | [`CerS-V/`](CerS-V/) | Reference to the DINOv2-based visual pretraining procedure |
| Multimodal training | [`CerS-M/`](CerS-M/) | Custom visual encoder integration, LoRA/full fine-tuning and inference code |
| Image/video RAG demo | [`CerS-M/RAG-Video-Demo/`](CerS-M/RAG-Video-Demo/) | Local multimodal inference with an optional Milvus retrieval interface |
| ROI linear probing | [`DownStream/Linear/`](DownStream/Linear/) | Frozen-encoder classification and evaluation |
| WSI weak supervision | [`DownStream/MIL/`](DownStream/MIL/) | Reference to the CLAM-based workflow |
| Segmentation | [`DownStream/Segment/`](DownStream/Segment/) | PFM-based segmentation models and training code |
| Zero-shot evaluation | [`DownStream/Zero-shot/`](DownStream/Zero-shot/) | Reference implementation used for zero-shot evaluation |
| Multimodal description | [`DownStream/MM-description/`](DownStream/MM-description/) | Reference implementation for the interactive inference interface |

## System overview

CerS-Path was developed using the CerS-140K cervical pathology collection:

- approximately 115,000 WSIs for self-supervised visual pretraining;
- approximately 13,000 WSIs for independent downstream benchmarking; and
- approximately 12,000 WSIs for clinical-deployment refinement and aggregator training.

The visual pretraining stage used approximately 190 million tissue patches. Multimodal enhancement used approximately 2.5 million pathology image-text pairs. Full cohort composition, eligibility criteria and task-specific sample sizes are described in the associated manuscript and Supplementary Information.

## Requirements

The main tested software versions are listed in [`requirements.txt`](requirements.txt):

- Python 3.10
- PyTorch 2.5.1
- torchvision 0.20.1
- torchaudio 2.5.1
- Triton 3.1.0
- timm 1.0.11
- transformers 4.48.0

A CUDA-capable Linux environment is recommended. Full visual or multimodal training requires substantially more GPU memory than downstream linear probing.

### Core environment

```bash
git clone https://github.com/rainyfog/CerS-Path.git
cd CerS-Path

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### CerS-M environment

For multimodal training, use the more complete Conda environment:

```bash
conda env create -f CerS-M/environment.yaml
conda activate qwen2-CL

pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

Install FlashAttention only after PyTorch and the remaining dependencies are available.

## Required configuration

Before running an experiment:

1. Replace the original site-specific paths in:
   - `CerS-M/scripts/*.sh`
   - `DownStream/Linear/train_val.py`
   - `DownStream/Segment/training.py`
   - `DownStream/Segment/PFM_Seg_Models.py`
2. Download the required base models or pathology foundation model weights under their original licenses and access conditions.
3. Set all dataset, checkpoint and output paths to local locations.
4. Keep every slide or image tile from the same case in only one data partition.
5. Do not place restricted clinical data, credentials or model-access tokens inside the repository.

## Usage

### 1. Multimodal model preparation and fine-tuning

The detailed workflow is provided in [`CerS-M/README.md`](CerS-M/README.md). In brief:

1. Prepare a local Qwen2.5-VL base model.
2. Set `vision_config.pretrained_model_path` to the CerS-V visual checkpoint.
3. Merge the custom visual encoder:

```bash
cd CerS-M

python src/merge_custom_vit.py \
  --model-base /path/to/qwen2_5_vl_base \
  --save-model-path /path/to/qwen2_5_with_cers_v \
  --safe-serialization
```

4. Update the data, image, model and output paths in the training scripts.
5. Run LoRA or full fine-tuning, for example:

```bash
bash scripts/finetune_lora_mlp.sh
```

The expected training data follow the conversation-style multimodal format handled in `CerS-M/src/training/data.py`.

### 2. ROI linear probing

The linear-probe entry point is [`DownStream/Linear/train_val.py`](DownStream/Linear/train_val.py). The current script uses single-dash argument names:

```bash
python DownStream/Linear/train_val.py \
  -train_dir /path/to/dataset \
  -model_name Cervix \
  -model_path /path/to/cers_v_checkpoint \
  -num_classes 28 \
  -save_root ./outputs/linear \
  -seed 2025
```

`Cervix` is the model identifier used by the current script for the CerS-Path visual encoder. Other implemented identifiers include `UNI`, `CONCHV15` and `virchow2`; their weights must be obtained separately.

### 3. Video or moving-field multimodal training

Video-capable multimodal training is implemented in [`CerS-M/scripts/finetune_video.sh`](CerS-M/scripts/finetune_video.sh). The corresponding data loader, visual-token forward path and interactive inference support are implemented in:

- `CerS-M/src/training/data.py`
- `CerS-M/src/training/monkey_patch_forward.py`
- `CerS-M/src/training/params.py`
- `CerS-M/src/serve/app.py`

A minimal training record contains a local video path and an English target description:

```json
[
  {
    "video": "example_case.mp4",
    "en_caption": "A de-identified pathology video example for testing the input pipeline."
  }
]
```

Update `--data_path` and `--image_folder` in `finetune_video.sh`, then run from the `CerS-M` directory:

```bash
cd CerS-M
bash scripts/finetune_video.sh
```

The repository contains code only. Do not upload clinical videos or frame
sequences. Before reproducing the manuscript experiment, also confirm the
Qwen2.5-VL model size and DeepSpeed configuration against the final experiment
record.

For local image/video inference and the optional RAG workflow, see
[`CerS-M/RAG-Video-Demo/`](CerS-M/RAG-Video-Demo/). Its server paths, model
paths and retrieval database are supplied at runtime and are not stored in the
repository.

### 4. Segmentation

Segmentation code is located in [`DownStream/Segment/`](DownStream/Segment/). The current training script uses a configuration block rather than a complete command-line interface. Update the image, mask and checkpoint paths in `training.py`, then run:

```bash
cd DownStream/Segment
python training.py
```

Supported encoder branches in `PFM_Seg_Models.py` include UNI, Virchow2 and CONCH v1.5, together with the project-specific visual encoder configuration used in the experiments.

### 5. WSI, zero-shot and multimodal-description tasks

- WSI-level weakly supervised experiments follow the [CLAM](https://github.com/mahmoodlab/CLAM) framework.
- Zero-shot evaluation follows the design of the [CONCH zero-shot example](https://github.com/mahmoodlab/CONCH/blob/main/notebooks/zeroshot_classification_example_ensemble.ipynb).
- The cleaned multimodal inference interface, including video input and optional RAG, is provided in [`CerS-M/RAG-Video-Demo/`](CerS-M/RAG-Video-Demo/).

These upstream repositories are not vendored here and remain subject to their respective licenses.

## Data and model availability

No raw in-house WSI or associated clinical metadata are distributed through this repository because the authors are not authorized to share them under the applicable ethics approvals and patient-privacy requirements.

Public datasets used in the study should be obtained from their original repositories, including:

- TissueNet — <https://doi.org/10.60597/eaqa-k904>
- TCGA-CESC — <https://portal.gdc.cancer.gov/>
- UCSC Xena — <https://xenabrowser.net/>
- ROI-CIN-9/LDCH — see the dataset source cited in the manuscript

Third-party foundation model weights, including Qwen2.5-VL, DINOv2, UNI, Virchow2 and CONCH/CONCH v1.5, are not redistributed. Users are responsible for complying with the original model licenses and access requirements.

## Reproducibility notes

- Classification, segmentation and survival experiments used five predefined cross-validation folds or random seeds where applicable.
- The manuscript reports fold-level or cohort-level source data for the corresponding figures and tables.
- Site-specific paths in the current scripts are historical experiment paths and do not indicate that the underlying private data are publicly accessible.
- Hardware, package versions, random seeds and dataset partitions can affect the reproduced values.

## Citation

Please cite the associated manuscript:

> Wang, Y. *et al.* **A Subspecialty Diagnostic System Powered by Self-Supervised Learning for Cervical Histopathology.** Manuscript under revision.

A `CITATION.cff` file and the version-specific Zenodo DOI will be added for the archival `v1.0.0` release.

## License

The project code is intended to be released under the MIT License in the archival release. Third-party code, models and datasets remain subject to their original licenses and terms of use.

## Acknowledgements

This repository builds on or references:

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [CONCH](https://github.com/mahmoodlab/CONCH)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

For questions about the code, please open a GitHub issue.

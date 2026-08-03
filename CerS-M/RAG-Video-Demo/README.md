# CerS-M RAG and video inference demo

This directory contains a cleaned, publication-ready version of the interactive
CerS-M inference code. It supports:

- pathology image input;
- pathology video or moving-field input;
- multi-turn text interaction; and
- optional retrieval-augmented generation (RAG) with a local Milvus Lite
  database.

The workflow first produces a multimodal model response. When RAG is enabled, it
uses the question and initial response as a retrieval query and then asks the
model to refine its answer using the retrieved text.

No model weights, retrieval database, patient data, clinical media, credentials,
or institution-specific paths are included.

## Install

Create the CerS-M environment described in the parent directory, then install
the demo dependencies:

```bash
cd CerS-M
pip install -r RAG-Video-Demo/requirements.txt
```

## Run without RAG

For a merged model:

```bash
python RAG-Video-Demo/app.py \
  --model-path /path/to/merged_model
```

For a LoRA checkpoint:

```bash
python RAG-Video-Demo/app.py \
  --model-path /path/to/lora_checkpoint \
  --model-base /path/to/base_model
```

The interface listens on `127.0.0.1:7860` by default. The `--share` option is
disabled by default and should not be enabled when handling restricted clinical
material.

## Run with RAG

```bash
python RAG-Video-Demo/app.py \
  --model-path /path/to/merged_model \
  --rag-db /path/to/knowledge_base.db \
  --rag-collection cervix \
  --embedding-model BAAI/bge-m3
```

The Milvus collection must contain:

| Field | Purpose |
|---|---|
| `dense_vector` | BGE-M3 dense embedding |
| `sparse_vector` | BGE-M3 sparse embedding |
| `text` | Retrieved reference text |
| `summary` | Optional reference summary |

The retrieval database must be built locally from material that the user is
authorized to process. This repository does not provide a clinical corpus.

## Input notes

- Supported video extensions include MP4, MOV, MKV, AVI, WebM and MPEG.
- Video frames are processed through the Qwen-VL video utility used by CerS-M.
- Uploaded files are handled by the local Gradio process; do not expose the
  service publicly unless the inputs are approved for that environment.
- Outputs are research results and require qualified human review.


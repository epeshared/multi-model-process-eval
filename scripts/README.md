# Scripts: Embedding Runs

This folder contains small shell wrappers around the Python entrypoints in `scripts/py/`.

## Yahoo Answers embedding

Script: `embedding/run_embedding_yahoo.sh`

### Yahoo: What it does

Runs text embeddings on the Yahoo Answers JSONL dataset via `scripts/py/run_embedding.py`.

### Yahoo: Inputs

- `DATASET_PATH`: path to `yahoo_answers_title_answer.jsonl` (or compatible JSONL)

You can also pass the dataset path as the first positional argument:

```bash
./embedding/run_embedding_yahoo.sh /path/to/yahoo_answers_title_answer.jsonl
```

### Yahoo: Common env vars

- `BACKEND` (default: `sglang-offline`)  
  Supported: `torch`, `sglang`, `sglang-offline`, `vllm`, `vllm-http`
- `MODEL` (default: `qwen3-embedding-4b`)
- `MODEL_ID` (default: `/home/xtang/models/Qwen/Qwen3-Embedding-4B`)
- `YAHOO_MODE` (default: `q`)  
  `q` | `a` | `q+a`
- `MAX_SAMPLES` (default: `1000`)
- `BATCH_SIZE` (default: `100`)
- `DEVICE` (default: `cpu`)
- `DTYPE` (default: `bfloat16`)

Backend-specific:

- If `BACKEND=sglang` (HTTP):
  - `BASE_URL` (example: `http://127.0.0.1:30000`)

Perf:

- `WARMUP_SAMPLES` (default: `1`)  
  If `>1`, runs a warmup embedding call on the first N samples (not included in timing).

CPU AMX (torch backend only):

- `USE_AMX` (default: `TRUE`)  
  When truthy, passes `--use-amx` to enable IPEX/AMX for torch+cpu embedding.

### Yahoo: Examples

Torch CPU + AMX:

```bash
BACKEND=torch DEVICE=cpu USE_AMX=1 WARMUP_SAMPLES=100 \
  ./embedding/run_embedding_yahoo.sh /home/xtang/datasets/yahoo_answers_title_answer.jsonl
```

SGLang HTTP (server must be running):

```bash
BACKEND=sglang BASE_URL=http://127.0.0.1:30000 WARMUP_SAMPLES=100 \
  ./embedding/run_embedding_yahoo.sh /home/xtang/datasets/yahoo_answers_title_answer.jsonl
```


## Flickr8k embedding (text + image)

Script: `embedding/run_embedding_flickr8k.sh`

### Flickr8k: What it does

Runs Flickr8k caption text embeddings and/or image embeddings (CLIP-style multimodal) via `scripts/py/run_embedding.py`.

### Flickr8k: Inputs

You can pass positional arguments:

```bash
./embedding/run_embedding_flickr8k.sh /path/to/Flickr8k.token.txt /path/to/Flicker8k_Dataset
```

Or set env vars:

- `FLICKR8K_CAPTIONS_FILE`: path to `Flickr8k.token.txt`
- `FLICKR8K_IMAGES_DIR`: path to the image directory

### Flickr8k: Common env vars

- `BACKEND` (default: `torch`)  
  Supported: `torch`, `sglang`, `sglang-offline`, `vllm`, `vllm-http`
- `MODEL` (default: `clip-vit-base-patch32`)
- `MODEL_ID` (default: `/home/xtang/models/openai/clip-vit-base-patch32`)
- `DEVICE` (default: `cpu`)
- `DTYPE` (default: unset)

Backend-specific:

- If `BACKEND=sglang` (HTTP):
  - `BASE_URL` (default: `http://127.0.0.1:30000`)
  - `API` (default: `v1`) (image embedding recommends `v1`)
  - `API_KEY` (default: empty)
  - `IMAGE_TRANSPORT` (default: `data-url`) (`data-url` | `base64` | `path/url`)

Flickr8k dataset options:

- `FLICKR8K_MODALITY` (default: `both`)  
  `both` | `text` | `image`
- `CAPTIONS_PER_IMAGE` (default: `1`)
- `MAX_SAMPLES` (default: `-1`)  
  Max images to use.

Perf:

- `WARMUP_SAMPLES` (default: `1`)  
  If `>1`, runs warmup separately for text and image.

### Flickr8k: Examples

SGLang HTTP, run both text+image:

```bash
BASE_URL=http://127.0.0.1:30000 BACKEND=sglang API=v1 \
FLICKR8K_MODALITY=both CAPTIONS_PER_IMAGE=1 MAX_SAMPLES=1000 WARMUP_SAMPLES=100 \
  ./embedding/run_embedding_flickr8k.sh /path/to/Flickr8k.token.txt /path/to/Flicker8k_Dataset
  
```

Text-only (does not require images to exist):

```bash
BACKEND=sglang BASE_URL=http://127.0.0.1:30000 FLICKR8K_MODALITY=text \
  ./embedding/run_embedding_flickr8k.sh /path/to/Flickr8k.token.txt
```

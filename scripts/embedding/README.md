# Embedding scripts

This folder contains convenience scripts for running text/image embedding benchmarks via `scripts/embedding/run_embedding.py`.

## Scripts

- `run_embedding_yahoo.sh`
  - Runs embeddings on the Yahoo Answers dataset (text).
- `run_embedding_flickr8k.sh`
  - Runs embeddings on Flickr8k (text and/or image, depending on options).
- `run_fix_token_len.sh`
  - Runs embeddings on a synthetic dataset with controlled input sizes.
  - `MODE=input_len` (default): fixed **character** length (`SYNTHETIC_INPUT_LEN`).
  - `MODE=token_len`: fixed **token** length (`SYNTHETIC_TOKEN_LEN`).

## Common environment variables

Most scripts support these overrides:

- `MODEL` / `MODEL_ID`
  - `MODEL` is a logical name used by the wrapper script.
  - `MODEL_ID` is the actual model identifier:
    - offline backends: local model directory or HF repo id
    - `vllm-http`: must match the vLLM OpenAI server `--served-model-name`.
- `BACKEND`
  - `torch` (local torch inference)
  - `sglang` (HTTP)
  - `sglang-offline`
  - `vllm` (offline)
  - `vllm-http`
- `BASE_URL`
  - for HTTP backends (`sglang`, `vllm-http`), e.g. `http://127.0.0.1:9090`
- `DEVICE`, `DTYPE`
  - used by some backends/paths (torch/offline)
- `BATCH_SIZE`, `MAX_SAMPLES`

## Fixed-length synthetic test (`run_fix_token_len.sh`)

This is intended to test throughput/latency under controlled input sizes.

Examples:

- Offline torch (CPU) with fixed 512-char inputs:

  ```bash
  MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 BACKEND=torch DEVICE=cpu \
    ./run_fix_token_len.sh
  ```

- Offline torch (CPU) with fixed 20-token inputs:

  ```bash
  MODE=token_len SYNTHETIC_TOKEN_LEN=20 MAX_SAMPLES=10000 BACKEND=torch DEVICE=cpu \
    ./run_fix_token_len.sh
  ```

- vLLM OpenAI server (HTTP):

  1) Start vLLM server (example port 9090):

  ```bash
  # In another terminal
  # See scripts/embedding/vllm/start_vllm_server.sh for embedding server startup.
  PORT=9090 ./vllm/start_vllm_server.sh
  ```

  2) Run client:

  ```bash
  BASE_URL=http://127.0.0.1:9090 BACKEND=vllm-http \
    SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 \
    ./run_fix_token_len.sh
  ```

Notes:
- The vLLM embedding endpoint is OpenAI-compatible (`/v1/embeddings`).
- For `vllm-http`, `MODEL_ID` must match the server's `--served-model-name`.

## Profiling

Some wrappers can pass profiling flags through to `run_embedding.py`:

- `PROFILE=1` enables `--profile`
- `PROFILE_RECORD_SHAPES=1`
- `PROFILE_ACTIVITIES=CPU,CUDA`
- `PROFILE_OUT_DIR`, `PROFILE_OUT_NAME`

(Profiling support depends on the backend; see `run_embedding.py --help` and backend implementations.)

## MTEB

There is an optional MTEB integration under `scripts/embedding/mteb`.

- Install: `pip install -r requirements.txt -r requirements-mteb.txt`
- Run: `python scripts/embedding/mteb/run_mteb.py --help`

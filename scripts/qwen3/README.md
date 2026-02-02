# Qwen3 test scripts

- `run_qwen3_test.sh`: Synthetic text stress-test for Qwen3 via `scripts/qwen3/run_qwen3.py`.

## Start servers

### vLLM (OpenAI-compatible)
```bash
cd scripts/qwen3/vllm
MODEL_DIR=/path/to/Qwen3-0.6B SERVED_MODEL_NAME=qwen3-0.6b ./start_vllm_server.sh
```

### SGLang
```bash
cd scripts/qwen3/sglang
MODEL_DIR=/path/to/Qwen3-0.6B ./start_sglang_server.sh
# CUDA:
MODEL_DIR=/path/to/Qwen3-1.7B ./start_sglang_server_cuda.sh
```

## Usage
```bash
# vLLM HTTP (default)
MODEL=qwen3-0.6b MODEL_ID=/mnt/models/Qwen/Qwen3-0.6B BASE_URL=http://127.0.0.1:8000 ./run_qwen3_test.sh

# SGLang HTTP
MODEL=qwen3-1.7b MODEL_ID=/mnt/models/Qwen/Qwen3-1.7B BACKEND=sglang BASE_URL=http://127.0.0.1:30000 ./run_qwen3_test.sh

# Tune synthetic dataset
SYNTHETIC_NUM_PROMPTS=50 SYNTHETIC_TOKEN_LEN=64 BATCH_SIZE=2 MAX_NEW_TOKENS=128 ./run_qwen3_test.sh
```

## vLLM built-in style benchmark (P50/P99/QPS/TTFT/TPOT)

If you want serving-style metrics (QPS + latency percentiles) directly against the vLLM **OpenAI-compatible** server,
use the benchmark script under `scripts/qwen3/vllm`.

### 1) Start vLLM server

```bash
cd scripts/qwen3/vllm
MODEL_DIR=/path/to/Qwen3-0.6B SERVED_MODEL_NAME=qwen3-0.6b ./start_vllm_server.sh
```

### 2) Run benchmark

```bash
cd scripts/qwen3/vllm

# Quick run: stream=true enables TTFT; TPOT needs streamed usage support
BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-0.6b NUM_PROMPTS=200 CONCURRENCY=16 MAX_TOKENS=256 \
	./run_benchmark_openai_server.sh

# Save raw per-request results
SAVE_JSON=./results/qwen3_bench.json ./run_benchmark_openai_server.sh
```

Notes:

- `MODEL` must match the server's `--served-model-name`.
- TTFT requires `STREAM=1` (default).
- TPOT requires the server to stream `usage` (best-effort via `stream_options={"include_usage": true}`); otherwise TPOT will be `null`.

## vLLM built-in benchmark (official `vllm.benchmarks.*`)

If you want to run vLLM's own benchmark module directly (instead of this repo's client-side metrics script),
use the wrapper below. It auto-detects CLI flag names across vLLM versions and forwards extra args.

```bash
cd scripts/qwen3/vllm

# Requires a running vLLM OpenAI-compatible server
BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-0.6b NUM_PROMPTS=200 CONCURRENCY=16 \
	IN_LEN=256 OUT_LEN=256 REQUEST_RATE=0 \
	./run_vllm_builtin_benchmark_serving.sh

# Pass through extra vLLM benchmark args after --
./run_vllm_builtin_benchmark_serving.sh -- --seed 0
```

## TTFT / TPOT

This repo prints TTFT/TPOT in `scripts/qwen3/run_qwen3_test.sh` (two extra summary lines) and also includes them in the JSON output from `scripts/qwen3/run_qwen3.py`.

### Definitions

- **TTFT (Time To First Token, seconds)**
	- Measured client-side using OpenAI-compatible **streaming** (`stream=true`).
	- Definition: wall-clock time from sending the HTTP request until receiving the first non-empty `delta.content` chunk.

- **TPOT (Time Per Output Token, seconds/token)**
	- Computed per request as:
    
		$$\text{TPOT} = \frac{\text{total\_sec} - \text{ttft\_sec}}{\max(1,\ \text{completion\_tokens} - 1)}$$
	- Requires the server to provide `usage.completion_tokens` (best-effort via `stream_options={"include_usage": true}` when streaming).
	- If `completion_tokens` is missing, TPOT is reported as `null`.

### What gets printed

- For `--dataset=synthetic`:
	- `ttft_sec_avg`: average of per-prompt TTFT
	- `tpot_sec_per_token_avg`: average of per-prompt TPOT
- For `--dataset=single`:
	- `ttft_sec`, `tpot_sec_per_token`

### Call flow (where the numbers come from)

1. `scripts/qwen3/run_qwen3_test.sh`
	 - Runs `python scripts/qwen3/run_qwen3.py ...` and captures its JSON output.
	 - Extracts `ttft_sec_avg`/`tpot_sec_per_token_avg` (or the single-run keys) and prints:
		 - `[run_qwen3_test] TTFT_sec=...`
		 - `[run_qwen3_test] TPOT_sec_per_token=...`

2. `scripts/qwen3/run_qwen3.py`
	 - Generates a synthetic text dataset (pseudo tokens like `w123 w456 ...`).
	 - Calls `src/tasks/qwen3.py:load_qwen3_session(...)` to build a reusable session.
	 - For each prompt (or prompt batch) it calls `session.chat_with_metrics(...)` when available.
	 - Aggregates TTFT/TPOT across prompts and prints a JSON summary.

3. HTTP clients (actual timing)
	 - SGLang: `src/tasks/qwen3_backends/sglang_http.py`
	 - vLLM: `src/tasks/qwen3_backends/vllm_http.py`
	 - Both implement an OpenAI-compatible SSE parser:
		 - Start timer at request start
		 - Set TTFT when the first `delta.content` arrives
		 - Track `usage` if the server sends it during streaming

### Notes / caveats

- TTFT/TPOT are **client-observed** metrics and include network + server queueing + decode time.
- If you disable streaming (`--no-stream`), TTFT will be `null` (no per-token timestamps).
- Some server versions do not stream `usage`; TPOT may be `null` even when TTFT exists.

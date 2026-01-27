# MTEB runner (container backends)

This folder integrates [MTEB](https://github.com/embeddings-benchmark/mteb) with the existing container embedding backends in this repo.

## Install

MTEB is an optional dependency.

```bash
pip install -r requirements.txt -r requirements-mteb.txt
```

## Run

Quick wrapper (env-driven, similar style to other scripts in this repo):

```bash
MODEL_ID=<served-model-name> \
BACKEND=vllm-http BASE_URL=http://127.0.0.1:9090 \
./scripts/embedding/mteb/run_mteb.sh
```

You can also override tasks:

```bash
MODEL_ID=<served-model-name> ./scripts/embedding/mteb/run_mteb.sh STSBenchmark
```

## 参数说明

下面把 `run_mteb.py`（Python CLI）和 `run_mteb.sh`（env wrapper）各自支持的参数含义写清楚。

### run_mteb.py（Python CLI）

后面示例里用到的命令：

```bash
python scripts/embedding/mteb/run_mteb.py --help
```

**后端 / 服务端连接**

- `--backend`：选择后端，当前支持 `vllm-http` / `sglang`。
- `--base-url`：HTTP 服务地址（不带路径），例如 `http://127.0.0.1:9090`。
- `--model-id`：服务端的模型名（必须和 server 对外暴露的 served-model-name 一致）。
- `--api`：仅对 `--backend sglang` 生效，用于选择 SGLang API 形态：
  - `v1`：OpenAI 兼容 `/v1/embeddings`
  - `native`：SGLang 原生接口
  - `openai`：通过 OpenAI SDK/兼容层
- `--api-key`：可选的 API key（如果服务端需要鉴权）。
- `--timeout`：HTTP 请求超时时间（秒）。
- `--encoding-format`：仅对 `--backend vllm-http` 生效，提示服务端返回 embedding 的编码格式偏好：`base64` 或 `float`。

**编码行为**

- `--batch-size`：一次请求/一次 encode 的 batch 大小（客户端侧）。
- `--no-normalize`：关闭 L2 normalize（默认会 normalize）。
- `--query-prefix`：当 MTEB 传入 `prompt_type=query` 时，为每条 query 文本添加的前缀。
- `--document-prefix`：当 `prompt_type=document` 时，为每条文档/段落添加的前缀。
- `--max-length`：当前对 HTTP 后端不生效（保留字段，和 repo 里 torch/offline 的接口保持一致）。

**Profiling（仅 sglang-http）**

- `--profile`：开启 profiling（会调用 SGLang 的 `/start_profile` 与 `/stop_profile`）。
- `--profile-kwargs`：传给 profiling 端点的参数（JSON 字符串或 JSON 文件路径）。

**任务选择**

- `--tasks`：逗号分隔的 task 名称列表，例如 `STSBenchmark,MSMARCO`。
- `--benchmark`：benchmark 名称（与 `--tasks` 二选一）。
- `--task-types`：按 task type 过滤（逗号分隔）。
- `--languages`：按语言过滤（逗号分隔），例如 `eng,zho`。
- `--domains`：按 domain 过滤（逗号分隔）。

**输出**

- `--output-folder`：MTEB cache / results 输出目录。
- `--overwrite`：删除已有结果 JSON 后强制重跑（否则 MTEB 默认可能会跳过已存在的结果）。

### run_mteb.sh（env wrapper）

`run_mteb.sh` 是对 `run_mteb.py` 的薄封装：通过环境变量提供默认值，并把剩余参数原样透传给 `run_mteb.py`。

**位置参数**

- `$1`（可选）：如果第一个参数不是以 `-`/`--` 开头，则会被当作 `TASKS`（逗号分隔任务名）。
  - 示例：`MODEL_ID=xxx ./scripts/embedding/mteb/run_mteb.sh STSBenchmark`

**环境变量（对应 run_mteb.py 的默认值）**

- `BACKEND`：默认 `vllm-http`。
- `BASE_URL`：默认 `http://127.0.0.1:9090`。
- `MODEL_ID`：必填（served-model-name）。
- `API`：默认 `v1`（仅 sglang）。
- `API_KEY`：默认空。
- `TIMEOUT`：默认 `120`。
- `ENCODING_FORMAT`：默认空；当 `BACKEND=vllm-http` 且未设置时，会自动设为 `base64`。

- `BATCH_SIZE`：默认 `128`。
- `MAX_LENGTH`：默认 `512`（当前对 HTTP 后端不生效）。
- `NORMALIZE`：默认 `1`；设为 `0/false/no/off` 等会自动追加 `--no-normalize`。
- `QUERY_PREFIX` / `DOCUMENT_PREFIX`：默认空。

- `TASKS`：默认 `STSBenchmark`。
- `TASK`：`TASKS` 的别名（只跑单个任务时写起来更短）；如果同时设置了 `TASKS`，以 `TASKS` 为准。
- `BENCHMARK`：默认空；与 `TASKS` 二选一。
- `TASK_TYPES` / `LANGUAGES` / `DOMAINS`：默认空（逗号分隔）。

- `PROFILE`：默认 `0`；设为 `1/true/yes/on` 会追加 `--profile`（仅 sglang 生效）。
- `PROFILE_KWARGS`：默认空；非空时会追加 `--profile-kwargs`。

- `OUTPUT_FOLDER`：默认 `scripts/embedding/mteb/results`。
- `OVERWRITE`：默认 `0`；设为 `1/true/yes/on` 会追加 `--overwrite`，强制重跑并刷新 `evaluation_time`/吞吐统计。

## 关于 evaluation_time

- `evaluation_time` 是 MTEB 写入的耗时数值（单位是 seconds），但它**只会在确实执行评测/写入新结果时更新**。
- 如果你重复运行同一个 task/model，而结果文件已存在，MTEB 可能会直接跳过计算（此时终端看起来很快），但 `evaluation_time` 仍然保留旧值。
- 想得到本次真实的耗时/吞吐，请用 `OVERWRITE=1` 或 `--overwrite` 强制重跑。

**透传规则**

- 你可以在命令末尾追加任何 `run_mteb.py` 支持的参数（例如 `--task-types ...`）。
- wrapper 会尽量避免重复追加你已经手动传入的 flag（例如你显式传了 `--no-normalize`，它就不会再加一次）。

**Python 解释器选择**

- wrapper 会优先使用仓库内的 `./.venv/bin/python`（如果存在），避免因为系统/conda 的 Python 版本差异导致的兼容性问题。
```

### vLLM HTTP embeddings

1) Start vLLM embedding server (example):

```bash
# In another terminal
PORT=9090 ./scripts/embedding/vllm/start_vllm_server.sh
```

2) Run MTEB:

```bash
python scripts/embedding/mteb/run_mteb.py \
  --backend vllm-http \
  --base-url http://127.0.0.1:9090 \
  --model-id <served-model-name> \
  --tasks STSBenchmark \
  --output-folder scripts/embedding/mteb/results
```

### SGLang HTTP embeddings

1) Start SGLang embedding server (example):

```bash
# In another terminal
PORT=9090 ./scripts/embedding/sglang/start_sglang_server.sh
```

2) Run MTEB:

```bash
python scripts/embedding/mteb/run_mteb.py \
  --backend sglang \
  --base-url http://127.0.0.1:9090 \
  --model-id <served-model-name> \
  --api v1 \
  --tasks STSBenchmark \
  --output-folder scripts/embedding/mteb/results
```

## Prompts / prefixes

Some embedding models expect different prefixes for retrieval queries vs documents.
You can set these via:

- `--query-prefix`
- `--document-prefix`

MTEB passes `prompt_type` as `query` or `document`, and the runner will prepend the configured prefix.

# multi-model-process-eval

提供一套统一的代码骨架，用 **torch / sglang / vLLM** 三类后端跑不同任务（embedding、LLM、VL、Omni）。本仓库以 `src/tasks/*` 作为任务逻辑层，以 `scripts/*` 作为可直接运行的脚本入口（含 server 启动脚本与压测脚本）。

## 快速导航（推荐从这里开始）

- 总入口与脚本说明：[`scripts/README.md`](scripts/README.md)
- Embedding（文本/图像向量 + 合成数据压测）：[`scripts/embedding/README.md`](scripts/embedding/README.md)
- Qwen3（LLM 压测 + TTFT/TPOT）：[`scripts/qwen3/README.md`](scripts/qwen3/README.md)
- Omni（多模态/音频等 Omni 模型压测）：[`scripts/omni/README.md`](scripts/omni/README.md)
- 工具脚本（如 FP8->FP16）：[`scripts/tools/README.md`](scripts/tools/README.md)

## 目录结构

- `src/data/`：数据集与输入构造（如 Yahoo / Flickr8k / synthetic）。
- `src/tasks/`：按任务划分的最小推理逻辑（embedding / qwen3 / vl / omni 等）及对应 backend 适配层。
- `scripts/`：可运行脚本入口与 server 启动脚本（sglang/vllm），按任务分目录。

## 安装

```bash
pip install -r requirements.txt
```

说明：不同任务/后端对环境要求不同，尤其是 `vllm` / `sglang`（CPU vs CUDA、openai 依赖版本等）。建议优先参考对应子目录 README 的“Start servers / Usage”。

## 常用示例

### Embedding：合成固定长度压测

详见 [`scripts/embedding/README.md`](scripts/embedding/README.md)。

```bash
cd scripts/embedding

# 固定字符长度（默认 MODE=input_len）
MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 BACKEND=torch DEVICE=cpu \
  ./run_fix_token_len.sh

# 固定 token 长度
MODE=token_len SYNTHETIC_TOKEN_LEN=64 MAX_SAMPLES=10000 BACKEND=torch DEVICE=cpu \
  ./run_fix_token_len.sh
```

### Embedding：vLLM OpenAI 兼容 HTTP

1) 启动 vLLM embedding server（示例端口 9090）：

```bash
cd scripts/embedding/vllm
PORT=9090 ./start_vllm_server.sh
```

2) 运行 client：

```bash
cd scripts/embedding
BASE_URL=http://127.0.0.1:9090 BACKEND=vllm-http \
  MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 \
  ./run_fix_token_len.sh
```

### Qwen3：LLM 压测

详见 [`scripts/qwen3/README.md`](scripts/qwen3/README.md)。

```bash
cd scripts/qwen3
./run_qwen3_test.sh
```

### Omni：多模态压测

详见 [`scripts/omni/README.md`](scripts/omni/README.md)。

```bash
cd scripts/omni
./run_qwen_omni_synthetic.sh
```

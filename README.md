# multi-model-process-eval

统一的多模型推理评测框架，支持在 **torch / SGLang / vLLM** 三种后端上运行 **Embedding、LLM (Qwen3)、VL（视觉语言）、Omni（多模态）** 四大类任务，面向 Intel CPU (AMX/AVX512) 优化场景的性能基准测试与横向对比。

## 快速导航

| 任务 | 文档 | 说明 |
|------|------|------|
| 总入口 | [`scripts/README.md`](scripts/README.md) | 脚本总览与环境变量 |
| Embedding | [`scripts/embedding/README.md`](scripts/embedding/README.md) | 文本/图像向量 + 合成数据压测 |
| Qwen3 LLM | [`scripts/qwen3/README.md`](scripts/qwen3/README.md) | LLM 压测 + TTFT/TPOT |
| VL | [`scripts/vl/`](scripts/vl/) | Qwen2.5-VL 图文对话 |
| Omni | [`scripts/omni/README.md`](scripts/omni/README.md) | 多模态/音频 Omni 压测 |
| VL-Embedding | [`scripts/vl-embedding/`](scripts/vl-embedding/) | 图像 embedding |
| 自动化测试 | [`scripts/auto-test/`](scripts/auto-test/) | 多实例/多配置批量压测 |
| 工具 | [`scripts/tools/README.md`](scripts/tools/README.md) | FP8→FP16 等辅助工具 |

## 架构概览

```
src/
├── data/                         # 数据加载层
│   ├── embedding_inputs.py       #   通用 embedding 输入（文本/JSONL/图片，去重 + 限数）
│   ├── flickr8k.py               #   Flickr8k 图文配对
│   └── yahoo_answers.py          #   Yahoo Answers JSONL 解析（q/a/q+a）
│
├── tasks/                        # 任务逻辑层 + 后端适配
│   ├── embedding.py              #   Embedding 任务入口
│   ├── qwen3.py                  #   Qwen3 LLM 任务入口
│   ├── vl.py                     #   Vision-Language 任务入口
│   ├── omni.py                   #   Omni 多模态任务入口
│   ├── embedding_backends/       #   Embedding 各后端实现
│   ├── qwen3_backends/           #   Qwen3 各后端实现
│   ├── vl_backends/              #   VL 各后端实现
│   └── omni_backends/            #   Omni 各后端实现
│
├── agent_service/                # Agent 服务 (FastAPI)
│   ├── app.py                    #   /v1/skills, /v1/agent/chat 端点
│   ├── config.py                 #   AGENT_* 环境变量配置
│   ├── llm_openai.py             #   OpenAI 兼容客户端 + tool call 循环
│   └── skills/                   #   可插拔 skill 注册系统
│
scripts/                          # 可运行脚本入口（按任务分目录）
├── embedding/                    #   run_embedding.py + server 启动脚本
├── qwen3/                        #   run_qwen3.py + benchmark 脚本
├── vl/                           #   run_vl.py + Flickr8k/合成图片测试
├── omni/                         #   run_omni.py + 合成数据压测
├── vl-embedding/                 #   run_image_embedding.py
├── auto-test/                    #   run_auto_test.py 自动化测试框架
└── tools/                        #   dequantize_fp8_to_fp16.py 等

tools/
└── generate_random_hf_weights_from_index.py  # 从 HF index 生成随机权重（测试用）
```

## 支持矩阵

| 任务 | 模型系列 | torch | sglang | sglang-offline | vllm | vllm-http |
|------|---------|:-----:|:------:|:--------------:|:----:|:---------:|
| **Embedding** | Qwen3-Embedding (0.6B, 4B), CLIP, Youtu | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Qwen3 LLM** | Qwen3 (0.6B, 1.7B, 4B) | — | ✅ | — | — | ✅ |
| **VL** | Qwen2.5-VL (3B, 7B) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Omni** | Qwen2.5-Omni (3B, 7B) | — | ✅ | — | ✅ | ✅ |

## 设计模式

- **Session 模式**：`load_*_session()` 做模型加载（一次性开销），`embed()` / `chat_with_session()` 做纯推理，确保基准测试准确
- **后端抽象**：每个任务的 `*_backends/` 目录封装 sglang-http、vllm-http、torch 等实现，上层代码通过 `_backend_tag` 识别后端能力
- **图像传输抽象**：统一的 data-url / base64 / 路径三种模式
- **Profiling 可选**：由任务层管控 profiler 生命周期，后端无感知

## 安装

```bash
# 基础依赖
pip install -r requirements.txt

# 按场景选装
pip install -r requirements-cpu.txt    # CPU (IPEX/AMX)
pip install -r requirements-cuda.txt   # CUDA
pip install -r requirements-agent.txt  # Agent 服务 (FastAPI)
pip install -r requirements-mteb.txt   # MTEB 评测
pip install -r requirements-emon.txt   # Intel Emon 能耗监控
```

> 不同后端（vllm / sglang）对 openai SDK 版本等有冲突，建议为不同后端建独立 venv。

## 常用示例

### Embedding：合成固定长度压测

```bash
cd scripts/embedding

# 固定字符长度
MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 BACKEND=torch DEVICE=cpu \
  ./run_fix_token_len.sh

# 固定 token 长度
MODE=token_len SYNTHETIC_TOKEN_LEN=64 MAX_SAMPLES=10000 BACKEND=torch DEVICE=cpu \
  ./run_fix_token_len.sh
```

### Embedding：vLLM OpenAI 兼容 HTTP

```bash
# 1) 启动 vLLM embedding server
cd scripts/embedding/vllm
PORT=9090 ./start_vllm_server.sh

# 2) 运行 client
cd scripts/embedding
BASE_URL=http://127.0.0.1:9090 BACKEND=vllm-http \
  MODE=input_len SYNTHETIC_INPUT_LEN=512 MAX_SAMPLES=10000 \
  ./run_fix_token_len.sh
```

### Embedding：SGLang CPU Server

```bash
cd scripts/embedding/sglang
MODEL_DIR=/path/to/model BATCH_SIZE=16 PORT=30000 ./start_sglang_server.sh
```

### Qwen3：LLM 压测

```bash
cd scripts/qwen3
./run_qwen3_test.sh
```

### VL：视觉语言压测

```bash
cd scripts/vl
./run_qwen_vl_flickr8k.sh        # Flickr8k 数据集
./run_qwen_vl_synthetic.sh       # 合成图片
```

### Omni：多模态压测

```bash
cd scripts/omni
./run_qwen_omni_synthetic.sh
```

### 自动化测试（多实例/多配置）

```bash
cd scripts/auto-test/embedding
python run_auto_test.py --config config_fix_token_len.json
```

## 关键环境变量

| 类别 | 变量 | 说明 |
|------|------|------|
| 模型 | `MODEL`, `MODEL_ID`, `MODEL_DIR` | 模型名 / HF 路径 / 本地路径 |
| 后端 | `BACKEND`, `DEVICE` | torch / sglang / vllm-http 等；cpu / cuda |
| HTTP | `BASE_URL`, `HOST`, `PORT` | 服务端点配置 |
| 性能 | `BATCH_SIZE`, `MAX_SAMPLES`, `WARMUP_SAMPLES` | 批大小 / 样本数 / 预热 |
| 精度 | `DTYPE`, `USE_AMX` | bf16 / fp16 / fp32；AMX 加速开关 |
| Profiling | `PROFILE`, `PROFILE_ACTIVITIES`, `PROFILE_OUT_DIR` | 性能分析 |
| SGLang | `SGLANG_MAX_TOTAL_TOKENS`, `SGLANG_CONTEXT_LENGTH`, `SGLANG_MEM_FRACTION_STATIC` | KV cache / 内存控制 |
| Auto-test | `AUTO_TEST_CPU_EXPR` | CPU 亲和性绑核表达式 |
| Agent | `AGENT_OPENAI_BASE_URL`, `AGENT_MODEL`, `AGENT_MAX_TOOL_STEPS` | Agent 服务配置 |

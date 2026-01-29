#!/usr/bin/env bash
set -euo pipefail

# Run MTEB using container embedding backends via scripts/embedding/mteb/run_mteb.py.
#
# Environment overrides:
#   BACKEND (default: vllm-http)              # vllm-http|sglang
#   BASE_URL (default: http://127.0.0.1:9090)
#   MODEL_ID (required)                      # must match server served-model-name
#   API (default: v1)                        # sglang only: native|v1|openai
#   API_KEY (default: empty)
#   TIMEOUT (default: 120)
#   ENCODING_FORMAT (default: base64 for vllm-http)
#
# Encoding:
#   BATCH_SIZE (default: 128)
#   MAX_LENGTH (default: 512)
#   NORMALIZE (default: 1)                   # set 0/false to disable
#   QUERY_PREFIX, DOCUMENT_PREFIX
#
# Task selection (use one):
#   TASKS (default: STSBenchmark)            # comma-separated
#   BENCHMARK (default: empty)
#   TASK_TYPES, LANGUAGES, DOMAINS           # comma-separated filters
#
# Profiling (sglang-http only):
#   PROFILE (default: 0)
#   PROFILE_KWARGS (default: empty)          # JSON string or path to a JSON file
#
# Output:
#   OUTPUT_FOLDER (default: scripts/embedding/mteb)
#     - MTEB will create OUTPUT_FOLDER/results and OUTPUT_FOLDER/view
#   OVERWRITE (default: 0)                  # set 1/true to force re-run
#   CLEAR_CACHE (default: 1)                # set 1/true to delete per-model cached results each run
#   PRUNE_OUTPUT (default: 1)
#     - if 1/true: delete generated cache/view files after run and keep only OUTPUT_FOLDER/results/<task>.json for tasks you ran
#
# Usage:
#   MODEL_ID=<served-model-name> $0 [TASKS] [extra python args...]

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

BACKEND=${BACKEND:-vllm-http}
BASE_URL=${BASE_URL:-http://127.0.0.1:9090}
MODEL_ID=${MODEL_ID:-}
API=${API:-v1}
API_KEY=${API_KEY:-}
TIMEOUT=${TIMEOUT:-120}

BATCH_SIZE=${BATCH_SIZE:-128}
MAX_LENGTH=${MAX_LENGTH:-512}
NORMALIZE=${NORMALIZE:-1}
QUERY_PREFIX=${QUERY_PREFIX:-}
DOCUMENT_PREFIX=${DOCUMENT_PREFIX:-}

# Backward/alternate naming:
# - prefer TASKS if provided
# - else allow TASK as a single-task shorthand
# - else default to STSBenchmark
TASK=${TASK:-}
TASKS=${TASKS:-}
if [[ -z "${TASKS}" ]] && [[ -n "${TASK}" ]]; then
  TASKS="${TASK}"
fi
TASKS=${TASKS:-STSBenchmark}
BENCHMARK=${BENCHMARK:-}
TASK_TYPES=${TASK_TYPES:-}
LANGUAGES=${LANGUAGES:-}
DOMAINS=${DOMAINS:-}

PROFILE=${PROFILE:-0}
PROFILE_KWARGS=${PROFILE_KWARGS:-}

OUTPUT_FOLDER=${OUTPUT_FOLDER:-scripts/embedding/mteb/}
OVERWRITE=${OVERWRITE:-1}
CLEAR_CACHE=${CLEAR_CACHE:-1}
PRUNE_OUTPUT=${PRUNE_OUTPUT:-1}

# Optional positional override: first arg = TASKS (if not a flag)
if [[ $# -gt 0 ]] && [[ "${1:-}" != "--"* ]] && [[ "${1:-}" != "-"* ]]; then
  TASKS="$1"
  shift
fi

if [[ -z "${MODEL_ID}" ]]; then
  echo "Error: MODEL_ID is required (must match server served-model-name)." >&2
  echo "Example: MODEL_ID=my-embed-model BACKEND=vllm-http BASE_URL=http://127.0.0.1:9090 $0" >&2
  exit 1
fi

if [[ -n "${TASKS}" ]] && [[ -n "${BENCHMARK}" ]]; then
  echo "Error: set only one of TASKS or BENCHMARK." >&2
  exit 1
fi

# vLLM OpenAI embeddings support encoding_format=float|base64.
# base64 avoids server-side JSON serialization failures when embeddings contain NaNs.
ENCODING_FORMAT=${ENCODING_FORMAT:-}
if [[ "${BACKEND}" == vllm-http* ]] && [[ -z "${ENCODING_FORMAT}" ]]; then
  ENCODING_FORMAT=base64
fi

cd "${ROOT_DIR}"

# Ensure repo root is importable as a module path (needed when environments set PYTHONSAFEPATH
# or otherwise do not add the current working directory to sys.path).
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="python"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

echo "[run_mteb] PYTHON=${PYTHON_BIN}"
echo "[run_mteb] BACKEND=${BACKEND}"
echo "[run_mteb] BASE_URL=${BASE_URL}"
echo "[run_mteb] MODEL_ID=${MODEL_ID}"
echo "[run_mteb] API=${API}"
echo "[run_mteb] TIMEOUT=${TIMEOUT}"
echo "[run_mteb] ENCODING_FORMAT=${ENCODING_FORMAT:-<unset>}"
echo "[run_mteb] BATCH_SIZE=${BATCH_SIZE}"
echo "[run_mteb] MAX_LENGTH=${MAX_LENGTH}"
echo "[run_mteb] NORMALIZE=${NORMALIZE}"
echo "[run_mteb] QUERY_PREFIX=${QUERY_PREFIX:-<unset>}"
echo "[run_mteb] DOCUMENT_PREFIX=${DOCUMENT_PREFIX:-<unset>}"
echo "[run_mteb] TASKS=${TASKS:-<unset>}"
echo "[run_mteb] BENCHMARK=${BENCHMARK:-<unset>}"
echo "[run_mteb] TASK_TYPES=${TASK_TYPES:-<unset>}"
echo "[run_mteb] LANGUAGES=${LANGUAGES:-<unset>}"
echo "[run_mteb] DOMAINS=${DOMAINS:-<unset>}"
echo "[run_mteb] PROFILE=${PROFILE}"
echo "[run_mteb] PROFILE_KWARGS=${PROFILE_KWARGS:-<unset>}"
echo "[run_mteb] OUTPUT_FOLDER=${OUTPUT_FOLDER}"
echo "[run_mteb] OVERWRITE=${OVERWRITE}"
echo "[run_mteb] CLEAR_CACHE=${CLEAR_CACHE}"
echo "[run_mteb] PRUNE_OUTPUT=${PRUNE_OUTPUT}"
if [[ $# -gt 0 ]]; then
  printf '[run_mteb] EXTRA_ARGS='; printf '%q ' "$@"; printf '\n'
else
  echo "[run_mteb] EXTRA_ARGS=<none>"
fi

ARGS=(
  --backend "${BACKEND}"
  --base-url "${BASE_URL}"
  --model-id "${MODEL_ID}"
  --api "${API}"
  --api-key "${API_KEY}"
  --timeout "${TIMEOUT}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --output-folder "${OUTPUT_FOLDER}"
)

case "${PRUNE_OUTPUT}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --prune-output "* ]]; then
      ARGS+=(--prune-output)
    fi
    ;;
esac

case "${CLEAR_CACHE}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --clear-cache "* ]]; then
      ARGS+=(--clear-cache)
    fi
    ;;
esac

case "${OVERWRITE}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --overwrite "* ]]; then
      ARGS+=(--overwrite)
    fi
    ;;
esac

if [[ -n "${ENCODING_FORMAT}" ]]; then
  if [[ " $* " != *" --encoding-format "* ]]; then
    ARGS+=(--encoding-format "${ENCODING_FORMAT}")
  fi
fi

case "${NORMALIZE}" in
  0|false|FALSE|no|NO|off|OFF)
    if [[ " $* " != *" --no-normalize "* ]]; then
      ARGS+=(--no-normalize)
    fi
    ;;
esac

if [[ -n "${QUERY_PREFIX}" ]]; then
  if [[ " $* " != *" --query-prefix "* ]]; then
    ARGS+=(--query-prefix "${QUERY_PREFIX}")
  fi
fi

if [[ -n "${DOCUMENT_PREFIX}" ]]; then
  if [[ " $* " != *" --document-prefix "* ]]; then
    ARGS+=(--document-prefix "${DOCUMENT_PREFIX}")
  fi
fi

case "${PROFILE}" in
  1|true|TRUE|yes|YES|on|ON)
    if [[ " $* " != *" --profile "* ]]; then
      ARGS+=(--profile)
    fi
    ;;
esac

if [[ -n "${PROFILE_KWARGS}" ]]; then
  if [[ " $* " != *" --profile-kwargs "* ]]; then
    ARGS+=(--profile-kwargs "${PROFILE_KWARGS}")
  fi
fi

if [[ -n "${BENCHMARK}" ]]; then
  if [[ " $* " != *" --benchmark "* ]]; then
    ARGS+=(--benchmark "${BENCHMARK}")
  fi
else
  if [[ -n "${TASKS}" ]]; then
    if [[ " $* " != *" --tasks "* ]]; then
      ARGS+=(--tasks "${TASKS}")
    fi
  fi
fi

if [[ -n "${TASK_TYPES}" ]]; then
  if [[ " $* " != *" --task-types "* ]]; then
    ARGS+=(--task-types "${TASK_TYPES}")
  fi
fi

if [[ -n "${LANGUAGES}" ]]; then
  if [[ " $* " != *" --languages "* ]]; then
    ARGS+=(--languages "${LANGUAGES}")
  fi
fi

if [[ -n "${DOMAINS}" ]]; then
  if [[ " $* " != *" --domains "* ]]; then
    ARGS+=(--domains "${DOMAINS}")
  fi
fi

"${PYTHON_BIN}" scripts/embedding/mteb/run_mteb.py "${ARGS[@]}" "$@"

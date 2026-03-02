#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# Tip: most workflows run this repo under `conda activate sglang-cpu`.
# If you want this script to best-effort auto-activate that env, run with:
#   AUTO_ACTIVATE_SGLANG_CPU=1 ./run_scale_fix_image_size.sh ...
if [[ "${AUTO_ACTIVATE_SGLANG_CPU:-0}" == "1" ]]; then
  if [[ "${CONDA_DEFAULT_ENV:-}" != "sglang-cpu" ]]; then
    if command -v conda >/dev/null 2>&1; then
      __conda_base="$(conda info --base 2>/dev/null || true)"
      if [[ -n "${__conda_base}" && -f "${__conda_base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "${__conda_base}/etc/profile.d/conda.sh" || true
        conda activate sglang-cpu || true
      fi
    fi
  fi
fi

CFG="${SCRIPT_DIR}/config/local/local.json"
NOHUP_MODE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/scale-test/vl-embedding/run_scale_fix_image_size.sh [--config <FILE>] [--nohup] [runner args...]

Options:
  --config <FILE>   Scale-test config JSON (default: config/local/local.json)
  --nohup           Run in background and write logs under <result_root>/<scale_id>/launcher_logs/

Forwarded runner flags (examples):
  --scale-id <ID>   Put all artifacts under <result_root>/<scale_id>/ (also used for remote dispatch)
  --resume          Skip jobs/hosts already completed successfully in the same <result_root>/<scale_id>/

Notes:
  - In --nohup mode, this script auto-chooses --scale-id if you didn't pass one.
  - If you want --resume to actually resume, pass a fixed --scale-id.
  - To monitor: bash scripts/scale-test/vl-embedding/monitor_scale_fix_image_size.sh --scale-id <scale_id>
EOF
}

# Accept either:
#   ./run_scale_fix_image_size.sh --tee
#   ./run_scale_fix_image_size.sh --config path/to/config.json --tee
#   ./run_scale_fix_image_size.sh path/to/config.json --tee   (back-compat)
if [[ ${1:-} == "--config" ]]; then
  if [[ -z ${2:-} ]]; then
    echo "error: --config requires a value" >&2
    exit 2
  fi
  CFG="$2"
  shift 2
elif [[ ${1:-} != "" && ${1:-} != -* ]]; then
  CFG="$1"
  shift || true
fi

# Wrapper-only flags (must be removed before forwarding to Python runner).
forward_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nohup)
      NOHUP_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

set -- "${forward_args[@]}"

# Default to streaming output (including remote SSH output) unless the user
# explicitly disables it.
want_default_tee=1
for a in "$@"; do
  if [[ "$a" == "--tee" || "$a" == "--no-tee" ]]; then
    want_default_tee=0
    break
  fi
done

extra_args=()
if [[ $want_default_tee -eq 1 ]]; then
  extra_args+=("--tee")
fi

if [[ "$NOHUP_MODE" == "1" ]]; then
  SCALE_ID=""
  RESULT_ROOT_IN=""
  args2=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scale-id)
        SCALE_ID="${2:-}"; args2+=("$1" "$2"); shift 2 ;;
      --result-root)
        RESULT_ROOT_IN="${2:-}"; args2+=("$1" "$2"); shift 2 ;;
      *)
        args2+=("$1"); shift ;;
    esac
  done
  set -- "${args2[@]}"

  if [[ -z "$SCALE_ID" ]]; then
    SCALE_ID=$(date -u +%Y%m%dT%H%M%SZ)
    set -- "--scale-id" "$SCALE_ID" "$@"
  fi

  result_root_cfg=$(
    python3 - <<PY 2>/dev/null || python - <<PY
import json
from pathlib import Path
cfg = Path(${CFG@Q}).expanduser()
obj = json.loads(cfg.read_text(encoding='utf-8'))
print((obj.get('result_root') or '').strip())
PY
  )
  RESULT_ROOT="${RESULT_ROOT_IN:-$result_root_cfg}"
  if [[ -z "$RESULT_ROOT" ]]; then
    echo "error: could not determine result_root (config missing result_root?)" >&2
    exit 2
  fi
  if [[ "$RESULT_ROOT" != /* ]]; then
    RESULT_ROOT="${REPO_ROOT%/}/${RESULT_ROOT}"
  fi

  RUN_DIR="${RESULT_ROOT%/}/${SCALE_ID}"
  LAUNCH_DIR="${RUN_DIR}/launcher_logs"
  mkdir -p "$LAUNCH_DIR"
  LOG_PATH="${LAUNCH_DIR}/nohup.log"
  PID_PATH="${LAUNCH_DIR}/nohup.pid"
  CMD_PATH="${LAUNCH_DIR}/command.txt"

  cmd=(PYTHONUNBUFFERED=1 python -u "${SCRIPT_DIR}/run_scale_fix_image_size.py" --config "$CFG" "${extra_args[@]}" "$@")
  printf '%q ' "${cmd[@]}" > "$CMD_PATH"
  echo >> "$CMD_PATH"

  nohup "${cmd[@]}" > "$LOG_PATH" 2>&1 &
  echo $! > "$PID_PATH"

  echo "[nohup] scale_id=${SCALE_ID}"
  echo "[nohup] run_dir=${RUN_DIR}"
  echo "[nohup] log=${LOG_PATH}"
  echo "[nohup] pid_file=${PID_PATH} (pid=$(cat "$PID_PATH"))"
  echo "[nohup] monitor: bash scripts/scale-test/vl-embedding/monitor_scale_fix_image_size.sh --scale-id ${SCALE_ID} --result-root ${RESULT_ROOT}"
  exit 0
fi

PYTHONUNBUFFERED=1 python -u "${SCRIPT_DIR}/run_scale_fix_image_size.py" --config "$CFG" "${extra_args[@]}" "$@"

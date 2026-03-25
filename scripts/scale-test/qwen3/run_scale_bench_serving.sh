#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

JOB_CFG="${SCRIPT_DIR}/config/local/smoke.json"
REMOTE_CFG=""
NOHUP_MODE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/scale-test/qwen3/run_scale_bench_serving.sh [--job-config <FILE>] [--remote-config <FILE>] [--nohup] [runner args...]

Options:
  --job-config <FILE>     Scale-test job config JSON (default: config/local/smoke.json)
  --remote-config <FILE>  Remote/server config JSON
  --nohup                 Run in background and write launcher logs under the run dir

Examples:
  bash scripts/scale-test/qwen3/run_scale_bench_serving.sh --job-config scripts/scale-test/qwen3/config/local/smoke.json --tee
  bash scripts/scale-test/qwen3/run_scale_bench_serving.sh --job-config scripts/scale-test/qwen3/config/local/local-sglang.json --tee
EOF
}

forward_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-config)
      JOB_CFG="${2:-}"
      shift 2
      ;;
    --remote-config)
      REMOTE_CFG="${2:-}"
      shift 2
      ;;
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

extra_args=()
want_default_tee=1
for a in "$@"; do
  if [[ "$a" == "--tee" || "$a" == "--no-tee" ]]; then
    want_default_tee=0
    break
  fi
done
if [[ $want_default_tee -eq 1 ]]; then
  extra_args+=("--tee")
fi

cmd=(env PYTHONUNBUFFERED=1 python -u "${SCRIPT_DIR}/run_scale_bench_serving.py" --job-config "$JOB_CFG")
if [[ -n "$REMOTE_CFG" ]]; then
  cmd+=(--remote-config "$REMOTE_CFG")
fi
cmd+=("${extra_args[@]}" "$@")

if [[ "$NOHUP_MODE" == "1" ]]; then
  SCALE_ID=""
  args2=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scale-id)
        SCALE_ID="${2:-}"
        args2+=("$1" "$2")
        shift 2
        ;;
      *)
        args2+=("$1")
        shift
        ;;
    esac
  done
  set -- "${args2[@]}"

  if [[ -z "$SCALE_ID" ]]; then
    SCALE_ID=$(date -u +%Y%m%dT%H%M%SZ)
    cmd+=(--scale-id "$SCALE_ID")
  fi

  result_root_cfg=$(python - <<PY
import json
from pathlib import Path
cfg = Path(${JOB_CFG@Q}).expanduser()
obj = json.loads(cfg.read_text(encoding="utf-8"))
print((obj.get("result_root") or "").strip())
PY
)
  if [[ -z "$result_root_cfg" ]]; then
    echo "error: could not determine result_root from $JOB_CFG" >&2
    exit 2
  fi
  if [[ "$result_root_cfg" != /* ]]; then
    result_root_cfg="${REPO_ROOT%/}/${result_root_cfg}"
  fi
  run_dir="${result_root_cfg%/}/${SCALE_ID}"
  launch_dir="${run_dir}/launcher_logs"
  mkdir -p "$launch_dir"
  log_path="${launch_dir}/nohup.log"
  pid_path="${launch_dir}/nohup.pid"

  nohup "${cmd[@]}" > "$log_path" 2>&1 &
  echo $! > "$pid_path"
  echo "[nohup] scale_id=${SCALE_ID}"
  echo "[nohup] run_dir=${run_dir}"
  echo "[nohup] log=${log_path}"
  echo "[nohup] pid_file=${pid_path} (pid=$(cat "$pid_path"))"
  exit 0
fi

exec "${cmd[@]}"
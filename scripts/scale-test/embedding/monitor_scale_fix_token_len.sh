#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  monitor_scale_fix_token_len.sh --scale-id <SCALE_ID> [--result-root <DIR>] [--interval <SECONDS>]

Prints periodic status lines by inspecting per-host logs under the scale-test result directory.

Examples:
  bash scripts/scale-test/embedding/monitor_scale_fix_token_len.sh --scale-id 20260212T051115Z
EOF
}

SCALE_ID=""
RESULT_ROOT="scripts/scale-test/embedding/result/fix_token_len"
INTERVAL="30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale-id)
      SCALE_ID="${2:-}"; shift 2 ;;
    --result-root)
      RESULT_ROOT="${2:-}"; shift 2 ;;
    --interval)
      INTERVAL="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$SCALE_ID" ]]; then
  echo "[error] --scale-id is required" >&2
  usage
  exit 2
fi

RUN_DIR="${RESULT_ROOT%/}/${SCALE_ID}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[error] run dir not found: $RUN_DIR" >&2
  exit 2
fi

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # Local orchestrator liveness (best-effort)
  if pgrep -af "scripts/scale-test/embedding/run_scale_fix_token_len\\.py" | grep -q "${SCALE_ID}"; then
    local_state="RUNNING"
  else
    local_state="NOT_RUNNING"
  fi

  echo "[$ts] scale_id=$SCALE_ID local=$local_state"

  if [[ -f "$RUN_DIR/launcher_logs/nohup.log" ]]; then
    echo "  launcher_log=present (tail)"
    tail -n 2 "$RUN_DIR/launcher_logs/nohup.log" 2>/dev/null | sed -E 's/^/    /'
  fi

  host_logs=$(find "$RUN_DIR/hosts" -maxdepth 2 -type f -name remote_run.log 2>/dev/null | sort || true)
  if [[ -z "$host_logs" ]]; then
    echo "  [warn] no host logs yet"
  else
    while IFS= read -r log; do
      host=$(basename "$(dirname "$log")")
      phase=$(
        { grep -E "^\[dispatch\] phase=" "$log" 2>/dev/null || true; } \
          | tail -n 1 \
          | sed -E 's/^\[dispatch\] phase=//'
      )
      phase=${phase:-unknown}

      err_count=$(
        { grep -E "Traceback|\bERROR\b" "$log" 2>/dev/null || true; } \
          | wc -l \
          | tr -d ' '
      )
      echo "  host=$host phase=$phase errors=$err_count"

      # Print last progress line that looks meaningful (best-effort)
      tail -n 2 "$log" | sed -E 's/^/    /'
    done <<< "$host_logs"
  fi

  if [[ -d "$RUN_DIR/analysis" ]]; then
    echo "  analysis_dir=present"
    find "$RUN_DIR/analysis" -maxdepth 1 -type f -printf "    %f\n" 2>/dev/null | sort || true
  fi

  echo
  sleep "$INTERVAL"
done

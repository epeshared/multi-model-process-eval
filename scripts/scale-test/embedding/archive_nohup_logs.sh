#!/usr/bin/env bash
set -euo pipefail

# Move suite-level nohup_*.log/.pid files into their matching <scale_id> run directory.
#
# Example:
#   bash scripts/scale-test/embedding/archive_nohup_logs.sh
#
# By default it operates on:
#   scripts/scale-test/embedding/result/fix_token_len

RESULT_ROOT="scripts/scale-test/embedding/result/fix_token_len"

usage() {
  cat <<'EOF'
Usage:
  archive_nohup_logs.sh [--result-root <DIR>]

Moves:
  <result_root>/nohup_<SCALE_ID>.log  -> <result_root>/<SCALE_ID>/launcher_logs/nohup_<SCALE_ID>.log
  <result_root>/nohup_<SCALE_ID>.pid  -> <result_root>/<SCALE_ID>/launcher_logs/nohup_<SCALE_ID>.pid
  <result_root>/nohup_retry_<SCALE_ID>.log -> <result_root>/<SCALE_ID>/launcher_logs/nohup_retry_<SCALE_ID>.log

Notes:
  - If <result_root>/<SCALE_ID>/ doesn't exist, it will be created.
  - Run directories without analysis/ won't show up in the web UI.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-root)
      RESULT_ROOT="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RESULT_ROOT" ]]; then
  echo "[error] --result-root is empty" >&2
  exit 2
fi

if [[ ! -d "$RESULT_ROOT" ]]; then
  echo "[error] result_root not found: $RESULT_ROOT" >&2
  exit 2
fi

shopt -s nullglob
moved=0
skipped=0

for src in "$RESULT_ROOT"/nohup_*.log "$RESULT_ROOT"/nohup_*.pid "$RESULT_ROOT"/nohup_retry_*.log; do
  base=$(basename "$src")
  if [[ "$base" =~ ^nohup(_retry)?_([0-9]{8}T[0-9]{6}Z)\.(log|pid)$ ]]; then
    scale_id="${BASH_REMATCH[2]}"
    dest_dir="$RESULT_ROOT/$scale_id/launcher_logs"
    mkdir -p "$dest_dir"
    dest="$dest_dir/$base"

    # Avoid overwriting existing artifacts.
    if [[ -e "$dest" ]]; then
      skipped=$((skipped+1))
      continue
    fi

    mv "$src" "$dest"
    moved=$((moved+1))
  else
    skipped=$((skipped+1))
  fi
done

echo "[ok] moved=$moved skipped=$skipped"

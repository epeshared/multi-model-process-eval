#!/usr/bin/env bash
set -euo pipefail

# Ensure a conda env exists on the remote host.
#
# Defaults align with our configs, but you can override via env vars:
# - CONDA_ENV_NAME (default: sglang-cpu)
# - CONDA_PYTHON_VERSION (default: 3.10)

env_name="${CONDA_ENV_NAME:-sglang-cpu}"
py_ver="${CONDA_PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] ensure_conda_env: conda not found in PATH" >&2
  exit 1
fi

# Quick existence check
if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
  echo "[ok] ensure_conda_env: env exists (${env_name})"
  exit 0
fi

echo "[info] ensure_conda_env: creating env (${env_name}) with python=${py_ver}"
conda create -y -n "${env_name}" "python=${py_ver}" pip

# Best-effort: make pip usable immediately.
conda run -n "${env_name}" python -m pip install -U pip >/dev/null 2>&1 || true

echo "[ok] ensure_conda_env: created (${env_name})"

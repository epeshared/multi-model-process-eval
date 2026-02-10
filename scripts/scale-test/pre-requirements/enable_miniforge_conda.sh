#!/usr/bin/env bash
set -euo pipefail

# Make `conda` available in PATH for subsequent commands in the same shell.
#
# This is useful on remote hosts where Miniforge/Conda is installed but not
# sourced by default for non-interactive shells.
#
# Resolution order:
# 1) MINIFORGE_PREFIX
# 2) /root/miniforge3
# 3) $HOME/miniforge3

prefix="${MINIFORGE_PREFIX:-}"
if [[ -z "${prefix}" ]]; then
  if [[ -x "/root/miniforge3/bin/conda" ]]; then
    prefix="/root/miniforge3"
  elif [[ -n "${HOME:-}" && -x "${HOME}/miniforge3/bin/conda" ]]; then
    prefix="${HOME}/miniforge3"
  fi
fi

if [[ -z "${prefix}" ]]; then
  echo "[warn] enable_miniforge_conda: could not find Miniforge prefix (set MINIFORGE_PREFIX)" >&2
  exit 0
fi

if [[ ! -x "${prefix}/bin/conda" ]]; then
  echo "[warn] enable_miniforge_conda: conda not found at ${prefix}/bin/conda" >&2
  exit 0
fi

export PATH="${prefix}/bin:${PATH}"

# Prefer conda.sh when present.
if [[ -f "${prefix}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  . "${prefix}/etc/profile.d/conda.sh"
fi

echo "[ok] enable_miniforge_conda: conda ready (${prefix})"

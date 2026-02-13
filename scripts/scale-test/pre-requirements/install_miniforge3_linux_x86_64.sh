#!/usr/bin/env bash
set -euo pipefail

URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
INSTALLER="Miniforge3-Linux-x86_64.sh"

DEFAULT_PREFIX="$HOME/miniforge3"
if [[ "$(id -u)" == "0" ]]; then
  DEFAULT_PREFIX="/root/miniforge3"
fi
PREFIX="${MINIFORGE_PREFIX:-$DEFAULT_PREFIX}"

workdir="${TMPDIR:-/tmp}/miniforge3-installer"
mkdir -p "$workdir"
cd "$workdir"

if [[ ! -f "$INSTALLER" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" "$URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$INSTALLER" "$URL"
  else
    echo "ERROR: need wget or curl to download Miniforge" >&2
    exit 1
  fi
fi

chmod a+x "$INSTALLER"

# If Miniforge is already installed, skip re-install.
if [[ -x "${PREFIX}/bin/conda" ]]; then
  echo "[ok] miniforge already installed at ${PREFIX}"
  exit 0
fi

# Equivalent to:
#   bash ./Miniforge3-Linux-x86_64.sh -b -u
# but we also pin the install prefix to match PATH export below.
bash "./$INSTALLER" -b -u -p "$PREFIX"

# Persist PATH for future login shells.
BASHRC="$HOME/.bashrc"
mkdir -p "$(dirname "$BASHRC")"
PATH_LINE="export PATH=$PREFIX/bin:\$PATH"
if [[ ! -f "$BASHRC" ]] || ! grep -Fqx "$PATH_LINE" "$BASHRC"; then
  echo "$PATH_LINE" >> "$BASHRC"
fi

# Make PATH available for the current session.
export PATH="$PREFIX/bin:$PATH"

# Best-effort: source bashrc (may no-op in non-interactive shells).
# Do not fail if it errors.
source "$BASHRC" >/dev/null 2>&1 || true

if command -v conda >/dev/null 2>&1; then
  conda --version
fi

#!/usr/bin/env bash
set -euo pipefail

# System deps needed by scripts/embedding/sglang/start_sglang_server.sh
# It requires:
#   /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
#   /usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2
# on Debian/Ubuntu-like systems.

need_paths=(
  "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
  "/usr/lib/x86_64-linux-gnu/libtbbmalloc.so.2"
)

missing=()
for p in "${need_paths[@]}"; do
  if [[ ! -e "$p" ]]; then
    missing+=("$p")
  fi
done

if ((${#missing[@]} == 0)); then
  echo "[ok] system deps already present (tcmalloc/tbbmalloc)"
  exit 0
fi

echo "[info] missing system deps: ${missing[*]}"

if [[ $EUID -ne 0 ]]; then
  echo "[error] install_sglang_system_deps.sh must run as root" >&2
  exit 1
fi

disable_stale_nvidia_local_repo_sources() {
  # Some hosts have a stale local NVIDIA driver repo like:
  #   deb [signed-by=/usr/share/keyrings/nvidia-driver-local-580.105.08-keyring.gpg] file:///var/nvidia-driver-local-repo-ubuntu2204-580.105.08 /
  # which breaks `apt-get update`.
  local pattern='nvidia-driver-local-repo-ubuntu2204-|file:/+var/nvidia-driver-local-repo-ubuntu2204-'
  local f
  for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
    [[ -f "$f" ]] || continue
    if grep -qE "$pattern" "$f"; then
      echo "[warn] disabling stale local NVIDIA repo in: $f" >&2
      # Keep the file in place (avoid apt warnings about filename extensions), just comment it out.
      cp -f "$f" "${f}.bak" || true
      if [[ "$f" == *.list ]]; then
        # Comment only matching lines.
        sed -i -E "/$pattern/ s/^[[:space:]]*(deb(-src)?[[:space:]]+)/# \\1/" "$f" || true
      else
        # Deb822-style .sources: easiest/safest is to comment the whole file.
        sed -i -E 's/^[[:space:]]*([^#])/# \1/' "$f" || true
      fi
    fi
  done
}

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  disable_stale_nvidia_local_repo_sources
  if ! apt-get update -y; then
    echo "[warn] apt-get update failed; retrying after disabling stale local NVIDIA repo entries" >&2
    disable_stale_nvidia_local_repo_sources
    apt-get update -y
  fi
  # libtcmalloc-minimal4 -> libtcmalloc.so.4
  # libtbbmalloc2        -> libtbbmalloc.so.2
  # Also install `numactl` + `lsof` which are used for CPU pinning and stale-server cleanup.
  apt-get install -y --no-install-recommends libtcmalloc-minimal4 libtbbmalloc2 numactl lsof || \
    apt-get install -y --no-install-recommends libtcmalloc-minimal4 libtbb2 numactl lsof
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y gperftools-libs tbb numactl lsof
elif command -v yum >/dev/null 2>&1; then
  yum install -y gperftools-libs tbb numactl lsof
else
  echo "[error] no supported package manager found (apt-get/dnf/yum)" >&2
  exit 1
fi

# Ubuntu's libtcmalloc-minimal4 provides libtcmalloc_minimal.so.4 (note the underscore).
# The sglang start script expects libtcmalloc.so.4, so provide a symlink if needed.
if [[ ! -e /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ]] && [[ -e /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 ]]; then
  ln -sf /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
fi

# Re-check
for p in "${need_paths[@]}"; do
  if [[ ! -e "$p" ]]; then
    echo "[error] still missing after install: $p" >&2
    ls -la "$(dirname "$p")" | head -n 50 >&2 || true
    exit 1
  fi
done

echo "[ok] installed system deps (tcmalloc/tbbmalloc)"

#!/usr/bin/env bash
set -euo pipefail

# This script often runs as root on remote hosts. Suppress pip's root warning and
# reduce noisy output so SSH logs don't look like they're "stuck".
export PIP_ROOT_USER_ACTION="ignore"
export PIP_PROGRESS_BAR="off"
export PIP_DISABLE_PIP_VERSION_CHECK="1"
export PIP_NO_INPUT="1"

repo_url="https://github.com/sgl-project/sglang.git"
src_dir="${SGLANG_SRC_DIR:-sglang}"

git_with_timeout() {
	# Avoid hanging forever on flaky networks.
	# Use timeout(1) when available; otherwise fall back to a tiny Python wrapper.
	local timeout_s="${GIT_TIMEOUT_SECONDS:-45}"
	if command -v timeout >/dev/null 2>&1; then
		GIT_TERMINAL_PROMPT=0 timeout "${timeout_s}" git "$@"
	elif command -v python >/dev/null 2>&1; then
		GIT_TERMINAL_PROMPT=0 python - "$@" <<'PY'
import os
import subprocess
import sys

timeout_s = int(os.environ.get("GIT_TIMEOUT_SECONDS", "45"))
cmd = ["git", *sys.argv[1:]]
try:
	subprocess.run(cmd, check=True, timeout=timeout_s)
except subprocess.TimeoutExpired:
	# Match GNU timeout's exit code for timeouts.
	sys.exit(124)
PY
	else
		GIT_TERMINAL_PROMPT=0 git "$@"
	fi
}

if [[ -d "${src_dir}/.git" ]]; then
	echo "[info] sglang repo exists; updating (${src_dir})"
	# GitHub connectivity can be flaky on some cloud networks; retry fetch, and
	# if it still fails, proceed with the existing checkout.
	fetch_ok=0
	for i in 1 2 3; do
		if git_with_timeout -C "${src_dir}" fetch --all --prune; then
			fetch_ok=1
			break
		fi
		echo "[warn] git fetch failed (attempt ${i}/3); retrying..." >&2
		sleep 2
	done
	if [[ "${fetch_ok}" != "1" ]]; then
		echo "[warn] git fetch failed; using existing sglang checkout" >&2
	fi
	# Best-effort: use origin/main when present.
	git -C "${src_dir}" reset --hard origin/main 2>/dev/null || git -C "${src_dir}" reset --hard origin/master 2>/dev/null || \
		echo "[warn] could not reset to origin/main; leaving current HEAD" >&2
elif [[ -e "${src_dir}" ]]; then
	echo "[error] ${src_dir} exists but is not a git repo; please remove it or set SGLANG_SRC_DIR" >&2
	exit 1
else
	clone_ok=0
	for i in 1 2 3; do
		if git_with_timeout clone "${repo_url}" "${src_dir}"; then
			clone_ok=1
			break
		fi
		echo "[warn] git clone failed (attempt ${i}/3); retrying..." >&2
		sleep 2
	done
	if [[ "${clone_ok}" != "1" ]]; then
		echo "[error] failed to clone sglang after retries" >&2
		exit 1
	fi
fi

cd "${src_dir}"
cd python
cp pyproject_cpu.toml pyproject.toml

# Hotfix: some sglang revisions call _get_quantization_config with a mismatched
# number of args (update_config.py vs loader.py). Patch to be arity-tolerant.
echo "[info] applying sglang _get_quantization_config arity hotfix (best-effort)"
python - <<'PY'
from __future__ import annotations

import re
from pathlib import Path


def patch_loader_signature() -> bool:
	p = Path("sglang/srt/model_loader/loader.py")
	if not p.exists():
		print(f"[warn] loader signature hotfix skipped; file not found: {p}")
		return False
	text = p.read_text(encoding="utf-8", errors="replace")
	# If already arity-tolerant, nothing to do.
	if re.search(r"def\s+_get_quantization_config\([^)]*packed_modules_mapping", text, flags=re.DOTALL):
		print("[ok] loader signature hotfix already applied")
		return True

	# Handle type-annotated multiline signatures, but be tolerant about whitespace and types.
	sig_pat = re.compile(
		r"def\s+_get_quantization_config\(\s*\n"
		r"(?P<indent>[ \t]+)model_config[^\n]*\n"
		r"(?P=indent)load_config[^\n]*\n"
		r"(?P=indent)\)\s*->\s*Optional\[QuantizationConfig\]\s*:",
		flags=re.MULTILINE,
	)
	m = sig_pat.search(text)
	if not m:
		# Fallback: single-line signature.
		sig_pat2 = re.compile(
			r"(^[ \t]*)def\s+_get_quantization_config\(\s*model_config\s*,\s*load_config\s*\)\s*->\s*Optional\[QuantizationConfig\]\s*:",
			flags=re.MULTILINE,
		)
		m = sig_pat2.search(text)
		if not m:
			print("[warn] loader signature hotfix pattern not found; leaving file unchanged")
			return False
		indent = m.group(1)
		replacement = f"{indent}def _get_quantization_config(model_config, load_config, packed_modules_mapping=None) -> Optional[QuantizationConfig]:"
	else:
		indent = m.group("indent")
		replacement = (
			"def _get_quantization_config(\n"
			f"{indent}model_config: ModelConfig,\n"
			f"{indent}load_config: LoadConfig,\n"
			f"{indent}packed_modules_mapping=None,\n"
			f"{indent}) -> Optional[QuantizationConfig]:"
		)

	text2 = text[: m.start()] + replacement + text[m.end() :]

	# Respect the provided mapping if one is passed in; otherwise compute default.
	map_pat = re.compile(
		r'^(?P<i>[ \t]*)packed_modules_mapping\s*=\s*getattr\(model_class,\s*"packed_modules_mapping",\s*\{\}\)\s*$',
		flags=re.MULTILINE,
	)
	text2 = map_pat.sub(
		r'\g<i>if packed_modules_mapping is None:\n\g<i>    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})',
		text2,
		count=1,
	)

	p.write_text(text2, encoding="utf-8")
	print("[ok] loader signature hotfix applied")
	return True


patch_loader_signature()

p = Path("sglang/srt/configs/update_config.py")
if not p.exists():
	print(f"[warn] hotfix skipped; file not found: {p}")
	raise SystemExit(0)

text = p.read_text(encoding="utf-8", errors="replace")
if "except TypeError:" in text and "packed_modules_mapping" in text:
	print("[ok] hotfix already applied")
	raise SystemExit(0)

pat = re.compile(
	r"(?P<indent>^[ \t]*)quant_config\s*=\s*_get_quantization_config\(\s*(?:\n\s*)?"
	r"model_config\s*,\s*(?:\n\s*)?load_config\s*,\s*(?:\n\s*)?packed_modules_mapping\s*(?:\n\s*)?\)\s*\n",
	flags=re.MULTILINE,
)

m = pat.search(text)
if not m:
	print("[warn] hotfix pattern not found; leaving file unchanged")
	raise SystemExit(0)

indent = m.group("indent")
replacement = (
	f"{indent}try:\n"
	f"{indent}    quant_config = _get_quantization_config(\n"
	f"{indent}        model_config, load_config, packed_modules_mapping\n"
	f"{indent}    )\n"
	f"{indent}except TypeError:\n"
	f"{indent}    quant_config = _get_quantization_config(model_config, load_config)\n"
)

text2 = text[: m.start()] + replacement + text[m.end() :]
p.write_text(text2, encoding="utf-8")
print("[ok] hotfix applied")
PY

# On CPU hosts, pulling the default PyPI torch wheels can drag in large CUDA runtime deps.
# Prefer the official CPU-only wheels when available; fall back to the default resolver if not.
install_cpu_torch="${SGLANG_INSTALL_CPU_TORCH:-1}"
torch_ver="${SGLANG_TORCH_VERSION:-2.9.0}"
torchvision_ver="${SGLANG_TORCHVISION_VERSION:-0.24.0}"
torchaudio_ver="${SGLANG_TORCHAUDIO_VERSION:-2.9.0}"

pip install --upgrade pip setuptools wheel

if [[ "${install_cpu_torch}" != "0" ]]; then
	echo "[info] attempting CPU-only torch wheels (${torch_ver}+cpu)"
	if ! pip install --extra-index-url https://download.pytorch.org/whl/cpu \
		"torch==${torch_ver}+cpu" \
		"torchvision==${torchvision_ver}+cpu" \
		"torchaudio==${torchaudio_ver}+cpu"; then
		echo "[warn] CPU-only torch install failed; falling back to default PyPI wheels" >&2
	fi
fi

pip install .
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml

# Some environments export CC/CXX with additional flags (e.g. "g++ -pthread -B ..."),
# which confuses CMake (it expects a compiler path). Prefer letting CMake detect
# system compilers.
unset CC CXX
if ! command -v g++ >/dev/null 2>&1; then
	echo "[error] g++ not found; install build-essential/g++ on the host" >&2
	exit 1
fi

# Some hosts have a /usr/local/cuda stub without nvcc; Torch's CMake config may
# attempt to enable CUDA and fail the build. Force-disable CUDA discovery.
export CUDA_HOME=""
export CUDACXX=""
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON"

pip install .
#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def _run_capture_help(module: str) -> str:
    # vLLM benchmarks print help to stdout in most versions.
    p = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return p.stdout or ""


def _help_has(help_text: str, flag: str) -> bool:
    if not help_text:
        return False
    # Match flag as a standalone token.
    return re.search(r"(^|\s)" + re.escape(flag) + r"(\s|,|$)", help_text) is not None


def _pick_first_flag(help_text: str, candidates: Sequence[str]) -> str:
    for c in candidates:
        if _help_has(help_text, c):
            return c
    return ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wrapper to run vLLM's built-in benchmark_serving against an OpenAI-compatible vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL")
    p.add_argument("--endpoint", default="/v1/chat/completions", help="Endpoint path")
    p.add_argument("--model", default="qwen3-0.6b", help="Served model name")

    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=16, help="Desired max in-flight requests")
    p.add_argument("--request-rate", type=float, default=0.0, help="Global request rate (req/s). 0=as fast as possible")

    p.add_argument("--random-input-len", type=int, default=256)
    p.add_argument("--random-output-len", type=int, default=256)
    p.add_argument("--random-prefix-len", type=int, default=0)

    p.add_argument("--print-cmd", action="store_true", help="Print the resolved vLLM command before running")

    # Pass-through for extra vLLM benchmark args.
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args passed to vllm.benchmarks.benchmark_serving")

    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    # Import check for friendlier errors.
    try:
        import vllm  # noqa: F401
    except Exception as e:
        print("ERROR: vLLM is not installed in this Python environment.", file=sys.stderr)
        print(f"  Import error: {e}", file=sys.stderr)
        print("  Fix: install CUDA env requirements (see requirements-cuda.txt) or pip install vllm", file=sys.stderr)
        sys.exit(2)

    module = "vllm.benchmarks.benchmark_serving"

    # NOTE: Newer/alternate vLLM wheels may not package benchmark modules.
    # In that case we fall back to this repo's OpenAI-compatible benchmark.
    try:
        import importlib.util

        module_spec = importlib.util.find_spec(module)
    except Exception:
        module_spec = None

    if module_spec is None:
        script_dir = Path(__file__).resolve().parent
        fallback = script_dir / "benchmark_openai_server.py"
        if not fallback.exists():
            print(
                f"ERROR: {module} is not available in your vLLM install, and fallback script was not found: {fallback}",
                file=sys.stderr,
            )
            sys.exit(2)

        if args.extra:
            print(
                "WARNING: vLLM built-in benchmark module is missing; using benchmark_openai_server.py instead.",
                file=sys.stderr,
            )
            print(
                "WARNING: Extra args after '--' are ignored by the fallback benchmark.",
                file=sys.stderr,
            )

        # Map wrapper args -> fallback args.
        # - random-output-len roughly corresponds to max_tokens.
        # - random-prefix-len has no equivalent in the fallback benchmark.
        cmd = [
            sys.executable,
            str(fallback),
            "--base-url",
            str(args.base_url),
            "--endpoint",
            str(args.endpoint),
            "--model",
            str(args.model),
            "--num-prompts",
            str(int(args.num_prompts)),
            "--concurrency",
            str(int(args.concurrency)),
            "--request-rate",
            str(float(args.request_rate)),
            "--random-input-len",
            str(int(args.random_input_len)),
            "--max-tokens",
            str(int(args.random_output_len)),
        ]

        if args.print_cmd:
            print("[fallback-openai] resolved cmd:")
            print(" ".join(subprocess.list2cmdline([c]) if " " in c else c for c in cmd))

        proc = subprocess.run(cmd)
        raise SystemExit(proc.returncode)

    help_text = _run_capture_help(module)

    # Resolve flag names across vLLM versions.
    flag_base_url = _pick_first_flag(help_text, ["--base-url", "--base_url"])
    flag_model = _pick_first_flag(help_text, ["--model"])
    flag_backend = _pick_first_flag(help_text, ["--backend"])
    flag_endpoint = _pick_first_flag(help_text, ["--endpoint", "--endpoint-path", "--endpoint_path"])

    flag_num_prompts = _pick_first_flag(help_text, ["--num-prompts", "--num_prompts"])
    flag_request_rate = _pick_first_flag(help_text, ["--request-rate", "--request_rate"])
    flag_concurrency = _pick_first_flag(help_text, ["--max-concurrency", "--max_concurrency", "--concurrency"])

    flag_in_len = _pick_first_flag(help_text, ["--random-input-len", "--random_input_len", "--input-len", "--input_len"])
    flag_out_len = _pick_first_flag(help_text, ["--random-output-len", "--random_output_len", "--output-len", "--output_len"])
    flag_prefix_len = _pick_first_flag(help_text, ["--random-prefix-len", "--random_prefix_len", "--prefix-len", "--prefix_len"])

    cmd: List[str] = [sys.executable, "-m", module]

    # Always prefer OpenAI-style backend when available.
    if flag_backend:
        # Most versions accept 'openai'.
        cmd += [flag_backend, "openai"]

    if flag_base_url:
        cmd += [flag_base_url, str(args.base_url)]
    if flag_model:
        cmd += [flag_model, str(args.model)]
    if flag_endpoint:
        cmd += [flag_endpoint, str(args.endpoint)]

    if flag_num_prompts:
        cmd += [flag_num_prompts, str(int(args.num_prompts))]
    if flag_request_rate and float(args.request_rate) >= 0:
        cmd += [flag_request_rate, str(float(args.request_rate))]
    if flag_concurrency:
        cmd += [flag_concurrency, str(int(args.concurrency))]

    if flag_in_len:
        cmd += [flag_in_len, str(int(args.random_input_len))]
    if flag_out_len:
        cmd += [flag_out_len, str(int(args.random_output_len))]
    if flag_prefix_len:
        cmd += [flag_prefix_len, str(int(args.random_prefix_len))]

    # Pass through additional args after a "--".
    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd += extra

    if args.print_cmd:
        print("[vllm-builtin] resolved cmd:")
        print(" ".join(subprocess.list2cmdline([c]) if " " in c else c for c in cmd))

    # Run and forward output.
    proc = subprocess.run(cmd)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

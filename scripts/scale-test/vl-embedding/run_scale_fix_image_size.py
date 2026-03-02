#!/usr/bin/env python3
"""Shim entrypoint for VL image-size scale tests.

We reuse the generic scale-test runner implemented in:
  scripts/scale-test/embedding/run_scale_fix_token_len.py

This file exists mainly so the directory structure under
`scripts/scale-test/vl-embedding/` matches `scripts/scale-test/embedding/`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "embedding" / "run_scale_fix_token_len.py"
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

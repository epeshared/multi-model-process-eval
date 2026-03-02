# Auto-test: VL Embedding (fix_image_size)

This folder contains the *test config* for benchmarking multimodal image embeddings.

- Script: `scripts/vl-embedding/run_fix_image_size.sh`
- Runner (shared harness): `scripts/auto-test/embedding/run_auto_test.py`

## Run locally

1) Make sure your SGLang server can serve `Qwen/Qwen3-VL-Embedding-2B` with multimodal enabled.

2) Generate some local images (see `scripts/scale-test/vl-embedding/README.md`).

3) Run:

- `python3 scripts/auto-test/embedding/run_auto_test.py --config scripts/auto-test/vl-embedding/config_fix_image_size.json --tee`

You can also provide jobs in the config, or have scale-test generate them.

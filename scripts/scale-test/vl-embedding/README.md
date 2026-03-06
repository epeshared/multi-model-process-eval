# Scale Test: VL Embedding (fix_image_size)

This scale-test sweeps **image resolution** (e.g. 224x224 → 1920x1080) and batch size for
**image embeddings** served by an SGLang embedding server (multimodal).

## Prepare images

Generate local test images (different M×N resolutions) under a folder that will be accessible on the runner host
(and on the server host if you use `IMAGE_TRANSPORT=path/url`). For `data-url` transport, only the runner needs access.

Example (local):

- `python3 scripts/scale-test/vl-embedding/gen_test_images.py --out scripts/scale-test/vl-embedding/images --sizes 224x224,384x384,512x512,1024x1024,1280x720,1920x1080 --per-size 8`

## Run

- Local (no SSH dispatch):
	- `bash scripts/scale-test/vl-embedding/run_scale_fix_image_size.sh --job-config scripts/scale-test/vl-embedding/config/local/smoke.json --no-ssh-dispatch --tee`

- Remote (SSH dispatch):
	- `bash scripts/scale-test/vl-embedding/run_scale_fix_image_size.sh --job-config scripts/scale-test/vl-embedding/config/t-cloud/s9e-16c-64G.json --tee`

- Background launch:
	- `bash scripts/scale-test/vl-embedding/run_scale_fix_image_size.sh --job-config scripts/scale-test/vl-embedding/config/t-cloud/s9e-16c-64G.json --nohup --scale-id <ID> --resume`
	- Monitor: `bash scripts/scale-test/vl-embedding/monitor_scale_fix_image_size.sh --scale-id <ID>`

Notes:

- The template config uses `scripts/embedding/sglang/start_sglang_server_cuda.sh` by default.
- Ensure the model path and conda env match your environment.

Device selection:

- Set `server_template.device` to `"cuda"` (GPU) or `"cpu"` in your scale-test JSON.
	- `cuda` uses `scripts/embedding/sglang/start_sglang_server_cuda.sh`
	- `cpu` uses `scripts/embedding/sglang/start_sglang_server.sh`

Python selection (`SGLANG_PYTHON`):

- You can omit `SGLANG_PYTHON` in JSON. When starting an SGLang server, the auto-test runner defaults it to its own interpreter (`sys.executable`).
- This means:
	- Local: run the scale-test under the desired conda env (e.g. `conda run -n <env> bash ...`), OR explicitly set `SGLANG_PYTHON`.
	- Remote: if you use `servers[*].conda_env` / `servers[*].remote_python`, the remote runner already uses the correct env, so `SGLANG_PYTHON` is typically not needed.

- Optional (JSON-only, portable): set `server_template.conda_env` to a conda env name (e.g. `"xtang-embedding-cuda"`). This will set `SGLANG_CONDA_ENV` and the server start script will run via `conda run -n <env> python ...`.

Common pitfalls:

- Model weights must exist under your `model_dir` (e.g. `model.safetensors`). If you only have `config.json`/tokenizer files, SGLang will fail to load.
- For Qwen3-VL-Embedding, SGLang effectively requires CUDA today. Running with `server_template.device="cpu"` can crash with errors like `NotImplementedError: sgl_kernel::rmsnorm ... CPU backend` (this still reproduces on SGLang main as of `sglang==0.5.9`).
- If you see HTTP 403 + an HTML page when calling a local server (e.g. `127.0.0.1`), your environment proxy may be intercepting localhost traffic. Ensure `NO_PROXY` includes `127.0.0.1,localhost`.
- SGLang CUDA server needs a CUDA-enabled torch env. Set `server_template.conda_env` / `SGLANG_CONDA_ENV` (or `SGLANG_PYTHON`) to a python where `torch.cuda.is_available()` is `True`.
- `libiomp5.so` is auto-discovered by the SGLang start script. Only set `SGLANG_LIB_IOMP` when your host truly doesn't provide it and you know the exact path.
	- Optional strict mode: set `SGLANG_REQUIRE_IOMP=1` to fail fast if `libiomp5.so` can't be found.
- If `huggingface.co` is blocked in your environment, try downloading weights via a mirror endpoint, e.g.
	- `HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir <MODEL_DIR>`

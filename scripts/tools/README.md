# Tools

This directory collects utility scripts used across workflows. More tools and documentation will be added here over time.

## dequantize_fp8_to_fp16.py

Dequantize FP8 (float8) safetensors checkpoints to FP16 (or BF16) for CPU-friendly use.

### Requirements

- Python 3
- `torch`
- `safetensors`

### Usage

```bash
python dequantize_fp8_to_fp16.py \
  --in-model-dir /path/to/fp8_model \
  --out-model-dir /path/to/fp16_model \
  --dtype float16
```

### Options

- `--in-model-dir`: Input directory containing `.safetensors` weights.
- `--out-model-dir`: Output directory for dequantized weights.
- `--dtype`: Target dtype (`float16` or `bfloat16`). Default: `float16`.
- `--overwrite`: Allow overwriting an existing output directory.
- `--keep-quant-aux`: Keep `*.weight_scale` and `*.input_scale` tensors.
- `--verbose`: Print per-tensor conversion details.

### Notes

- This is a lossy conversion because FP8 weights are already quantized.
- The script multiplies each FP8 weight by its corresponding `weight_scale`.
- It copies non-`.safetensors` files from the input directory to the output directory.

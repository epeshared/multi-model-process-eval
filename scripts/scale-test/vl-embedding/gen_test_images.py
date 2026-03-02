#!/usr/bin/env python3
"""Generate local test images of various resolutions.

This is intended for embedding throughput/latency tests where you want a
repeatable, dependency-light set of images without relying on external URLs.

Examples:
  python3 scripts/scale-test/vl-embedding/gen_test_images.py --out /tmp/mmpe_imgs \
    --sizes 224x224,384x384,512x512,1024x1024,1280x720,1920x1080 --per-size 4

  python3 scripts/scale-test/vl-embedding/gen_test_images.py --out ./images \
    --sizes 512x512 --pattern noise --per-size 32 --format jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple

if TYPE_CHECKING:
    from PIL import Image as PILImage


def _parse_sizes(spec: str) -> List[Tuple[int, int]]:
    items: List[Tuple[int, int]] = []
    for raw in (spec or "").split(","):
        s = raw.strip().lower()
        if not s:
            continue
        if "x" in s:
            w_s, h_s = s.split("x", 1)
            w, h = int(w_s), int(h_s)
        else:
            w = h = int(s)
        if w <= 0 or h <= 0:
            raise SystemExit(f"invalid size: {raw!r}")
        items.append((w, h))
    if not items:
        raise SystemExit("--sizes is empty")
    return items


def _draw_checkerboard(*, w: int, h: int, block: int = 32) -> "PILImage.Image":
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                c = (210, 210, 210)
            else:
                c = (60, 60, 60)
            draw.rectangle((x, y, min(x + block, w), min(y + block, h)), fill=c)
    return img


def _draw_gradient(*, w: int, h: int) -> "PILImage.Image":
    from PIL import Image

    # Build small gradients then resize up for speed.
    base_w, base_h = 256, 256
    gx = Image.linear_gradient("L").resize((base_w, base_h))
    gy = gx.rotate(90, expand=False)
    r = gx.resize((w, h))
    g = gy.resize((w, h))
    b = gx.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((w, h))
    return Image.merge("RGB", (r, g, b))


def _draw_noise(*, w: int, h: int, rng: random.Random) -> "PILImage.Image":
    from PIL import Image

    # Generate pseudo-random RGB bytes deterministically.
    buf = bytes(rng.getrandbits(8) for _ in range(w * h * 3))
    return Image.frombytes("RGB", (w, h), buf)


def _overlay_label(img: "PILImage.Image", *, label: str) -> "PILImage.Image":
    from PIL import ImageDraw

    out = img.copy()
    draw = ImageDraw.Draw(out)

    # Simple readable label with a dark background box.
    pad = 8
    text_w, text_h = draw.textbbox((0, 0), label)[2:4]
    box = (pad, pad, pad + text_w + pad, pad + text_h + pad)
    draw.rectangle(box, fill=(0, 0, 0))
    draw.text((pad + 4, pad + 2), label, fill=(255, 255, 255))
    return out


def _iter_jobs(sizes: Iterable[Tuple[int, int]], per_size: int) -> Iterable[Tuple[int, int, int]]:
    idx = 0
    for w, h in sizes:
        for _ in range(per_size):
            yield w, h, idx
            idx += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate test images of various resolutions")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument(
        "--sizes",
        default="224x224,384x384,512x512,1024x1024,1280x720,1920x1080",
        help="Comma-separated list like 224x224,1280x720 (or '512' for square)",
    )
    ap.add_argument("--per-size", type=int, default=4, help="How many images per resolution")
    ap.add_argument(
        "--pattern",
        choices=["checker", "gradient", "noise"],
        default="checker",
        help="Image content pattern",
    )
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic seed (used for noise)")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_sizes(str(args.sizes))
    per_size = int(args.per_size)
    if per_size <= 0:
        raise SystemExit("--per-size must be > 0")

    try:
        import PIL  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Pillow is required. Install via: python3 -m pip install pillow (or use requirements-cpu.txt/requirements-cuda.txt)"
        ) from e

    rng = random.Random(int(args.seed))

    total = 0
    for w, h, idx in _iter_jobs(sizes, per_size):
        if args.pattern == "checker":
            img = _draw_checkerboard(w=w, h=h)
        elif args.pattern == "gradient":
            img = _draw_gradient(w=w, h=h)
        else:
            img = _draw_noise(w=w, h=h, rng=rng)

        label = f"{w}x{h} idx={idx} pattern={args.pattern}"
        img = _overlay_label(img, label=label)

        ext = str(args.format).lower()
        path = out_dir / f"img_{w}x{h}_{idx:05d}.{ext}"
        save_kwargs = {}
        if ext == "jpg":
            save_kwargs = {"quality": 92, "subsampling": 0}
        img.save(path, **save_kwargs)
        total += 1

    print(f"wrote {total} images under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

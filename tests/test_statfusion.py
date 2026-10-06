"""Basic tests on a synthetic document: python tests/test_statfusion.py"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from statfusion import binarize, scale_factor


def synthetic_page(h=300, w=420, seed=0):
    """Dark text on bright paper with an illumination gradient, a stain and noise."""
    rng = np.random.default_rng(seed)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    for k, line in enumerate(["Statistical fusion", "of degraded pages", "training-free"]):
        draw.text((20, 30 + 80 * k), line, fill=255, font=font)
    text = np.asarray(mask) > 128
    yy, xx = np.mgrid[0:h, 0:w]
    paper = 210 - 50 * xx / w - 15 * np.exp(-((xx - 0.7 * w) ** 2 + (yy - 0.5 * h) ** 2) / (2 * 30 ** 2))
    img = np.where(text, 40.0, paper) + rng.normal(0, 4, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8), text


def test_page():
    img, text = synthetic_page()
    out, info = binarize(img)
    found = out == 0
    tp = np.sum(found & text)
    fm = 2 * tp / (found.sum() + text.sum())
    print(f"F-measure {fm:.3f}, info {info}")
    assert fm > 0.8 and info["T_I"] < info["B_I"]


def test_scale_factor_range():
    for t in range(5, 250, 15):
        for b in range(t + 5, 256, 15):
            assert 0.0 < scale_factor(t, b) <= 1.0


if __name__ == "__main__":
    test_page()
    test_scale_factor_range()
    print("all tests passed")

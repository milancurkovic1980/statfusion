"""
Training-free statistical fusion binarization of degraded document images.

Reference implementation of the method described in
M. Ćurković, A. Ćurković, "An interpretable, training-free statistical fusion system
for degraded document image binarization".

The method has four constants (see PARAMETERS below). Everything else is computed from
the image itself:
    1. Strong edges E: non-maximum-suppressed gradient edges that are stronger than the
       magnitude-weighted median of all edges of the page.
    2. Contour intensity B_I: median intensity on E.
       Ink intensity T_I: median intensity of the darker neighbours of the edge pixels.
    3. Window radius K: median size of the connected components of E.
    4. Scale factor f: a function of t = T_I/255 and b = B_I/255.
    5. A pixel is text if I < f * L_m and I < L_a, where L_m and L_a are the median and
       the mean of the (2K+1) x (2K+1) window around the pixel.

Usage:
    python statfusion.py page.jpg page_bin.png
    python statfusion.py input_folder output_folder

    from statfusion import binarize
    binary, info = binarize(gray_uint8_image)
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# PARAMETERS (the only constants of the method, all fixed in the paper)
# ---------------------------------------------------------------------------
SIGMA = 1.6   # standard deviation (pixels) of the Gaussian smoothing applied before the Sobel gradient
GAMMA = 0.5   # selectivity of the ink samples: a darker neighbour q of an edge pixel p is accepted
              # as ink if I(q) <= d * |d|^GAMMA, with d = I(p) - I(q)
C = 2.0       # window factor: K is based on C times the diagonal of the edge components
RHO = 0.5     # upper limit of the window radius as a fraction of min(width, height)

# Fixed implementation details (not tuned): connected components use 8-connectivity
# and ignore components smaller than 5 pixels.
_MIN_COMPONENT = 5


# ---------------------------------------------------------------------------
# Step 1: strong edges
# ---------------------------------------------------------------------------
@njit(cache=True)
def _non_maximum_suppression(mag, ori):
    h, w = mag.shape
    out = np.zeros_like(mag)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = ori[y, x] * 180.0 / math.pi
            if a < 0:
                a += 180.0
            if a < 22.5 or a >= 157.5:
                n1 = mag[y, x - 1]; n2 = mag[y, x + 1]
            elif a < 67.5:
                n1 = mag[y - 1, x - 1]; n2 = mag[y + 1, x + 1]
            elif a < 112.5:
                n1 = mag[y - 1, x]; n2 = mag[y + 1, x]
            else:
                n1 = mag[y - 1, x + 1]; n2 = mag[y + 1, x - 1]
            if mag[y, x] >= n1 and mag[y, x] >= n2:
                out[y, x] = mag[y, x]
    return out


def _median_from_histogram(hist, count):
    """Lower median of the values counted in a 256-bin histogram (-1 if empty)."""
    if count <= 0:
        return -1
    acc, m = 0, 0
    while 2 * acc < count:
        acc += int(hist[m]); m += 1
    return m - 1


def _quantile(im, p):
    """p-quantile of the image intensities (fallback only)."""
    hist = np.bincount(im.ravel(), minlength=256)
    target, acc, v = int(p * im.size), 0, 0
    while v < 255 and acc + hist[v] <= target:
        acc += int(hist[v]); v += 1
    return v


def strong_edges(im):
    """Inverted 8-bit edge image e and the magnitude-weighted median e* (Eqs. 1 to 3)."""
    g = ndimage.gaussian_filter(im.astype(np.float32) / 255.0, SIGMA, mode="nearest", truncate=4.0)
    gx = ndimage.sobel(g, axis=1, mode="nearest")
    gy = ndimage.sobel(g, axis=0, mode="nearest")
    nms = _non_maximum_suppression(np.hypot(gx, gy).astype(np.float32), np.arctan2(gy, gx).astype(np.float32))
    lo, hi = float(nms.min()), float(nms.max())
    nms = (nms - lo) / (hi - lo) if hi > lo else np.zeros_like(nms)
    e = (255 - np.floor(nms * 255.0)).astype(np.int32)
    hist = np.bincount(e[e < 255].ravel(), minlength=256)[:255].astype(np.int64)
    hist = hist * (255 - np.arange(255, dtype=np.int64))          # weight w(i) = 255 - i
    half, acc, e_star = int(hist.sum()) // 2, 0, 0
    while acc < half:
        acc += int(hist[e_star]); e_star += 1
    return e, e_star - 1


# ---------------------------------------------------------------------------
# Step 2: document-level intensities
# ---------------------------------------------------------------------------
def document_intensities(im, e, e_star):
    """Contour intensity B_I (Eq. 4) and ink intensity T_I (Eqs. 5 and 6)."""
    strong = e <= e_star
    on_edges = im[strong]
    b_i = _median_from_histogram(np.bincount(on_edges, minlength=256), on_edges.size)

    a = im.astype(np.float64)
    h, w = im.shape
    core = strong[1:-1, 1:-1]
    p = a[1:-1, 1:-1]
    hist = np.zeros(256, dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            q = a[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            d = p - q
            accepted = core & (q <= d * np.power(np.abs(d), GAMMA))
            hist += np.bincount(q[accepted].astype(np.int64), minlength=256)
    t_i = _median_from_histogram(hist, int(hist.sum()))

    # fallbacks if no samples exist (did not occur on any test image)
    if t_i < 0:
        t_i = _quantile(im, 0.05)
    if b_i < 0:
        b_i = _quantile(im, 0.50)
    return max(int(t_i), 1), int(b_i)


# ---------------------------------------------------------------------------
# Step 3: window radius
# ---------------------------------------------------------------------------
def window_radius(e, e_star, shape):
    """K: median of {C * diagonal} and {pixel count} of the components of E (Eq. 7)."""
    h, w = shape
    labels, n = ndimage.label(e <= e_star, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel())[1:]
    sizes_kept, diags = [], []
    for i, sl in enumerate(ndimage.find_objects(labels)):
        if sl is None or sizes[i] < _MIN_COMPONENT:
            continue
        diags.append(int(C * math.hypot(sl[1].stop - sl[1].start - 1, sl[0].stop - sl[0].start - 1)))
        sizes_kept.append(int(sizes[i]))
    if not diags:
        return max(1, min(w, h) // 30)
    pooled = sorted(diags + sizes_kept)
    k = min(pooled[len(pooled) // 2], int(RHO * min(w, h)))
    return k if k >= 1 else max(1, min(w, h) // 30)


# ---------------------------------------------------------------------------
# Step 4: scale factor
# ---------------------------------------------------------------------------
def scale_factor(t_i, b_i):
    """f of Eqs. 8 and 9: m = max(t, 1 - b), f = m + (1 - m) / (1 + m^2 exp(m + (1/2 - t)/(m t)))."""
    t = np.float32(max(t_i, 1)) / np.float32(255.0)
    b = np.float32(b_i) / np.float32(255.0)
    m = max(max(t, np.float32(1.0) - b), np.float32(1.0 / 255.0))
    with np.errstate(over="ignore"):
        g = np.exp(np.float32(m + (1.0 / m) * (np.float32(0.5) - t) / t))
    return float(np.float32(m + (1 - m) / (1 + m * m * g)))


# ---------------------------------------------------------------------------
# Step 5: pixel decision with a sliding histogram (rows in parallel)
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _decide(pad, h, w, k, f, out):
    n = (2 * k + 1) * (2 * k + 1)
    for j in prange(h):
        hist = np.zeros(256, np.int64)
        total = np.int64(0)
        for jj in range(j, j + 2 * k + 1):
            for ii in range(0, 2 * k + 1):
                hist[pad[jj, ii]] += 1
                total += pad[jj, ii]
        for i in range(w):
            if i > 0:
                for r in range(2 * k + 1):
                    old = np.int64(pad[j + r, i - 1]); new = np.int64(pad[j + r, i + 2 * k])
                    hist[old] -= 1; hist[new] += 1
                    total += new - old
            acc = 0; l_m = 0
            while acc < n // 2:               # local median L_m
                acc += hist[l_m]; l_m += 1
            if l_m != 0:
                l_m -= 1
            # the first column reads the window centre as in the original implementation
            pixel = np.int64(pad[j, k]) if i == 0 else np.int64(pad[j + k, i + k])
            l_a = total // n                  # local mean L_a (integer)
            out[j, i] = 0 if (pixel < f * l_m and pixel < l_a) else 255


def binarize(im):
    """Binarize a 2-D uint8 grayscale image. Returns (binary image with text 0, info dict)."""
    if im.dtype != np.uint8 or im.ndim != 2:
        raise ValueError("expected a 2-D uint8 grayscale image")
    e, e_star = strong_edges(im)
    t_i, b_i = document_intensities(im, e, e_star)
    k = window_radius(e, e_star, im.shape)
    f = scale_factor(t_i, b_i)
    out = np.empty(im.shape, np.uint8)
    _decide(np.pad(im, k, mode="edge"), im.shape[0], im.shape[1], k, np.float32(f), out)
    return out, {"T_I": t_i, "B_I": b_i, "K": k, "f": f}


def load_gray(path):
    """Grayscale with luminance weights 0.299, 0.587, 0.114 (values truncated)."""
    img = Image.open(path)
    if img.mode in ("L", "1"):
        return np.asarray(img.convert("L"), dtype=np.uint8)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.floor(gray), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Training-free statistical fusion binarization")
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("output", help="output file (.png) or folder")
    a = ap.parse_args()
    src, dst = Path(a.input), Path(a.output)
    ext = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        pairs = [(p, dst / (p.stem + ".png")) for p in sorted(src.iterdir()) if p.suffix.lower() in ext]
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        pairs = [(src, dst)]
    if not pairs:
        sys.exit("no images found")
    binarize(np.full((64, 64), 200, np.uint8))           # compile the Numba code before timing
    for p, q in pairs:
        im = load_gray(p)
        t = time.perf_counter()
        out, info = binarize(im)
        dt = time.perf_counter() - t
        Image.fromarray(out).save(q)
        print(f"{p.name}: {dt:.2f} s  T_I={info['T_I']} B_I={info['B_I']} K={info['K']} f={info['f']:.3f} -> {q}")


if __name__ == "__main__":
    main()

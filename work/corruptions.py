"""CIFAR-10-C style corruption suite implemented with numpy/scipy only.

Everything here is written for the final notebook: it must run with just
Python 3.14 + numpy + scipy + scikit-learn + joblib, and no custom object
ends up inside the saved model.

Images are (N, 32, 32, 3) uint8. Each corruption takes a float batch in
[0, 1] and a severity in 1..5 (CIFAR-10-C convention) and returns float in
[0, 1].
"""

import numpy as np
from scipy import ndimage


H, W = 32, 32


def _lin(sev, lo, hi):
    """Map severity 1..5 onto the interval [lo, hi]."""
    return lo + (hi - lo) * (np.asarray(sev, dtype=np.float32) - 1.0) / 4.0


# Parameter tables follow the CIFAR-10-C benchmark (Hendrycks & Dietterich) so
# that training-time augmentations have the same severity ladder as the
# corruptions the hidden test set is built from.
GAUSS_SIGMA = [0.08, 0.12, 0.18, 0.26, 0.38]
SHOT_C = [60.0, 25.0, 12.0, 5.0, 3.0]
IMPULSE_AMOUNT = [0.03, 0.06, 0.09, 0.17, 0.27]
DEFOCUS_RADIUS = [1, 2, 3, 4, 6]
GLASS = [(0.7, 1, 1), (0.9, 2, 2), (1.1, 2, 2), (1.3, 3, 3), (1.5, 4, 4)]
MOTION = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
ZOOM_MAX = [0.11, 0.16, 0.19, 0.22, 0.25]
SNOW_AMOUNT = [0.1, 0.2, 0.3, 0.4, 0.5]
FROST_SCALE = [0.5, 0.75, 1.0, 1.25, 1.5]
FOG_COVERAGE = [0.2, 0.3, 0.4, 0.5, 0.6]
BRIGHTNESS_DELTA = [0.1, 0.2, 0.3, 0.4, 0.5]
CONTRAST_FACTOR = [0.4, 0.3, 0.2, 0.1, 0.05]
ELASTIC_ALPHA = [6.0, 9.0, 12.0, 15.0, 18.0]
PIXELATE_SCALE = [0.6, 0.5, 0.4, 0.3, 0.25]
JPEG_QUALITY = [25, 18, 15, 10, 7]


def _tab(table, sev):
    return table[int(np.asarray(sev).reshape(-1)[0]) - 1]


# ---------------------------------------------------------------- noise ----
def gaussian_noise(x, sev, rng):
    sigma = _tab(GAUSS_SIGMA, sev)
    return x + rng.normal(0.0, sigma, x.shape).astype(np.float32)


def shot_noise(x, sev, rng):
    counts = _tab(SHOT_C, sev)
    return rng.poisson(np.clip(x, 0, 1) * counts) / counts


def impulse_noise(x, sev, rng):
    amount = _tab(IMPULSE_AMOUNT, sev)
    mask = rng.random(x.shape) < amount
    salt = rng.random(x.shape) < 0.5
    out = x.copy()
    out[mask] = np.where(salt[mask], 1.0, 0.0)
    return out


# ----------------------------------------------------------------- blur ----
def _disk_kernel(radius):
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    k = ((yy**2 + xx**2) <= r * r).astype(np.float32)
    return k / k.sum()


def defocus_blur(x, sev, rng):
    kernel = _disk_kernel(_tab(DEFOCUS_RADIUS, sev))[None, :, :, None]
    return ndimage.convolve(x, kernel, mode="reflect")


def _line_kernel(size, angle_deg):
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1.0
    k = ndimage.rotate(k, angle_deg, reshape=False, order=1)
    k[k < 1e-3] = 0.0
    total = k.sum()
    return k / total if total > 0 else k


def motion_blur(x, sev, rng):
    size, angle = _tab(MOTION, sev)
    angles = np.linspace(angle, angle + 180, 6, endpoint=False)
    offsets = rng.choice(len(angles) + 1, len(x))
    out = np.empty_like(x)
    for j in range(len(angles) + 1):
        m = offsets == j
        if not m.any():
            continue
        if j == 0:
            out[m] = x[m]
        else:
            out[m] = ndimage.convolve(x[m], _line_kernel(size, angles[j - 1])[None, :, :, None], mode="reflect")
    return out


def glass_blur(x, sev, rng):
    _, delta, iters = _tab(GLASS, sev)
    d = int(delta)
    out = x
    for _ in range(int(iters)):
        off_y = rng.integers(-d, d + 1, out.shape[:-1])
        off_x = rng.integers(-d, d + 1, out.shape[:-1])
        ys = np.clip(np.arange(H)[None, :, None] + off_y, 0, H - 1)
        xs = np.clip(np.arange(W)[None, None, :] + off_x, 0, W - 1)
        rows = np.arange(len(out))[:, None, None]
        out = out[rows, ys, xs]
    return ndimage.gaussian_filter(out, sigma=(0, 0.5, 0.5, 0), mode="reflect")


def zoom_blur(x, sev, rng):
    factors = 1.0 + np.linspace(0.02, _tab(ZOOM_MAX, sev), 6)
    acc = np.zeros_like(x)
    for f in factors:
        zoomed = ndimage.zoom(x, (1, f, f, 1), order=1, mode="reflect")
        y0 = max(0, (zoomed.shape[1] - H) // 2)
        x0 = max(0, (zoomed.shape[2] - W) // 2)
        if zoomed.shape[1] >= H:
            zoomed = zoomed[:, y0 : y0 + H, x0 : x0 + W]
        else:
            pad_y = H - zoomed.shape[1]
            pad_x = W - zoomed.shape[2]
            zoomed = np.pad(
                zoomed,
                ((0, 0), (pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2), (0, 0)),
                mode="edge",
            )
        acc += zoomed
    return acc / len(factors)


# -------------------------------------------------------------- weather ----
def fog(x, sev, rng):
    strength = _tab(FOG_COVERAGE, sev) / 1.4
    small = rng.normal(0.6, 0.25, (len(x), 4, 4, 3)).astype(np.float32)
    mask = ndimage.zoom(small, (1, H / 4, W / 4, 1), order=1, mode="reflect")
    mask = np.clip(mask, 0, 1)
    fog_color = np.array([0.75, 0.78, 0.80], dtype=np.float32)
    alpha = (mask * strength).astype(np.float32)
    return x * (1 - alpha) + fog_color * alpha


def snow(x, sev, rng):
    dep = _tab(SNOW_AMOUNT, sev) / 1.6
    layer = np.zeros_like(x)
    for _ in range(3):
        flakes = rng.normal(0, 0.06 + 0.12 * dep, x.shape).astype(np.float32)
        flakes[flakes < 0.25] = 0
        layer += ndimage.gaussian_filter(flakes, sigma=(0, 0.6, 0.6, 0), mode="reflect")
    return x * (1 - dep) + (x + layer) * dep


def frost(x, sev, rng):
    pattern = rng.normal(0, 1, (len(x), 4, 4, 3)).astype(np.float32)
    pattern = ndimage.zoom(pattern, (1, H / 4, W / 4, 1), order=1, mode="reflect")
    pattern = ndimage.gaussian_filter(np.abs(pattern), sigma=(0, 1.2, 1.2, 0), mode="reflect")
    pattern = np.clip(pattern * _tab(FROST_SCALE, sev), 0, 1)
    return x * (1 - pattern * 0.6) + pattern * 0.6


# ------------------------------------------------------------- digital ----
def brightness(x, sev, rng):
    delta = _tab(BRIGHTNESS_DELTA, sev)
    sign = rng.choice([-1.0, 1.0])
    return x + sign * delta


def contrast(x, sev, rng):
    factor = _tab(CONTRAST_FACTOR, sev)
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    return (x - mean) * factor + mean


def pixelate(x, sev, rng):
    scale = _tab(PIXELATE_SCALE, sev)
    n = max(2, int(round(H * scale)))
    small = ndimage.zoom(x, (1, n / H, n / W, 1), order=1, mode="reflect")
    return ndimage.zoom(small, (1, H / n, W / n, 1), order=0, mode="reflect")


_JPEG_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


def jpeg_at_quality(x, quality):
    """JPEG round-trip at an explicit quality factor (no severity ladder)."""
    from scipy.fft import dctn, idctn

    quality = float(quality)
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    table = _JPEG_LUMA * scale / 100.0
    table_chroma = table * 1.4

    y = x * 255.0 - 128.0
    r, g, b = y[..., 0], y[..., 1], y[..., 2]
    yc = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    stacked = np.stack([yc, cb - 128.0, cr - 128.0], -1)

    blocks = stacked.reshape(len(x), H // 8, 8, W // 8, 8, 3).transpose(0, 1, 3, 5, 2, 4)
    coef = dctn(blocks, axes=(-2, -1), norm="ortho")
    q = np.stack([table, table_chroma, table_chroma])  # (3, 8, 8)
    q = np.repeat(q[None], len(x) * (H // 8) * (W // 8), axis=0).reshape(coef.shape)
    coef = np.round(coef / q) * q
    blocks = idctn(coef, axes=(-2, -1), norm="ortho")
    stacked = blocks.transpose(0, 1, 4, 2, 5, 3).reshape(len(x), H, W, 3)

    yc, cb, cr = stacked[..., 0], stacked[..., 1] + 128.0, stacked[..., 2] + 128.0
    r = yc + 1.402 * (cr - 128.0)
    g = yc - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = yc + 1.772 * (cb - 128.0)
    return (np.stack([r, g, b], -1) + 128.0) / 255.0


def jpeg_compression(x, sev, rng):
    return jpeg_at_quality(x, _tab(JPEG_QUALITY, sev))


def elastic(x, sev, rng):
    alpha = _tab(ELASTIC_ALPHA, sev)
    disp = rng.normal(0, 1, (len(x), 4, 4, 2)).astype(np.float32)
    disp = ndimage.zoom(disp, (1, H / 4, W / 4, 1), order=1, mode="reflect")
    disp = ndimage.gaussian_filter(disp, sigma=(0, 2.0, 2.0, 0), mode="reflect") * alpha
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack(
        [
            np.clip(yy[None] + disp[..., 0], 0, H - 1.001),
            np.clip(xx[None] + disp[..., 1], 0, W - 1.001),
        ],
        axis=-1,
    )
    return _bilinear(x, coords)


def _bilinear(images, coords):
    y, x = coords[..., 0], coords[..., 1]
    y0, x0 = np.floor(y).astype(np.int32), np.floor(x).astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (y - y0)[..., None], (x - x0)[..., None]
    rows = np.arange(len(images))[:, None, None]
    tl = images[rows, y0, x0]
    tr = images[rows, y0, x1]
    bl = images[rows, y1, x0]
    br = images[rows, y1, x1]
    top = tl * (1 - wx) + tr * wx
    bottom = bl * (1 - wx) + br * wx
    return top * (1 - wy) + bottom * wy


SPECKLE_C = [0.06, 0.1, 0.15, 0.2, 0.25]
GAUSSIAN_BLUR_SIGMA = [1.0, 1.4, 1.9, 2.6, 3.6]
SPATTER_AMOUNT = [0.02, 0.04, 0.07, 0.11, 0.16]
SATURATE_FACTOR = [1.15, 1.5, 2.0, 2.6, 3.2]
TRANSLATE_FRAC = [0.06, 0.1, 0.15, 0.2, 0.27]
ROTATE_DEG = [10.0, 15.0, 20.0, 25.0, 30.0]


def speckle_noise(x, sev, rng):
    c = _tab(SPECKLE_C, sev)
    return x + x * rng.normal(0, c, x.shape).astype(np.float32)


def gaussian_blur(x, sev, rng):
    sigma = _tab(GAUSSIAN_BLUR_SIGMA, sev)
    return ndimage.gaussian_filter(x, sigma=(0, sigma, sigma, 0), mode="reflect")


def spatter(x, sev, rng):
    """Dark mud-like blobs, the classic CIFAR-10-C spatter look."""
    amount = _tab(SPATTER_AMOUNT, sev)
    out = x.copy()
    for i in range(len(x)):
        n = rng.integers(3, 7)
        for _ in range(n):
            cy, cx = rng.integers(0, 32, 2)
            r = rng.integers(2, 6)
            yy, xx = np.mgrid[0:32, 0:32]
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
            out[i][m] = out[i][m] * (1 - amount * 4) + np.array([0.25, 0.2, 0.15]) * amount * 4
    return out


def saturate(x, sev, rng):
    factor = _tab(SATURATE_FACTOR, sev)
    gray = (x * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1, keepdims=True)
    return gray + (x - gray) * factor


def translate(x, sev, rng):
    frac = _tab(TRANSLATE_FRAC, sev)
    out = np.zeros_like(x)
    for i in range(len(x)):
        dy = int(rng.integers(-32, 33) * frac)
        dx = int(rng.integers(-32, 33) * frac)
        y0, y1 = max(0, dy), min(H, H + dy)
        x0, x1 = max(0, dx), min(W, W + dx)
        out[i, y0:y1, x0:x1] = x[i, y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def rotate(x, sev, rng):
    deg = _tab(ROTATE_DEG, sev)
    out = np.empty_like(x)
    for i in range(len(x)):
        angle = rng.uniform(-deg, deg)
        out[i] = ndimage.rotate(x[i], angle, reshape=False, order=1, mode="constant", cval=0.0)
    return out


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic": elastic,
    "pixelate": pixelate,
    "jpeg": jpeg_compression,
    "speckle_noise": speckle_noise,
    "gaussian_blur": gaussian_blur,
    "spatter": spatter,
    "saturate": saturate,
    "translate": translate,
    "rotate": rotate,
}

CORRUPTION_NAMES = list(CORRUPTIONS)


def corrupt_batch(images_uint8, name, sev, rng):
    """Apply one corruption to a uint8 batch; returns uint8."""
    x = images_uint8.astype(np.float32) / 255.0
    out = CORRUPTIONS[name](x, sev, rng)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def make_all(images_uint8, sev, rng):
    """Pick a random corruption type per image at a fixed severity."""
    out = np.empty_like(images_uint8)
    choice = rng.integers(0, len(CORRUPTION_NAMES), len(images_uint8))
    for k in np.unique(choice):
        m = choice == k
        out[m] = corrupt_batch(images_uint8[m], CORRUPTION_NAMES[k], sev, rng)
    return out

"""Experiment harness for the AIAA 3102 recipe search.

Fixed data split + cached features, so a classifier / capacity change can be
measured in minutes instead of by re-running the whole notebook.

Everything that ends up *inside* the pickled model is built from numpy / scipy
module-level callables and plain sklearn transformers, exactly like the
submission notebook has to be.
"""

import hashlib
import json
import os
import sys

import numpy as np
from scipy import ndimage
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corruptions import CORRUPTION_NAMES, corrupt_batch  # noqa: E402


CACHE = "work/cache"
MIXTURE = {"clean": 16000, "shift": 10000, "corr13": 10000, "corr15": 10000, "shift2": 7000}


# ---------------------------------------------------------------- registry
FEATURE_SPECS = {
    # the recipe currently in train_models.ipynb (18 filters, 8x8 grid, colour means)
    "f0": dict(angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0), cells=8, colour="mean"),
    # + per-cell L2 normalisation of the filter responses
    "f0_norm": dict(angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0), cells=8, colour="mean", norm=True),
    # + sqrt compression of the response magnitudes
    "f0_gamma": dict(angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0), cells=8, colour="mean", gamma=True),
    # wider filter bank: more orientations, more scales, larger kernels
    "fbig": dict(
        angles=12,
        sigmas=(1.0, 1.6, 2.6, 4.0),
        laplacians=(1.6, 3.0, 4.5),
        cells=8,
        colour="mean",
    ),
    "fbig_norm": dict(
        angles=12,
        sigmas=(1.0, 1.6, 2.6, 4.0),
        laplacians=(1.6, 3.0, 4.5),
        cells=8,
        colour="mean",
        norm=True,
    ),
    "fbig_col": dict(
        angles=12,
        sigmas=(1.0, 1.6, 2.6, 4.0),
        laplacians=(1.6, 3.0, 4.5),
        cells=8,
        colour="mean+std",
    ),
    "fbig_norm_col": dict(
        angles=12,
        sigmas=(1.0, 1.6, 2.6, 4.0),
        laplacians=(1.6, 3.0, 4.5),
        cells=8,
        colour="mean+std",
        norm=True,
        gamma=True,
    ),
    # two-layer learned-filter features, alone and combined with the oriented bank
    "pc": dict(pcanet="pc8", pcanet_cells=4, colour="mean", cells=8),
    "pc_or": dict(
        pcanet="pc8",
        pcanet_cells=4,
        colour="mean",
        cells=8,
        angles=8,
        sigmas=(1.2, 2.2),
        laplacians=(1.6, 3.0),
    ),
    "pc16": dict(pcanet="pc16", pcanet_cells=4, colour="mean", cells=8),
    "pc16_or": dict(
        pcanet="pc16",
        pcanet_cells=4,
        colour="mean",
        cells=8,
        angles=8,
        sigmas=(1.2, 2.2),
        laplacians=(1.6, 3.0),
    ),
    # pooling granularity sweep for the winning two-layer feature set
    "pc_or_c8": dict(
        pcanet="pc8", pcanet_cells=8, colour="mean", cells=8,
        angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
    ),
    "pc_or_c2": dict(
        pcanet="pc8", pcanet_cells=2, colour="mean", cells=8,
        angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
    ),
    # patch-size sweep
    "pc3_or": dict(
        pcanet="pc8p3", pcanet_cells=4, colour="mean", cells=8,
        angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
    ),
    "pc7_or": dict(
        pcanet="pc8p7", pcanet_cells=4, colour="mean", cells=8,
        angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
    ),
    # fewer stage-1 maps, more stage-2 filters each
    "pc4x16_or": dict(
        pcanet="pc4n16", pcanet_cells=4, colour="mean", cells=8,
        angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
    ),
}


# --------------------------------------------------------------- plumbing
def _ft(func, **kw):
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


def reshape(shape):
    return _ft(np.reshape, shape=shape)


def ksize(sigma):
    """Odd kernel size wide enough for the Gaussian's support."""
    return 7 if sigma <= 2.5 else 11


# ---------------------------------------------------------------- kernels
def oriented_derivative_kernel(angle_deg, sigma, size):
    axis = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    gauss = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    gx = -(xx / sigma**2) * gauss
    gy = -(yy / sigma**2) * gauss
    theta = np.deg2rad(angle_deg)
    kernel = np.cos(theta) * gx + np.sin(theta) * gy
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-8
    return kernel.astype(np.float32)[None, :, :, None]


def laplacian_kernel(sigma, size):
    axis = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    r2 = xx**2 + yy**2
    kernel = (r2 - 2.0 * sigma**2) / sigma**4 * np.exp(-r2 / (2.0 * sigma**2))
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-8
    return kernel.astype(np.float32)[None, :, :, None]


# -------------------------------------------------------------- branches
def filter_pool_branch(kernel, cells, gamma=False):
    """|filter response| averaged over channels, mean-pooled over a cells x cells grid."""
    size = 32 // cells
    steps = [("convolve", _ft(ndimage.convolve, weights=kernel)), ("abs", _ft(np.abs))]
    if gamma:
        # sqrt compression: keeps the ordering, shrinks the dynamic range of the
        # large responses so the PCA is not dominated by a few strong edges
        steps.append(("gamma", _ft(np.sqrt)))
    steps.append(("channel_mean", _ft(np.mean, axis=3)))
    steps += [
        ("reshape", reshape((-1, cells, size, cells, size))),
        ("pool", _ft(np.mean, axis=(2, 4))),
        ("flatten", reshape((-1, cells * cells))),
    ]
    return Pipeline(steps)


def colour_branch(cells, std=False):
    """Per-cell colour statistics, kept in the feature vector so the model sees
    the coarse colour layout (which the oriented filters throw away)."""
    size = 32 // cells

    def pooled(stat):
        return Pipeline(
            [
                ("reshape", reshape((-1, cells, size, cells, size, 3))),
                ("pool", _ft(stat, axis=(2, 4))),
                ("flatten", reshape((-1, cells * cells * 3))),
            ]
        )

    if std:
        return FeatureUnion([("mean", pooled(np.mean)), ("std", pooled(np.std))], n_jobs=1)
    return pooled(np.mean)


def build_features(cfg):
    """Raw uint8 (N, 32, 32, 3) -> pooled oriented-filter features."""
    cells = cfg.get("cells", 8)
    as_float32 = lambda: _ft(np.asarray, dtype=np.float32)  # noqa: E731
    outer = []

    if cfg.get("angles"):
        filters = []
        for sigma in cfg["sigmas"]:
            for angle in np.linspace(0, 180, cfg["angles"], endpoint=False):
                filters.append(
                    (
                        f"o{int(sigma * 10)}_{int(angle)}",
                        filter_pool_branch(oriented_derivative_kernel(angle, sigma, ksize(sigma)), cells, cfg.get("gamma", False)),
                    )
                )
        for sigma in cfg["laplacians"]:
            filters.append(
                (f"l{int(sigma * 10)}", filter_pool_branch(laplacian_kernel(sigma, ksize(sigma)), cells, cfg.get("gamma", False)))
            )

        n_filt = len(filters)
        inner = [("union", FeatureUnion(filters, n_jobs=1))]
        if cfg.get("norm"):
            inner.append(
                (
                    "both",
                    FeatureUnion(
                        [
                            ("raw", FunctionTransformer()),
                            (
                                "norm",
                                Pipeline(
                                    [
                                        ("as3", reshape((-1, n_filt, cells * cells))),
                                        ("transpose", _ft(np.transpose, axes=(0, 2, 1))),
                                        ("rows", reshape((-1, n_filt))),
                                        ("normalize", Normalizer(norm="l2")),
                                        ("flatten", reshape((-1, n_filt * cells * cells))),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=1,
                    ),
                )
            )
        outer.append(
            ("oriented", Pipeline([("as_float32", as_float32()), ("filters", Pipeline(inner))]))
        )

    if cfg.get("pcanet"):
        from pcanet_lib import build_branches, ensure_filters

        filters = ensure_filters(cfg["pcanet"])
        pc = cfg.get("pcanet_cells", 4)
        branches = build_branches(filters, pc, cfg.get("pcanet_l1_pool", True))
        outer.append(
            (
                "pcanet",
                Pipeline(
                    [
                        ("as_float32", as_float32()),
                        ("gray", _ft(np.mean, axis=3)),
                        ("maps", FeatureUnion(branches, n_jobs=1)),
                    ]
                ),
            )
        )

    if cfg.get("colour"):
        outer.append(
            (
                "colour",
                Pipeline(
                    [
                        ("as_float32", as_float32()),
                        ("c", colour_branch(cells, std=cfg["colour"] == "mean+std")),
                    ]
                ),
            )
        )

    return Pipeline([("features", FeatureUnion(outer, n_jobs=1))])


def feature_dim(cfg):
    cells = cfg.get("cells", 8)
    dim = 0
    if cfg.get("angles"):
        n_filt = len(cfg["sigmas"]) * cfg["angles"] + len(cfg["laplacians"])
        dim += n_filt * cells * cells * (2 if cfg.get("norm") else 1)
    if cfg.get("pcanet"):
        from pcanet_lib import branches_dim, ensure_filters

        pc = cfg.get("pcanet_cells", 4)
        dim += branches_dim(ensure_filters(cfg["pcanet"]), pc, cfg.get("pcanet_l1_pool", True))
    if cfg.get("colour"):
        dim += cells * cells * 3 * (2 if cfg["colour"] == "mean+std" else 1)
    return dim


# ------------------------------------------------------------------- data
def random_flip_shift(images, rng, max_shift=3):
    x = images.copy()
    flip = rng.random(len(x)) < 0.5
    if flip.any():
        x[flip] = x[flip][:, :, ::-1]
    dy = rng.integers(-max_shift, max_shift + 1, len(x))
    dx = rng.integers(-max_shift, max_shift + 1, len(x))
    rows = np.arange(len(x))[:, None, None]
    ys = np.clip(np.arange(32)[None, :, None] + dy[:, None, None], 0, 31)
    xs = np.clip(np.arange(32)[None, None, :] + dx[:, None, None], 0, 31)
    return x[rows, ys, xs]


def make_corrupted(images, rng, sev_lo, sev_hi):
    out = np.empty_like(images)
    choice = rng.integers(0, len(CORRUPTION_NAMES), len(images))
    sev = rng.integers(sev_lo, sev_hi + 1, len(images))
    for k in np.unique(choice):
        for s in np.unique(sev[choice == k]):
            m = (choice == k) & (sev == s)
            out[m] = corrupt_batch(images[m], CORRUPTION_NAMES[k], int(s), rng)
    return out


def cutout(images, rng, size=8):
    """Random erasing: blank one square patch per image."""
    out = images.copy()
    n = len(images)
    ys = rng.integers(0, 32 - size + 1, n)
    xs = rng.integers(0, 32 - size + 1, n)
    for i in range(n):
        out[i, ys[i] : ys[i] + size, xs[i] : xs[i] + size] = 0
    return out


def jitter(images, rng, max_bright=0.15, max_contrast=0.30):
    """Mild brightness / contrast jitter, independent per image."""
    x = images.astype(np.float32) / 255.0
    b = rng.uniform(-max_bright, max_bright, (len(x), 1, 1, 1)).astype(np.float32)
    c = (1.0 + rng.uniform(-max_contrast, max_contrast, (len(x), 1, 1, 1))).astype(np.float32)
    m = x.mean((1, 2, 3), keepdims=True)
    return np.clip(((x - m) * c + m + b) * 255.0, 0, 255).astype(np.uint8)


def mix_corruptions(images, rng, sev_lo, sev_hi, p_second=0.5):
    """Every image gets one corruption; a random subset gets a second one on top."""
    out = make_corrupted(images, rng, sev_lo, sev_hi)
    m = rng.random(len(out)) < p_second
    if m.any():
        out[m] = make_corrupted(out[m], rng, sev_lo, sev_hi)
    return out


def load_raw():
    X = np.load("CIFAR10_Data/x_train.npy")
    y = np.load("CIFAR10_Data/y_train.npy").ravel()
    Xv = np.load("validation_set_to_students/images.npy")
    yv = np.load("validation_set_to_students/labels.npy")
    return X, y, Xv, yv


def build_mixture(X, y, seed=42, sizes=None, aug=None):
    """Deterministic mixture, same shape as the submission recipe.

    `aug` switches on the extra augmentation levers:
      cutout        random erasing on the shifted copies
      jitter        brightness/contrast jitter on the shifted copies
      double        50% of the corrupted copies get a second corruption
      sev_hi15      severity ceiling for the second corrupted component
    """
    sizes = sizes or MIXTURE
    aug = aug or {}
    rng = np.random.default_rng(seed)
    n_clean = sizes["clean"]
    X_pool, y_pool = X[:n_clean], y[:n_clean]
    idx_s = rng.choice(n_clean, sizes["shift"], replace=False)
    idx_13 = rng.choice(n_clean, sizes["corr13"], replace=False)
    idx_15 = rng.choice(n_clean, sizes["corr15"], replace=False)
    idx_2 = rng.choice(n_clean, sizes.get("shift2", 0), replace=False)

    X_shift = random_flip_shift(X_pool[idx_s], rng)
    if aug.get("jitter"):
        X_shift = jitter(X_shift, rng)
    if aug.get("cutout"):
        X_shift = cutout(X_shift, rng, size=aug.get("cutout", 8))

    corr13 = (mix_corruptions if aug.get("double") else make_corrupted)(
        X_pool[idx_13], rng, 1, 3
    )
    corr15 = (mix_corruptions if aug.get("double") else make_corrupted)(
        X_pool[idx_15], rng, 1, aug.get("sev_hi15", 5)
    )
    X_shift2 = random_flip_shift(X_pool[idx_2], rng)
    parts = {
        "clean": (X_pool, y_pool),
        "shift": (X_shift, y_pool[idx_s]),
        "corr13": (corr13, y_pool[idx_13]),
        "corr15": (corr15, y_pool[idx_15]),
        "shift2": (X_shift2, y_pool[idx_2]),
    }
    return parts


# ---------------------------------------------------------------- caching
def features_for(cfg, parts, tag=None):
    """Extract (and cache) the feature matrix of every mixture component."""
    key = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    # the mixture itself takes part in the key: same features, different training
    # set must never share a cache entry
    digest = hashlib.sha1()
    for name in sorted(parts):
        digest.update(name.encode())
        digest.update(parts[name][0].tobytes())
    key = f"{tag or key}_{key}_{digest.hexdigest()[:8]}"
    out = {}
    feats = build_features(cfg)
    for name, (images, _) in parts.items():
        path = os.path.join(CACHE, key, f"{name}.npy")
        if os.path.exists(path):
            out[name] = np.load(path)
            print(f"    cached {name:7s} {out[name].shape}", flush=True)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        F = feats.transform(images).astype(np.float32)
        np.save(path, F)
        out[name] = F
        print(f"    built  {name:7s} {F.shape}", flush=True)
    return feats, out

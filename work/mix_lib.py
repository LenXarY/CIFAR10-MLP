"""A mixed-channel two-layer bank: stage-1 filters, a learned 1x1 channel mix,
rectification, then pooling.

The plain PCANet-lite bank rectifies each stage-1 map independently and only then
concatenates, so nothing ever sees a *sum* of channels. A convolutional network
does the opposite - it sums channels first and rectifies the sum - and that is
the part of its structure a shallow bank cannot express. The mixing matrix is
learned unsupervised by PCA over the stage-1 responses, so nothing outside
NumPy / SciPy / scikit-learn is involved, and every step below is a plain
module-level callable that survives pickling.
"""

import os

import numpy as np
from scipy import ndimage
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


def _ft(func, **kw):
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


MIX_STORE = {}


def reshape(shape):
    return _ft(np.reshape, shape=shape)


def stage1_pool(cells, tag):
    size = 32 // cells
    return [
        (f"{tag}_reshape", reshape((-1, cells, size, cells, size))),
        (f"{tag}_pool", _ft(np.mean, axis=(2, 4))),
        (f"{tag}_flat", reshape((-1, cells * cells))),
    ]


def stage1_branch(kernel, cells, tag):
    """conv -> ReLU -> mean-pool to a cells x cells grid, flattened."""
    return Pipeline([
        (f"{tag}_conv", _ft(ndimage.convolve, weights=kernel, mode="reflect", axes=(1, 2))),
        (f"{tag}_relu", _ft(np.clip, a_min=0.0, a_max=None)),
    ] + stage1_pool(cells, tag))


def _maps_union(kernels, cells):
    return FeatureUnion(
        [(f"m{i}", stage1_branch(k, cells, f"m{i}")) for i, k in enumerate(kernels)],
        n_jobs=1,
    )


def build_branches(kernels, mixing, cells1=8, cells2=4):
    """Returns (stage-1 only, channel-mixed) branch lists."""
    k1 = len(kernels)
    l = mixing.shape[1]
    s2 = cells1 // cells2
    return [
        (
            "stage1",
            Pipeline([("maps", _maps_union(kernels, cells1))]),
        ),
        (
            "mixed",
            Pipeline([
                ("maps", _maps_union(kernels, cells1)),
                ("split", reshape((-1, k1, cells1 * cells1))),
                ("transpose", _ft(np.transpose, axes=(0, 2, 1))),
                ("rows", reshape((-1, k1))),
                ("mix", _ft(np.dot, b=mixing)),
                ("relu", _ft(np.clip, a_min=0.0, a_max=None)),
                ("back", reshape((-1, cells1, cells1, l))),
                ("pool_reshape", reshape((-1, cells2, s2, cells2, s2, l))),
                ("pool_mean", _ft(np.mean, axis=(2, 4))),
                ("flat", reshape((-1, cells2 * cells2 * l))),
            ]),
        ),
    ]


def learn_mixing(images, kernels, cells1=8, n_sample=1500, n_components=64, seed=0):
    """PCA over the stage-1 responses gives the 1x1 mixing matrix (k1, l)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(images), min(n_sample, len(images)), replace=False))
    gray = images[idx].mean(3).astype(np.float32)
    size = 32 // cells1
    feats = []
    for k in kernels:
        r = ndimage.convolve(gray, k, mode="reflect", axes=(1, 2))
        np.clip(r, 0.0, None, out=r)
        feats.append(r.reshape(len(gray), cells1, size, cells1, size).mean((2, 4)))
    stacked = np.stack(feats, -1)
    x = stacked.reshape(-1, len(kernels)).astype(np.float64)
    x -= x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:n_components].T.astype(np.float32)


def save(store, path):
    np.savez_compressed(path, kernels=MIX_STORE[store]["kernels"],
                        mixing=MIX_STORE[store]["mixing"])


def ensure(name, cache_dir="work/cache"):
    """Load a learned mixing bank from disk on first use."""
    if name not in MIX_STORE:
        with np.load(os.path.join(cache_dir, f"{name}.npz")) as z:
            MIX_STORE[name] = {"kernels": z["kernels"], "mixing": z["mixing"]}
    return MIX_STORE[name]


def build_features(kernels, mixing, cells1=8, cells2=4, colour_cells=8,
                   angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0)):
    """Raw uint8 (N, 32, 32, 3) -> concatenated feature vector."""
    from lab_lib import (colour_branch, filter_pool_branch, ksize, laplacian_kernel,
                         oriented_derivative_kernel)

    oriented = []
    for sigma in sigmas:
        for angle in np.linspace(0, 180, angles, endpoint=False):
            oriented.append((
                f"o{int(sigma * 10)}_{int(angle)}",
                filter_pool_branch(oriented_derivative_kernel(angle, sigma, ksize(sigma)),
                                   colour_cells),
            ))
    for sigma in laplacians:
        oriented.append((f"l{int(sigma * 10)}",
                         filter_pool_branch(laplacian_kernel(sigma, ksize(sigma)), colour_cells)))

    outer = [("mixed_bank", Pipeline([
        ("as_float32", _ft(np.asarray, dtype=np.float32)),
        ("gray", _ft(np.mean, axis=3)),
        ("union", FeatureUnion(build_branches(kernels, mixing, cells1, cells2), n_jobs=1)),
    ]))]
    outer.append(("oriented", Pipeline([
        ("as_float32", _ft(np.asarray, dtype=np.float32)),
        ("filters", FeatureUnion(oriented, n_jobs=1)),
    ])))
    outer.append(("colour", Pipeline([
        ("as_float32", _ft(np.asarray, dtype=np.float32)),
        ("cells", colour_branch(colour_cells)),
    ])))
    return Pipeline([("features", FeatureUnion(outer, n_jobs=1))])

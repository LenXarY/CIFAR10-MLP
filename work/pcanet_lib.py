"""A two-layer, learned-filter feature extractor ("PCANet-lite").

The single-layer oriented-filter bank is saturated: widening it from 18 to 51
filters, adding per-cell L2 normalisation and adding sqrt compression all left
the validation accuracy flat or slightly lower (see work/lab_results.jsonl).
What that bank cannot express is a *composition* - a second nonlinear stage on
top of the first one.

Stage 1   learn `n_l1` PCA filters from 5x5 grayscale patches of the training
          images, convolve, rectify.
Stage 2   learn `n_l2` PCA filters per stage-1 map from 5x5 patches of that
          map, convolve, rectify, and block-average-pool each response map.

Everything applied at prediction time is a composition of numpy / scipy
module-level callables, so it pickles by reference exactly like the existing
feature block: the learned filters are plain arrays carried in `kw_args`.

The stage-2 filters of a stage-1 map are applied *inside* that map's own
branch, so the extractor never materialises an (N, 32, 32, n_l1) array.
"""

import os

import numpy as np
from scipy import ndimage
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


def _ft(func, **kw):
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


def reshape(shape):
    return _ft(np.reshape, shape=shape)


FILTER_STORE = {}


# ------------------------------------------------------------ filter learning
def sample_patches(maps, patch, n_patch, rng, chunk=20000):
    """(n_img, H, W) float32 -> (n_patch, patch*patch) contrast-normalised patches."""
    windows = np.lib.stride_tricks.sliding_window_view(maps, (patch, patch), axis=(1, 2))
    n_img, ny, nx = windows.shape[:3]
    rows = rng.integers(0, n_img, n_patch)
    ys = rng.integers(0, ny, n_patch)
    xs = rng.integers(0, nx, n_patch)
    patches = np.zeros((n_patch, patch * patch), np.float32)
    for i in range(0, n_patch, chunk):
        sl = slice(i, min(i + chunk, n_patch))
        block = windows[rows[sl], ys[sl], xs[sl]]
        block = block.reshape(len(block), -1)
        block -= block.mean(1, keepdims=True)
        var = block.var(1, keepdims=True)
        keep = var.ravel() > 1e-3
        block[keep] /= np.sqrt(var[keep])
        patches[sl] = block
    return patches


def pca_filters(patches, n_components):
    """Top principal directions of a patch sample, shaped (n, patch, patch)."""
    x = patches.astype(np.float64)
    x -= x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    side = int(round(np.sqrt(patches.shape[1])))
    return vt[:n_components].reshape(n_components, side, side).astype(np.float32)


def kmeans_filters(patches, n_components, seed=0):
    """Dictionary filters from scikit-learn k-means (Coates & Ng style).

    Same role as `pca_filters`, but the atoms are cluster centroids rather than
    principal directions; k-means atoms are usually better localised edge/blob
    detectors and, unlike a convolution stack, this stays entirely inside
    scikit-learn.
    """
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_components,
        batch_size=4096,
        n_init=3,
        max_iter=200,
        random_state=seed,
    ).fit(patches)
    side = int(round(np.sqrt(patches.shape[1])))
    return km.cluster_centers_.reshape(n_components, side, side).astype(np.float32)


def learn(X_train, n_l1=8, n_l2=8, patch=5, n_patch=120000, n_images=8000, seed=0,
          method="pca"):
    """Return (l1_filters, [l2_filters of each l1 map]).

    `method="pca"` reproduces the original PCANet-lite bank; `method="kmeans"`
    learns the same two-layer bank with scikit-learn's MiniBatchKMeans instead.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_train), min(n_images, len(X_train)), replace=False))
    gray = X_train[idx].mean(3).astype(np.float32)

    def make(atoms, k, s):
        return kmeans_filters(atoms, k, seed=s) if method == "kmeans" else pca_filters(atoms, k)

    l1 = make(sample_patches(gray, patch, n_patch, rng), n_l1, seed)

    l2 = []
    for k in range(n_l1):
        resp = ndimage.convolve(gray, l1[k], mode="reflect", axes=(1, 2))
        np.maximum(resp, 0, out=resp)
        l2.append(make(sample_patches(resp, patch, n_patch, rng), n_l2, seed + 1 + k))
    return l1, l2


def save(filters, path):
    l1, l2 = filters
    np.savez_compressed(path, l1=l1, **{f"l2_{i}": a for i, a in enumerate(l2)})


def load(path):
    with np.load(path) as z:
        l1 = z["l1"]
        n = len([k for k in z.files if k.startswith("l2_")])
        l2 = [z[f"l2_{i}"] for i in range(n)]
    return l1, l2


def register(filters, name):
    FILTER_STORE[name] = filters
    return name


def ensure_filters(name, cache_dir="work/cache"):
    """Return the registered filter bank, loading it from disk on first use."""
    if name not in FILTER_STORE:
        FILTER_STORE[name] = load(os.path.join(cache_dir, f"pcanet_{name}.npz"))
    return FILTER_STORE[name]


# ------------------------------------------------------------- the transform
def _pool(cells, mode="mean"):
    size = 32 // cells
    return Pipeline(
        [
            ("reshape", reshape((-1, cells, size, cells, size))),
            ("pool", _ft(np.max if mode == "max" else np.mean, axis=(2, 4))),
            ("flatten", reshape((-1, cells * cells))),
        ]
    )


def _response(kernel, cells, mode="mean"):
    return Pipeline(
        [
            ("convolve", _ft(ndimage.convolve, weights=kernel, mode="reflect", axes=(1, 2))),
            ("relu", _ft(np.clip, a_min=0.0, a_max=None)),
            ("pool", _pool(cells, mode)),
        ]
    )


def build_branches(filters, cells=4, l1_pool=True, mode="mean"):
    """Branches taking an (N, 32, 32) grayscale batch."""
    l1, l2 = filters
    branches = []
    for i, kernel in enumerate(l1):
        head = [(f"l2_{j}", _response(k, cells, mode)) for j, k in enumerate(l2[i])]
        if l1_pool:
            head.append(("l1", _pool(cells, mode)))
        branches.append(
            (
                f"map{i}",
                Pipeline(
                    [
                        ("convolve", _ft(ndimage.convolve, weights=kernel, mode="reflect", axes=(1, 2))),
                        ("relu", _ft(np.clip, a_min=0.0, a_max=None)),
                        ("head", FeatureUnion(head, n_jobs=1)),
                    ]
                ),
            )
        )
    return branches


def branches_dim(filters, cells=4, l1_pool=True):
    l1, l2 = filters
    return len(l1) * (len(l2[0]) * cells * cells + (cells * cells if l1_pool else 0))

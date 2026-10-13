"""Hand-built image feature extractors that survive pickling.

Constraint: Model_Evaluation.ipynb only imports numpy/joblib/sklearn, so the
saved pipeline may only reference callables reachable from those modules (or
from scipy, which pickle imports on load). Everything here is a composition of
numpy / scipy.ndimage / scikit-learn transformers.
"""

import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _ft(func, **kw):
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


def reshape(shape):
    return _ft(np.reshape, shape=shape)


# ------------------------------------------------------------------ kernels
def oriented_derivative_kernel(angle_deg, sigma, size=7):
    """Derivative of a Gaussian steered to `angle_deg`, shaped (1, size, size, 1)."""
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


def laplacian_kernel(sigma, size=7):
    """Normalised Laplacian of Gaussian, shaped (1, size, size, 1)."""
    axis = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    r2 = xx**2 + yy**2
    kernel = (r2 - 2.0 * sigma**2) / sigma**4 * np.exp(-r2 / (2.0 * sigma**2))
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-8
    return kernel.astype(np.float32)[None, :, :, None]


# --------------------------------------------------------------- pipelines
def filter_pool_branch(kernel, cells, gray=False):
    """Per-channel filter magnitude, averaged over channels, pooled over cells."""
    size = 32 // cells
    steps = [("convolve", _ft(ndimage.convolve, weights=kernel[..., 0] if gray else kernel))]
    steps.append(("abs", _ft(np.abs)))
    if not gray:
        steps.append(("channel_mean", _ft(np.mean, axis=3)))
    steps += [
        ("reshape", reshape((-1, cells, size, cells, size))),
        ("pool", _ft(np.mean, axis=(2, 4))),
        ("flatten", reshape((-1, cells * cells))),
    ]
    return Pipeline(steps)


def filter_pool_pyramid_branch(kernel, cells_list=(8, 4)):
    """One convolution, pooled at several cell sizes (spatial pyramid)."""
    pools = []
    for cells in cells_list:
        size = 32 // cells
        pools.append(
            (
                f"p{cells}",
                Pipeline(
                    [
                        ("reshape", reshape((-1, cells, size, cells, size))),
                        ("pool", _ft(np.mean, axis=(2, 4))),
                        ("flatten", reshape((-1, cells * cells))),
                    ]
                ),
            )
        )
    return Pipeline(
        [
            ("convolve", _ft(ndimage.convolve, weights=kernel)),
            ("abs", _ft(np.abs)),
            ("channel_mean", _ft(np.mean, axis=3)),
            ("pyramid", FeatureUnion(pools, n_jobs=1)),
        ]
    )


def colour_branch(cells):
    """Mean colour of every cell (keeps the low-frequency colour layout)."""
    size = 32 // cells
    return Pipeline(
        [
            ("reshape", reshape((-1, cells, size, cells, size, 3))),
            ("pool", _ft(np.mean, axis=(2, 4))),
            ("flatten", reshape((-1, cells * cells * 3))),
        ]
    )


def hog_branches(cells=4, angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
                 pyramid=False, cells_list=(8, 4), raw_union=False, gray=False,
                 colour=True):
    branches = []
    for sigma in sigmas:
        for angle in np.linspace(0, 180, angles, endpoint=False):
            name = f"or{int(sigma * 10)}_{int(angle)}"
            kernel = oriented_derivative_kernel(angle, sigma)
            if pyramid:
                branches.append((name, filter_pool_pyramid_branch(kernel, cells_list)))
            else:
                branches.append((name, filter_pool_branch(kernel, cells, gray=gray)))
    for sigma in laplacians:
        kernel = laplacian_kernel(sigma)
        if pyramid:
            branches.append((f"lap{int(sigma * 10)}", filter_pool_pyramid_branch(kernel, cells_list)))
        else:
            branches.append((f"lap{int(sigma * 10)}", filter_pool_branch(kernel, cells, gray=gray)))
    if colour:
        branches.append(("colour", colour_branch(cells)))
    if raw_union:
        branches.append(("raw", Pipeline([("flatten", reshape((-1, 32 * 32 * 3)))])))
    return branches


def build_hog_model(cells=4, angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0),
                    hidden=(768,), alpha=1e-4, epochs=80, seed=42, pca_dim=None,
                    pyramid=False, cells_list=(8, 4), raw_union=False, log1p=False,
                    gray=False):
    """Raw uint8 (N, 32, 32, 3) -> pooled oriented-filter features -> MLP."""
    steps = [("as_float32", _ft(np.asarray, dtype=np.float32))]
    if gray:
        # convolution is linear, so mean over channels then convolve ==
        # convolve per channel then mean, at one third of the cost
        hog = Pipeline([
            ("gray", _ft(np.mean, axis=3)),
            ("union", FeatureUnion(
                hog_branches(cells, angles, sigmas, laplacians, pyramid, cells_list, False, True, False),
                n_jobs=1,
            )),
        ])
        steps.append((
            "features",
            FeatureUnion([("hog", hog), ("colour", colour_branch(cells))], n_jobs=1),
        ))
    else:
        steps.append((
            "features",
            FeatureUnion(
                hog_branches(cells, angles, sigmas, laplacians, pyramid, cells_list, raw_union, False),
                n_jobs=1,
            ),
        ))
    if log1p:
        steps.append(("log1p", _ft(np.log1p)))
    steps.append(("standardize", StandardScaler()))
    if pca_dim:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
    steps.append(
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=hidden,
                alpha=alpha,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=epochs,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def hog_feature_dim(cells=4, angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0)):
    return (len(sigmas) * angles + len(laplacians)) * cells * cells + cells * cells * 3

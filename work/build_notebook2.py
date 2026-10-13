"""Generate the final train_models.ipynb (two-layer learned-filter recipe).

The corruption suite is copied verbatim from the previous, already validated
notebook so that only the parts that changed are re-derived.
"""

import nbformat as nbf


OLD = nbf.read("train_models.ipynb", as_version=4)
CORRUPTIONS_SRC = next(
    c.source
    for c in OLD.cells
    if c.cell_type == "code" and "CIFAR-10-C style corruption suite" in c.source
)


INTRO = """# AIAA 3102 — Assignment 1: CIFAR-10 Image Classification

## Running environment

The submitted `.joblib` files were produced with the environment required by the
course, and this notebook is meant to be re-run in that same environment:

| Package | Version |
| --- | --- |
| Python | 3.14.6 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |

## Data paths

* `CIFAR10_Data/x_train.npy`, `CIFAR10_Data/y_train.npy` — 50,000 labelled training images
* `validation_set_to_students/images.npy`, `labels.npy` — 5,000 labelled validation images

## Final method and why it was chosen

**The problem with raw pixels.** An MLP on flattened pixels tops out near 51% clean
accuracy here, so everything depends on the feature representation.

**Features: a two-layer learned filter bank plus an oriented-filter bank.**

*Stage 1* learns 8 principal-component filters from 5x5 grayscale patches of the training
images. *Stage 2* learns, for each stage-1 response map, 8 further principal-component
filters from 5x5 patches of that map; each stage-2 response is rectified and
average-pooled over a 4x4 grid of 8x8-pixel blocks. That is a fixed-weight two-layer
convolutional network — the composition of two nonlinear stages is what a single filter
bank cannot express — and it costs nothing at prediction time beyond two
`scipy.ndimage.convolve` banks.

The two-layer bank (1,152 values) is concatenated with a steerable
derivative-of-Gaussian bank (8 orientations x 2 scales, plus two Laplacian-of-Gaussian
filters, each rectified and pooled on an 8x8 grid: 1,152 values) and with per-cell mean
colour (192 values), giving 2,496 features.

Measured on the validation set with seed 42 and everything else held fixed
(reduced screening budget, single model, horizontal-flip TTA):

| Feature representation | Validation accuracy |
| --- | --- |
| oriented-filter bank only (1,344 values) | 54.9% |
| 51-filter oriented bank + per-cell L2 normalisation (6,912 values) | 55.1% |
| **two-layer learned bank + oriented bank (2,496 values)** | **57.7%** |

Widening the oriented bank from 18 to 51 filters, adding per-cell normalisation and
adding sqrt compression all moved the result by less than a point: that bank is
saturated. The two-layer bank is what produced a real gain, which is why it is the core
of this solution.

**Classifier.** An MLP (1,024 hidden units, Adam, `alpha=1e-4`) on the principal
components of the standardised features.

**Ensemble: bagging plus test-time augmentation.** Each submitted model is a soft-voting
ensemble of `n_members` MLPs that all see the same augmented training set (clean,
flipped/translated and corrupted copies) but start from different random initialisations.
A shared `StandardScaler` + `PCA` is fitted once and reused by every member, so the
projection is stored only once in the file. Each member is also present a second time
behind a horizontal-flip transform: because the flipped member shares the *same fitted*
head object, the flip view is test-time augmentation that costs no extra model size.

Screening runs with the same recipe at reduced scale:

| Configuration | Validation accuracy |
| --- | --- |
| one MLP, no flip TTA | 57.9% |
| 3 bagged members + flip TTA | 59.4% |
| 6 bagged members + flip TTA | 59.7% |
| 5 members, each on its own data mixture (previous recipe) | 56.4% |

Members specialised on a single data mixture were clearly worse than members trained on
the union of the mixtures, which is why the final recipe bags over the full augmented
set rather than splitting it.

**Augmentation.** Corruptions follow the CIFAR-10-C benchmark definitions (the 15
classical types plus the four extra CIFAR-10-C types and geometric shifts), implemented
with NumPy/SciPy only. The hidden test set mixes original and corrupted images and its
corruptions may differ in type or strength from the validation set, so the training set
deliberately contains clean, geometrically shifted and corrupted copies. Augmentation
only affects the training data, so it is never part of the pickled model.

**Seeds.** The pipeline, mixtures and hyper-parameters are fixed; the three submitted
models are trained from scratch with the random seeds 42, 2026 and 3102.

**Runtime.** Run All learns the two filter banks, extracts features for every mixture
and trains one MLP per member per seed on a CPU. Nothing here uses a GPU.
"""


ENV_SRC = '''import os
import time

import joblib
import numpy as np
import scipy
import sklearn

print("Python  ", os.sys.version.split()[0])
print("NumPy   ", np.__version__)
print("SciPy   ", scipy.__version__)
print("sklearn ", sklearn.__version__)
print("joblib  ", joblib.__version__)
'''


DATA_SRC = '''DATA_DIR = "CIFAR10_Data"
VAL_DIR = "validation_set_to_students"
MODEL_FILES = {
    42: "model_seed42.joblib",
    2026: "model_seed2026.joblib",
    3102: "model_seed3102.joblib",
}

X_train_images = np.load(os.path.join(DATA_DIR, "x_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).ravel()

X_val_images = np.load(os.path.join(VAL_DIR, "images.npy"))
y_val = np.load(os.path.join(VAL_DIR, "labels.npy"))

print("train images ", X_train_images.shape, X_train_images.dtype)
print("train labels ", y_train.shape, np.unique(y_train))
print("val   images ", X_val_images.shape, X_val_images.dtype)
print("val   labels ", y_val.shape, np.unique(y_val))
'''


FEATURE_SRC = '''from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler


def _ft(func, **kw):
    """FunctionTransformer wrapper; the kwargs are pickled together with the model.

    Only module-level callables (numpy / scipy / scikit-learn) may appear here.
    Model_Evaluation.ipynb imports nothing but numpy, joblib and scikit-learn, so a
    callable defined in this notebook would not survive unpickling.
    """
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


def reshape(shape):
    return _ft(np.reshape, shape=shape)


# ------------------------------------------------------- oriented filter bank
def oriented_derivative_kernel(angle_deg, sigma, size=7):
    """Steerable derivative-of-Gaussian filter, shaped (1, size, size, 1)."""
    axis = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    gx = -(xx / sigma ** 2) * gauss
    gy = -(yy / sigma ** 2) * gauss
    theta = np.deg2rad(angle_deg)
    kernel = np.cos(theta) * gx + np.sin(theta) * gy
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-8
    return kernel.astype(np.float32)[None, :, :, None]


def laplacian_kernel(sigma, size=7):
    """Laplacian of Gaussian, shaped (1, size, size, 1)."""
    axis = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    kernel = (r2 - 2.0 * sigma ** 2) / sigma ** 4 * np.exp(-r2 / (2.0 * sigma ** 2))
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-8
    return kernel.astype(np.float32)[None, :, :, None]


def filter_pool_branch(kernel, cells):
    """|filter response|, averaged over channels, mean-pooled over a cells x cells grid."""
    size = 32 // cells
    return Pipeline([
        ("convolve", _ft(ndimage.convolve, weights=kernel)),
        ("abs", _ft(np.abs)),
        ("channel_mean", _ft(np.mean, axis=3)),
        ("reshape", reshape((-1, cells, size, cells, size))),
        ("pool", _ft(np.mean, axis=(2, 4))),
        ("flatten", reshape((-1, cells * cells))),
    ])


def colour_branch(cells):
    """Mean colour of every cell, so the model keeps the coarse colour layout."""
    size = 32 // cells
    return Pipeline([
        ("reshape", reshape((-1, cells, size, cells, size, 3))),
        ("pool", _ft(np.mean, axis=(2, 4))),
        ("flatten", reshape((-1, cells * cells * 3))),
    ])


# --------------------------------------------------- two-layer learned filters
def sample_patches(maps, patch, n_patch, rng, chunk=20000):
    """(n_img, H, W) -> (n_patch, patch*patch) mean-removed, unit-variance patches."""
    windows = np.lib.stride_tricks.sliding_window_view(maps, (patch, patch), axis=(1, 2))
    n_img, ny, nx = windows.shape[:3]
    rows = rng.integers(0, n_img, n_patch)
    ys = rng.integers(0, ny, n_patch)
    xs = rng.integers(0, nx, n_patch)
    patches = np.zeros((n_patch, patch * patch), np.float32)
    for i in range(0, n_patch, chunk):
        sl = slice(i, min(i + chunk, n_patch))
        block = windows[rows[sl], ys[sl], xs[sl]].reshape(-1, patch * patch)
        block = block - block.mean(1, keepdims=True)
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


def learn_two_layer_filters(images, n_l1, n_l2, patch, n_patch, n_images, seed):
    """Stage-1 filters from the images; stage-2 filters from each stage-1 map."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(images), min(n_images, len(images)), replace=False))
    gray = images[idx].mean(3).astype(np.float32)
    stage1 = pca_filters(sample_patches(gray, patch, n_patch, rng), n_l1)
    stage2 = []
    for k in range(n_l1):
        response = ndimage.convolve(gray, stage1[k], mode="reflect", axes=(1, 2))
        np.clip(response, 0.0, None, out=response)
        stage2.append(pca_filters(sample_patches(response, patch, n_patch, rng), n_l2))
    return stage1, stage2


def learned_pool(cells):
    size = 32 // cells
    return Pipeline([
        ("reshape", reshape((-1, cells, size, cells, size))),
        ("pool", _ft(np.mean, axis=(2, 4))),
        ("flatten", reshape((-1, cells * cells))),
    ])


def learned_response(kernel, cells):
    """Convolve a single (N, 32, 32) map, rectify, block-pool."""
    return Pipeline([
        ("convolve", _ft(ndimage.convolve, weights=kernel, mode="reflect", axes=(1, 2))),
        ("relu", _ft(np.clip, a_min=0.0, a_max=None)),
        ("pool", learned_pool(cells)),
    ])


def learned_branches(filters, cells):
    """One branch per stage-1 map; the stage-2 bank of a map lives inside its own
    branch, so the extractor never materialises an (N, 32, 32, n_l1) array."""
    stage1, stage2 = filters
    branches = []
    for i, kernel in enumerate(stage1):
        head = [(f"l2_{j}", learned_response(k, cells)) for j, k in enumerate(stage2[i])]
        head.append(("l1", learned_pool(cells)))
        branches.append((
            f"map{i}",
            Pipeline([
                ("convolve", _ft(ndimage.convolve, weights=kernel, mode="reflect", axes=(1, 2))),
                ("relu", _ft(np.clip, a_min=0.0, a_max=None)),
                ("head", FeatureUnion(head, n_jobs=1)),
            ]),
        ))
    return branches


# ------------------------------------------------------------ feature block
def build_feature_block(filters, cells=8, learned_cells=4, angles=8, sigmas=(1.2, 2.2),
                        laplacians=(1.6, 3.0)):
    """Raw uint8 (N, 32, 32, 3) -> pooled feature vector. Stateless: never fitted."""
    oriented = []
    for sigma in sigmas:
        for angle in np.linspace(0, 180, angles, endpoint=False):
            oriented.append((
                f"o{int(sigma * 10)}_{int(angle)}",
                filter_pool_branch(oriented_derivative_kernel(angle, sigma), cells),
            ))
    for sigma in laplacians:
        oriented.append((f"l{int(sigma * 10)}", filter_pool_branch(laplacian_kernel(sigma), cells)))

    return Pipeline([
        ("features", FeatureUnion([
            ("oriented", Pipeline([
                ("as_float32", _ft(np.asarray, dtype=np.float32)),
                ("filters", FeatureUnion(oriented, n_jobs=1)),
            ])),
            ("learned", Pipeline([
                ("as_float32", _ft(np.asarray, dtype=np.float32)),
                ("gray", _ft(np.mean, axis=3)),
                ("maps", FeatureUnion(learned_branches(filters, learned_cells), n_jobs=1)),
            ])),
            ("colour", Pipeline([
                ("as_float32", _ft(np.asarray, dtype=np.float32)),
                ("cells", colour_branch(cells)),
            ])),
        ], n_jobs=1)),
    ])
'''


RECIPE_SRC = '''RECIPE = {
    "n_clean": 50000,        # all original training images
    "n_shift": 30000,        # flipped / translated copies
    "n_shift2": 20000,       # second, smaller flipped / translated copy set
    "n_corr13": 30000,       # corrupted copies, severity 1-3
    "n_corr15": 30000,       # corrupted copies, severity 1-5
    "learned": dict(n_l1=8, n_l2=8, patch=5, n_patch=120000, n_images=8000),
    "pca": 768,
    "hidden": (1024,),
    "epochs": 55,
    "alpha": 1e-4,
    "n_members": 3,          # bagged MLPs behind the shared scaler/PCA
    "prep_fit_rows": 60000,  # the scaler/PCA are fitted on this many rows
    "chunk": 20000,          # feature-extraction chunk size, caps peak memory
    "seeds": [42, 2026, 3102],
}


def random_flip_shift(images, rng, max_shift=3):
    """Horizontal flip plus a small translation (edge-clamped)."""
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
    """Give every image a random corruption type and a random severity."""
    out = np.empty_like(images)
    choice = rng.integers(0, len(CORRUPTION_NAMES), len(images))
    sev = rng.integers(sev_lo, sev_hi + 1, len(images))
    for k in np.unique(choice):
        for s in np.unique(sev[choice == k]):
            m = (choice == k) & (sev == s)
            out[m] = corrupt_batch(images[m], CORRUPTION_NAMES[k], int(s), rng)
    return out


def transform_in_chunks(feats, images, chunk):
    """The feature block is stateless; running it in chunks keeps peak memory low."""
    parts = []
    for start in range(0, len(images), chunk):
        parts.append(feats.transform(images[start:start + chunk]).astype(np.float32))
    return np.concatenate(parts)


def build_model(seed, X_pool, y_pool, feats, recipe):
    """One submitted model: bagged soft-voting ensemble plus horizontal-flip TTA."""
    t0 = time.time()
    rng = np.random.default_rng(seed)

    idx_shift = rng.choice(len(X_pool), recipe["n_shift"], replace=False)
    idx_shift2 = rng.choice(len(X_pool), recipe["n_shift2"], replace=False)
    idx_corr13 = rng.choice(len(X_pool), recipe["n_corr13"], replace=False)
    idx_corr15 = rng.choice(len(X_pool), recipe["n_corr15"], replace=False)

    names = ("clean", "shifted", "shifted2", "corrupted13", "corrupted15")
    images = {
        "clean": X_pool,
        "shifted": random_flip_shift(X_pool[idx_shift], rng),
        "shifted2": random_flip_shift(X_pool[idx_shift2], rng),
        "corrupted13": make_corrupted(X_pool[idx_corr13], rng, 1, 3),
        "corrupted15": make_corrupted(X_pool[idx_corr15], rng, 1, 5),
    }
    labels = {
        "clean": y_pool,
        "shifted": y_pool[idx_shift],
        "shifted2": y_pool[idx_shift2],
        "corrupted13": y_pool[idx_corr13],
        "corrupted15": y_pool[idx_corr15],
    }

    # every member sees the same augmented pool, so the features are extracted once
    blocks = []
    for name in names:
        block = transform_in_chunks(feats, images[name], recipe["chunk"])
        blocks.append(block)
        print(f"    features {name:12s} {block.shape}  {time.time() - t0:.0f}s", flush=True)
        images[name] = None
    X_all = np.concatenate(blocks)
    y_all = np.concatenate([labels[n] for n in names])
    del blocks, images, labels

    # one shared projection for every member: it is stored once in the .joblib file
    rows = np.random.default_rng(seed).choice(
        len(X_all), min(recipe["prep_fit_rows"], len(X_all)), replace=False
    )
    prep = Pipeline([
        ("standardize", StandardScaler()),
        ("pca", PCA(n_components=recipe["pca"], random_state=seed)),
    ]).fit(X_all[rows])
    del rows
    X_proj = prep.transform(X_all)
    X_all = None
    print(
        f"    {X_proj.shape[0]} samples x {X_proj.shape[1]} principal components"
        f"  {time.time() - t0:.0f}s",
        flush=True,
    )

    members = []
    for i in range(recipe["n_members"]):
        tm = time.time()
        head = MLPClassifier(
            hidden_layer_sizes=recipe["hidden"],
            alpha=recipe["alpha"],
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=recipe["epochs"],
            random_state=seed + i,
        ).fit(X_proj, y_all)
        inner = Pipeline([("prep", prep), ("head", head)])
        members.append(Pipeline([("features", feats), ("inner", inner)]))
        print(f"    member {i}: {head.n_iter_} epochs, {time.time() - tm:.0f}s", flush=True)

    # the flipped twins share the *fitted* head object, so they add no model size
    flipped = [
        Pipeline([
            ("flip", Pipeline([("hflip", _ft(np.flip, axis=2)), ("features", feats)])),
            ("inner", m.named_steps["inner"]),
        ])
        for m in members
    ]

    ensemble = VotingClassifier(
        [(f"member{i}", m) for i, m in enumerate(members + flipped)], voting="soft"
    )
    ensemble.estimators_ = members + flipped
    ensemble.le_ = LabelEncoder().fit(y_pool)
    ensemble.classes_ = ensemble.le_.classes_
    return ensemble
'''


TRAIN_SRC = '''t0 = time.time()
filters = learn_two_layer_filters(X_train_images, seed=RECIPE["seeds"][0], **RECIPE["learned"])
print(f"learned {len(filters[0])} stage-1 filters and "
      f"{len(filters[1])} x {len(filters[1][0])} stage-2 filters  ({time.time() - t0:.0f}s)")

feats = build_feature_block(filters)
print("feature vector length:", feats.transform(X_train_images[:8]).shape[1])

summary = []
for seed in RECIPE["seeds"]:
    t0 = time.time()
    print(f"seed {seed}: building the training mixture")
    X_pool = X_train_images[: RECIPE["n_clean"]]
    y_pool = y_train[: RECIPE["n_clean"]]
    model = build_model(seed, X_pool, y_pool, feats, RECIPE)
    val_acc = accuracy_score(y_val, model.predict(X_val_images))
    path = MODEL_FILES[seed]
    joblib.dump(model, path)
    size_mb = os.path.getsize(path) / 1e6
    summary.append((seed, val_acc, size_mb, time.time() - t0))
    print(f"seed {seed}: validation accuracy {val_acc:.4f}  |  "
          f"{size_mb:.1f} MB  |  {time.time() - t0:.0f}s", flush=True)
'''


VERIFY_SRC = '''accuracies = []
for seed, path in MODEL_FILES.items():
    model = joblib.load(path)
    acc = accuracy_score(y_val, model.predict(X_val_images))
    accuracies.append(acc)
    print(f"{path}: validation accuracy {acc:.4f}  ({os.path.getsize(path) / 1e6:.1f} MB)")

print("")
print("Individual accuracies:")
for (seed, _, size_mb, secs), acc in zip(summary, accuracies):
    print(f"  model_seed{seed}.joblib: {acc:.4f}  ({size_mb:.1f} MB, {secs:.0f}s)")
print(f"Average validation accuracy: {np.mean(accuracies):.4f}")
print(f"All files below 100 MB: {all(os.path.getsize(p) / 1e6 < 100 for p in MODEL_FILES.values())}")
'''


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(INTRO),
        md("## 1. Environment check"),
        code(ENV_SRC),
        md("## 2. Data paths and loading\n\n"
           "`y_train.npy` is stored as a column vector `(50000, 1)`, so it is flattened to\n"
           "`(50000,)` before training."),
        code(DATA_SRC),
        md("## 3. Feature extraction\n\n"
           "Everything below is built from `scipy.ndimage.convolve` plus NumPy reshaping and\n"
           "pooling, so the saved model contains no custom class and can be unpickled with\n"
           "only NumPy and scikit-learn available."),
        code(FEATURE_SRC),
        md("## 4. Corruption and augmentation suite\n\n"
           "CIFAR-10-C style corruptions, NumPy/SciPy only, severity 1-5. Used to build the\n"
           "corrupted parts of the training set."),
        code(CORRUPTIONS_SRC),
        md("## 5. Training recipe\n\n"
           "The two-layer filter banks are learned from the training images, the feature block\n"
           "is stateless and computed once per augmented pool, a single `StandardScaler` +\n"
           "`PCA` is fitted on that pool and shared by every member, and each member is also\n"
           "added a second time behind a horizontal flip."),
        code(RECIPE_SRC),
        md("## 6. Train and save the three models"),
        code(TRAIN_SRC),
        md("## 7. Verification: reload the saved models"),
        code(VERIFY_SRC),
    ]
    nbf.write(nb, "train_models.ipynb")
    print("wrote train_models.ipynb with", len(nb.cells), "cells")


if __name__ == "__main__":
    main()

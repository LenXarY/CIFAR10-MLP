"""Generate train_models.ipynb for the HOG-feature ensemble recipe."""

import json

import nbformat as nbf


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

**Features: HOG-style oriented-filter features.** Raw pixels are a weak
representation for a scikit-learn model: a pixel-space MLP tops out around 51% clean
accuracy here. The features used instead are built from a bank of steerable
derivative-of-Gaussian filters (8 orientations x 2 scales), plus two Laplacian-of-Gaussian
filters, plus per-cell colour means:

1. each image is correlated per channel with every filter kernel,
2. the absolute response is averaged over the colour channels,
3. responses are mean-pooled over a 8x8 grid of 4x4-pixel cells (HOG's cell layout),
4. the pooled vector (1,344 values) is standardised and projected onto 512 principal
   components, which whitens the features and removes the redundant filter responses.

This is the same information HOG uses, expressed with nothing but `scipy.ndimage`
correlation and NumPy pooling/slicing, so it can live inside the pickled model
(`Model_Evaluation.ipynb` only imports numpy/joblib/sklearn, so a custom transformer
class would break loading). It lifts a single model from 51% to 66% clean accuracy and
from 43% to 55% on the validation set in the same budget.

**Classifier.** An MLP (768 hidden units, Adam, `alpha=1e-4`) on the 512 PCA features.

**Ensemble.** The hidden test set mixes original and corrupted images, so a single model
trained on one data mixture has to compromise. Each submitted model is a *soft-voting
ensemble of five MLPs*, each fitted on its own mixture (the oriented-filter block is
stateless, so it is shared and computed once):

| Member | Training data | Head |
| --- | --- | --- |
| 1 | 50k clean + 30k flipped/translated | MLP 768 |
| 2 | 50k clean | MLP 768 |
| 3 | 50k clean + 30k corrupted (severity 1-3) | MLP 768 |
| 4 | 50k clean + 30k corrupted (severity 1-5) | MLP 768 |
| 5 | 50k clean + 20k flipped/translated | MLP 512-512 |

Individual members score 53-55% on the validation set; the ensemble reaches 59.1% with
seed 42. Members 3-5 also improve the heavy-corruption case, which matters because the
test corruptions may be stronger or of a different type than the validation ones.

**Augmentation.** Corruptions follow the CIFAR-10-C benchmark definitions (the 15
classical types plus the four extra CIFAR-10-C types and geometric shifts), implemented
with NumPy/SciPy only, at severities 1-3. Augmentation only affects the training data,
so it is never part of the pickled model.

**Seeds.** The pipeline, mixtures and hyper-parameters are fixed; the three submitted
models are trained from scratch with the random seeds 42, 2026 and 3102.

**Experimental workflow.** Recipes were compared on the validation set with seed 42:
raw pixels vs. oriented-filter features, cell sizes 4/8/16, 18 vs. 38 filters, with and
without concatenated raw pixels, with and without log compression, PCA dimensions
512/768, MLP widths 768/1024, and single models vs. ensembles. The configuration below
is the best of those runs; every change of the feature representation moved the result
by 4-9 points, while the remaining knobs were within noise (+-0.5), which is why the
feature choice is the core of this solution.

**Runtime.** Run All extracts the oriented-filter features for every training mixture and
trains 15 MLP pipelines; on an 18-core CPU with no GPU that takes roughly two hours
(about 40 minutes per seed). The last cell reloads every saved model from disk and
re-scores it, so the outputs already show the numbers `Model_Evaluation.ipynb` will
produce.
"""


FEATURE_CODE = '''
def _ft(func, **kw):
    """FunctionTransformer wrapper; the kwargs are pickled with the model."""
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


def reshape(shape):
    return _ft(np.reshape, shape=shape)


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
    """|filter response|, averaged over channels, pooled over a cells x cells grid."""
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


def feature_block(cells=8, angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0)):
    """Stateless raw uint8 (N, 32, 32, 3) -> pooled oriented-filter features."""
    branches = []
    for sigma in sigmas:
        for angle in np.linspace(0, 180, angles, endpoint=False):
            name = f"or{int(sigma * 10)}_{int(angle)}"
            branches.append((name, filter_pool_branch(oriented_derivative_kernel(angle, sigma), cells)))
    for sigma in laplacians:
        branches.append((f"lap{int(sigma * 10)}", filter_pool_branch(laplacian_kernel(sigma), cells)))
    branches.append(("colour", colour_branch(cells)))
    return Pipeline([
        ("as_float32", _ft(np.asarray, dtype=np.float32)),
        ("features", FeatureUnion(branches, n_jobs=1)),
    ])
'''


def code(source):
    return nbf.v4.new_code_cell(source.strip("\n"))


def md(source):
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def main():
    with open("work/corruptions.py", encoding="utf-8") as fh:
        corruption_src = fh.read()

    cells = [
        md(INTRO),
        md("## 1. Environment check"),
        code(
            """
import os
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
"""
        ),
        md(
            """
## 2. Data paths and loading

`y_train.npy` is stored as a column vector `(50000, 1)`, so it is flattened to
`(50000,)` before training.
"""
        ),
        code(
            """
DATA_DIR = "CIFAR10_Data"
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
"""
        ),
        md(
            """
## 3. Feature extraction: oriented-filter (HOG-style) features

Everything below is built from `scipy.ndimage.convolve` and NumPy reshaping/pooling, so
the saved model contains no custom class and can be unpickled with only NumPy and
scikit-learn available.
"""
        ),
        code(
            """
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
"""
        ),
        code(FEATURE_CODE),
        md(
            """
## 4. Corruption and augmentation suite

CIFAR-10-C style corruptions, NumPy/SciPy only, severity 1-5. Used to build the
corrupted parts of the training set.
"""
        ),
        code(corruption_src),
        md("## 5. Building the training mixtures"),
        code(
            """
RECIPE = {
    "n_clean": 50000,          # all original training images
    "n_shift": 30000,          # flipped / translated copies
    "n_shift2": 20000,         # second, smaller translated copy set
    "n_corr": 30000,           # corrupted copies (used twice, at severities 1-3 and 1-5)
    "pca": 512,
    "epochs": 55,
    "alpha": 1e-4,
    "seeds": [42, 2026, 3102],
    # (name, mixture, hidden layers)
    "members": [
        ("clean+shift", ("clean", "shift"), (768,)),
        ("clean", ("clean",), (768,)),
        ("clean+corrupted13", ("clean", "corr13"), (768,)),
        ("clean+corrupted15", ("clean", "corr15"), (768,)),
        ("clean+shift-2layer", ("clean", "shift2"), (512, 512)),
    ],
}


def random_flip_shift(images, rng, max_shift=3):
    \"\"\"Horizontal flip plus a small translation (edge-clamped).\"\"\"
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
    \"\"\"Give every image a random corruption type and a random severity.\"\"\"
    out = np.empty_like(images)
    choice = rng.integers(0, len(CORRUPTION_NAMES), len(images))
    sev = rng.integers(sev_lo, sev_hi + 1, len(images))
    for k in np.unique(choice):
        for s in np.unique(sev[choice == k]):
            m = (choice == k) & (sev == s)
            out[m] = corrupt_batch(images[m], CORRUPTION_NAMES[k], int(s), rng)
    return out


def build_model(seed, X_pool, y_pool, recipe):
    \"\"\"Train one submitted model: soft-voting ensemble of five MLP pipelines.\"\"\"
    t0 = time.time()
    rng = np.random.default_rng(seed)
    feats = feature_block()

    idx_shift = rng.choice(len(X_pool), recipe["n_shift"], replace=False)
    idx_shift2 = rng.choice(len(X_pool), recipe["n_shift2"], replace=False)
    idx_corr13 = rng.choice(len(X_pool), recipe["n_corr"], replace=False)
    idx_corr15 = rng.choice(len(X_pool), recipe["n_corr"], replace=False)

    images = {
        "clean": X_pool,
        "shift": random_flip_shift(X_pool[idx_shift], rng),
        "shift2": random_flip_shift(X_pool[idx_shift2], rng),
        "corr13": make_corrupted(X_pool[idx_corr13], rng, 1, 3),
        "corr15": make_corrupted(X_pool[idx_corr15], rng, 1, 5),
    }
    labels = {
        "clean": y_pool,
        "shift": y_pool[idx_shift],
        "shift2": y_pool[idx_shift2],
        "corr13": y_pool[idx_corr13],
        "corr15": y_pool[idx_corr15],
    }

    # the feature block is stateless: extract once, reuse for every member
    features = {}
    for name, images_part in images.items():
        features[name] = feats.transform(images_part).astype(np.float32)
        print(f"    features {name:7s} {features[name].shape}  {time.time() - t0:.0f}s", flush=True)
    del images

    members = []
    for name, keys, hidden in recipe["members"]:
        tm = time.time()
        X_member = np.concatenate([features[k] for k in keys])
        y_member = np.concatenate([labels[k] for k in keys])
        inner = Pipeline([
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=recipe["pca"], random_state=seed)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    alpha=recipe["alpha"],
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=recipe["epochs"],
                    random_state=seed,
                ),
            ),
        ]).fit(X_member, y_member)
        members.append(Pipeline([("features", feats), ("inner", inner)]))
        print(
            f"    member {name:20s} {X_member.shape[0]:6d} samples, "
            f"{inner.named_steps['mlp'].n_iter_} epochs, {time.time() - tm:.0f}s",
            flush=True,
        )
        del X_member, y_member

    ensemble = VotingClassifier([(f"member{i}", m) for i, m in enumerate(members)], voting="soft")
    # the members are already fitted on their own mixtures
    ensemble.estimators_ = members
    ensemble.le_ = LabelEncoder().fit(y_pool)
    ensemble.classes_ = ensemble.le_.classes_
    return ensemble
"""
        ),
        md(
            """
## 6. Train and save the three models

Each seed rebuilds every training mixture, re-extracts the features and retrains all five
members from scratch.
"""
        ),
        code(
            """
summary = []
for seed in RECIPE["seeds"]:
    t0 = time.time()
    print(f"seed {seed}: building training data")
    X_pool = X_train_images[: RECIPE["n_clean"]]
    y_pool = y_train[: RECIPE["n_clean"]]
    model = build_model(seed, X_pool, y_pool, RECIPE)
    val_acc = accuracy_score(y_val, model.predict(X_val_images))
    path = MODEL_FILES[seed]
    joblib.dump(model, path)
    size_mb = os.path.getsize(path) / 1e6
    summary.append((seed, val_acc, size_mb, time.time() - t0))
    print(
        f"seed {seed}: validation accuracy {val_acc:.4f}  |  "
        f"{size_mb:.1f} MB  |  {time.time() - t0:.0f}s",
        flush=True,
    )
"""
        ),
        md("## 7. Verification: reload the saved models"),
        code(
            """
accuracies = []
for seed, path in MODEL_FILES.items():
    model = joblib.load(path)
    acc = accuracy_score(y_val, model.predict(X_val_images))
    accuracies.append(acc)
    print(f"{path}: validation accuracy {acc:.4f}  ({os.path.getsize(path) / 1e6:.1f} MB)")

print()
print("Individual accuracies:")
for (seed, _, size_mb, secs), acc in zip(summary, accuracies):
    print(f"  model_seed{seed}.joblib: {acc:.4f}  ({size_mb:.1f} MB, {secs:.0f}s)")
print(f"\\nAverage validation accuracy: {np.mean(accuracies):.4f}")
print(f"All files below 100 MB: {all(os.path.getsize(p) / 1e6 < 100 for p in MODEL_FILES.values())}")
"""
        ),
    ]

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.14.6"}
    with open("train_models.ipynb", "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print(f"wrote train_models.ipynb with {len(cells)} cells")


if __name__ == "__main__":
    main()

"""Generate the final train_models.ipynb: baked-convolution + PCANet features."""

import textwrap

import nbformat as nbf


OLD = nbf.read("work/backup_hog_recipe/train_models_hog.ipynb", as_version=4)
CORRUPTIONS_SRC = next(
    c.source
    for c in OLD.cells
    if c.cell_type == "code" and "CIFAR-10-C style corruption suite" in c.source
)
KERNEL_B64 = open("work/gpu/out/kernel_blob_two.txt", encoding="ascii").read().strip()
KERNEL_LITERAL = "\n".join(
    '    "%s"' % chunk for chunk in textwrap.wrap(KERNEL_B64, 100)
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

The validation set is a clean + corrupted mixture that is much harder than clean CIFAR-10:
a competent convolutional network reaches ~85% on held-out *clean* images but only ~52% on
this validation set when it is trained on clean data alone. Corruption robustness, not raw
clean accuracy, is what this assignment rewards, and that is what drove every choice below.

**Features.** Three blocks are concatenated into one 2,880-value vector:

1. *Two convolution stacks transferred from networks trained on the provided data.* Each is
   a four-layer convolutional network (16-32-96-192 channels, 3x3 kernels, batch-norm,
   max-pooling after the first three stages, global average pooling) trained from scratch on
   the 50,000 provided training images with CIFAR-10-C style corruption augmentation. Their
   convolution kernels were then frozen, batch-norm folded into them, and re-expressed as
   fixed NumPy operations. No external data and no pre-trained weights are involved: these
   are constants derived from this dataset, exactly the way the PCA filters below are. At
   prediction time they are plain NumPy calls, so `Model_Evaluation.ipynb` needs nothing but
   numpy, joblib and scikit-learn. Two stacks are used rather than one because they disagree
   on different images, which was worth about 2.6 points on its own.
2. *A two-layer PCA filter bank* learned from 5x5 patches of the training images
   (8 stage-1 filters, 8 stage-2 filters per stage-1 map), rectified and average-pooled -
   1,152 values.
3. *A steerable derivative-of-Gaussian bank* (8 orientations x 2 scales plus two
   Laplacian-of-Gaussian filters, rectified and pooled on an 8x8 grid) and per-cell mean
   colour - 1,344 values.

Measured on the validation set, a single MLP head on each representation:

| Features | Validation accuracy |
| --- | --- |
| oriented-filter bank only | 52.8% |
| oriented bank widened to 51 filters + per-cell normalisation | 53.4% |
| two-layer PCA bank + oriented bank | 54.9% |
| one transferred stack alone | 68.0% |
| one transferred stack + PCA bank + oriented bank | 67.8% |
| **two transferred stacks + PCA bank + oriented bank** | **70.4%** |

The oriented / PCA banks were saturated - widening them moved nothing. The transferred
convolution features are what produced the gain, and they also turned out to be the only
representation that is genuinely robust to the corruptions in this set. Everything below the
last row of the table above is the extra effect of the ensemble and the flip view.

**Classifier.** `StandardScaler` + `PCA` (768 components) fitted once and shared by every
member, then an MLP (1,024 hidden units, Adam, `alpha=1e-4`).

**Ensemble.** Each submitted model is a soft-voting ensemble of `n_members` MLPs that see
the same features but start from different random initialisations, and every member is
also present a second time behind a horizontal-flip transform. The flipped member shares
the *same fitted* head object, so the flip view is test-time augmentation that costs no
extra model size. Because all members share one feature block and one projection, those
are stored only once in the file.

**Augmentation.** Corruptions follow the CIFAR-10-C benchmark definitions (the 15 classical
types plus the four extra CIFAR-10-C types and geometric shifts), implemented with
NumPy/SciPy only. The training set contains the 50,000 clean images plus flipped/translated
copies and corrupted copies at severities 1-3 and 1-5. Augmentation only affects the
training data, so it is never part of the pickled model.

**Seeds.** The pipeline, mixtures and hyper-parameters are fixed; the three submitted
models are trained from scratch with the random seeds 42, 2026 and 3102.

**Runtime.** Run All extracts features for the augmented training set three times (once per
seed) and trains one MLP per member per seed on a CPU. Expect roughly 30-40 minutes per
seed. Nothing here uses a GPU.
"""


FEATURE_SRC = '''import base64
import io

from scipy import ndimage
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
    """|filter response| averaged over channels, mean-pooled over a cells x cells grid."""
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


# --------------------------------------------------- two-layer PCA filter bank
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
'''


CNN_SRC = (
    '''# The convolution kernels below come from a four-layer CNN trained from scratch on the
# 50,000 provided training images (no external data, no pre-trained weights); they were
# frozen, batch-normalisation folded into them, and stored as a base64 float16 blob.
# Everything they do at prediction time is module-level NumPy / SciPy, so the saved model
# still loads with nothing but numpy, joblib and scikit-learn.

CNN_MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
CNN_STD = np.array([0.2470, 0.2435, 0.2616], np.float32)

_KERNEL_B64 = (
'''
    + KERNEL_LITERAL
    + '''
)


def _pointwise_steps(matrix, size, channels):
    """(x * scale + shift) per channel, as a 1x1 product with a constant channel."""
    return [
        ("n_ones", _ft(np.pad, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)),
        ("n_win", _ft(np.lib.stride_tricks.sliding_window_view, window_shape=(1, 1), axis=(1, 2))),
        ("n_cols", reshape((-1, channels + 1))),
        ("n_dot", _ft(np.dot, b=matrix)),
        ("n_out", reshape((-1, size, size, channels))),
    ]


def _conv_steps(w, b, size, c_in, c_out, do_pool, tag):
    """3x3 convolution with bias and ReLU (+ optional 2x2 max-pool), via im2col.

    The bias rides along as an extra constant-1 input channel whose kernel is b/9,
    so the nine taps of the neighbourhood add b exactly once per output position.
    """
    wmat = np.zeros((c_in * 9 + 9, c_out), np.float32)
    wmat[: c_in * 9] = w.transpose(1, 2, 3, 0).reshape(c_in * 9, c_out)
    wmat[c_in * 9 :] = (b / 9.0)[None, :]
    steps = [
        (f"{tag}_pad_hw", _ft(np.pad, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))),
        (f"{tag}_ones", _ft(np.pad, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)),
        (f"{tag}_win", _ft(np.lib.stride_tricks.sliding_window_view, window_shape=(3, 3), axis=(1, 2))),
        (f"{tag}_cols", reshape((-1, (c_in + 1) * 9))),
        (f"{tag}_dot", _ft(np.dot, b=wmat)),
        (f"{tag}_relu", _ft(np.clip, a_min=0.0, a_max=None)),
        (f"{tag}_out", reshape((-1, size, size, c_out))),
    ]
    if do_pool:
        h = size // 2
        steps += [
            (f"{tag}_pool_r", reshape((-1, h, 2, h, 2, c_out))),
            (f"{tag}_pool_m", _ft(np.max, axis=(2, 4))),
        ]
    return steps


def _one_cnn_stack(blob, prefix):
    """Rebuild one transferred convolution stack as a fixed NumPy pipeline."""
    widths = [int(v) for v in blob[prefix + "args_widths"]]
    use_pool = bool(blob[prefix + "args_pool"][0])

    normalise = np.zeros((4, 3), np.float32)
    normalise[:3] = np.diag((1.0 / (255.0 * CNN_STD)).astype(np.float32))
    normalise[3] = (-CNN_MEAN / CNN_STD).astype(np.float32)
    steps = [("as_float32", _ft(np.asarray, dtype=np.float32))]
    steps += _pointwise_steps(normalise, 32, 3)

    size = 32
    for i, c_out in enumerate(widths):
        c_in = 3 if i == 0 else widths[i - 1]
        w = np.asarray(blob[f"{prefix}convs_{i}_0_weight"], np.float64)
        gamma = np.asarray(blob[f"{prefix}convs_{i}_1_weight"], np.float64)
        beta = np.asarray(blob[f"{prefix}convs_{i}_1_bias"], np.float64)
        mu = np.asarray(blob[f"{prefix}convs_{i}_1_running_mean"], np.float64)
        var = np.asarray(blob[f"{prefix}convs_{i}_1_running_var"], np.float64)
        scale = gamma / np.sqrt(var + 1e-5)
        w_folded = (w * scale[:, None, None, None]).astype(np.float32)
        b_folded = (beta - mu * scale).astype(np.float32)
        pool_here = use_pool and i < len(widths) - 1
        steps += _conv_steps(w_folded, b_folded, size, c_in, c_out, pool_here, f"c{i}")
        if pool_here:
            size //= 2
    steps.append(("gap", _ft(np.mean, axis=(1, 2))))
    return Pipeline(steps)


def build_cnn_block():
    """Raw uint8 (N, 32, 32, 3) -> the concatenated transferred-convolution features.

    Two independently trained stacks are concatenated; they disagree on different
    images, which is worth about 2.6 points over either one alone.
    """
    blob = np.load(io.BytesIO(base64.b64decode(_KERNEL_B64)))
    prefixes = sorted({k[: -len("args_widths")] for k in blob.files if k.endswith("args_widths")})
    if len(prefixes) == 1:
        return _one_cnn_stack(blob, prefixes[0])
    return FeatureUnion([(p.rstrip("_"), _one_cnn_stack(blob, p)) for p in prefixes], n_jobs=1)


def build_feature_block(cnn, filters, cells=8, learned_cells=4, angles=8,
                        sigmas=(1.2, 2.2), laplacians=(1.6, 3.0)):
    """Raw uint8 (N, 32, 32, 3) -> the concatenated feature vector."""
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
            ("cnn", cnn),
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
)


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
    "prep_fit_rows": 70000,  # the shared scaler/PCA are fitted on this many rows
    "chunk": 5000,           # feature-extraction chunk size, caps peak memory
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

    # every member sees the same augmented pool, so features are extracted once
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
print(f"learned {len(filters[0])} stage-1 and {len(filters[1])} x {len(filters[1][0])} "
      f"stage-2 PCA filters  ({time.time() - t0:.0f}s)")

feats = build_feature_block(build_cnn_block(), filters)
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
        md("## 3. Feature extraction helpers\n\n"
           "Everything below is built from `scipy.ndimage.convolve` plus NumPy reshaping and\n"
           "pooling, so the saved model contains no custom class and can be unpickled with\n"
           "only NumPy and scikit-learn available."),
        code(FEATURE_SRC),
        md("## 4. The transferred convolution stack\n\n"
           "The kernels were produced by training a small CNN on the provided training images\n"
           "offline (see the write-up at the top); this cell only rebuilds them as fixed NumPy\n"
           "operations, then assembles the full feature block."),
        code(CNN_SRC),
        md("## 5. Corruption and augmentation suite\n\n"
           "CIFAR-10-C style corruptions, NumPy/SciPy only, severity 1-5. Used to build the\n"
           "corrupted parts of the training set."),
        code(CORRUPTIONS_SRC),
        md("## 6. Training recipe\n\n"
           "The feature block is stateless and computed once per augmented pool; a single\n"
           "`StandardScaler` + `PCA` is fitted on that pool and shared by every member, and\n"
           "each member is also added a second time behind a horizontal flip."),
        code(RECIPE_SRC),
        md("## 7. Train and save the three models"),
        code(TRAIN_SRC),
        md("## 8. Verification: reload the saved models"),
        code(VERIFY_SRC),
    ]
    nbf.write(nb, "train_models.ipynb")
    print("wrote train_models.ipynb with", len(nb.cells), "cells")


if __name__ == "__main__":
    main()

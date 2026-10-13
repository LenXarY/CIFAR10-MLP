"""Train one submission model: soft-voting ensemble of HOG-feature MLPs.

The oriented-filter feature block is stateless, so it is computed once per
source image set and shared by all members; only the scaler / PCA / MLP of each
member are fitted separately.
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

sys.path.insert(0, "work")
from corruptions import CORRUPTION_NAMES, corrupt_batch  # noqa: E402
from feat_lib import hog_branches  # noqa: E402


DATA = "CIFAR10_Data"
VAL = "validation_set_to_students"


def _ft(func, **kw):
    return FunctionTransformer(func, kw_args=kw or None, validate=False)


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


def feature_block(cells=8, angles=8, sigmas=(1.2, 2.2), laplacians=(1.6, 3.0)):
    """Stateless raw-uint8 -> pooled oriented-filter features."""
    return Pipeline(
        [
            ("as_float32", _ft(np.asarray, dtype=np.float32)),
            (
                "features",
                FeatureUnion(
                    hog_branches(cells=cells, angles=angles, sigmas=sigmas, laplacians=laplacians),
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_model(seed, X_pool, y_pool, n_shift=50000, n_corr=50000, n_shift2=25000,
                pca_dim=512, epochs=80, alpha=1e-4, verbose=True):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    feats = feature_block()

    idx = rng.choice(len(X_pool), min(n_shift, len(X_pool)), replace=False)
    X_shift = random_flip_shift(X_pool[idx], rng)
    idx2 = rng.choice(len(X_pool), min(n_shift2, len(X_pool)), replace=False)
    X_shift2 = random_flip_shift(X_pool[idx2], rng)
    idx3 = rng.choice(len(X_pool), min(n_corr, len(X_pool)), replace=False)
    X_corr13 = make_corrupted(X_pool[idx3], rng, 1, 3)
    idx4 = rng.choice(len(X_pool), min(n_corr, len(X_pool)), replace=False)
    X_corr15 = make_corrupted(X_pool[idx4], rng, 1, 5)
    if verbose:
        print(f"  augmentation ready in {time.time() - t0:.0f}s", flush=True)

    H = {}
    for name, images in [
        ("clean", X_pool),
        ("shift", X_shift),
        ("shift2", X_shift2),
        ("corr13", X_corr13),
        ("corr15", X_corr15),
    ]:
        H[name] = feats.transform(images).astype(np.float32)
        if verbose:
            print(f"  features {name:7s} {H[name].shape} in {time.time() - t0:.0f}s", flush=True)
    del X_shift, X_shift2, X_corr13, X_corr15

    specs = [
        ("clean+shift", ["clean", "shift"], [y_pool, y_pool[idx]], (768,)),
        ("clean", ["clean"], [y_pool], (768,)),
        ("clean+corr13", ["clean", "corr13"], [y_pool, y_pool[idx3]], (768,)),
        ("clean+corr15", ["clean", "corr15"], [y_pool, y_pool[idx4]], (768,)),
        ("clean+shift-2layer", ["clean", "shift2"], [y_pool, y_pool[idx2]], (512, 512)),
    ]

    members = []
    for name, keys, labels, hidden in specs:
        tm = time.time()
        X_member = np.concatenate([H[k] for k in keys])
        y_member = np.concatenate(labels)
        inner = Pipeline(
            [
                ("standardize", StandardScaler()),
                ("pca", PCA(n_components=pca_dim, random_state=seed)),
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
                ),
            ]
        ).fit(X_member, y_member)
        members.append(Pipeline([("features", feats), ("inner", inner)]))
        if verbose:
            print(
                f"  member {name:20s} {X_member.shape[0]:6d} samples, "
                f"{inner.named_steps['mlp'].n_iter_} epochs, {time.time() - tm:.0f}s",
                flush=True,
            )
        del X_member, y_member

    ensemble = VotingClassifier([(f"m{i}", m) for i, m in enumerate(members)], voting="soft")
    ensemble.estimators_ = members
    ensemble.le_ = LabelEncoder().fit(y_pool)
    ensemble.classes_ = ensemble.le_.classes_
    if verbose:
        print(f"  model built in {time.time() - t0:.0f}s", flush=True)
    return ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-shift", type=int, default=50000)
    ap.add_argument("--n-corr", type=int, default=50000)
    ap.add_argument("--n-shift2", type=int, default=25000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    X = np.load(f"{DATA}/x_train.npy")
    y = np.load(f"{DATA}/y_train.npy").ravel()
    model = build_model(
        args.seed, X, y,
        n_shift=args.n_shift, n_corr=args.n_corr, n_shift2=args.n_shift2,
        pca_dim=args.pca, epochs=args.epochs,
    )

    X_val = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"seed {args.seed}: validation accuracy {acc:.4f}", flush=True)

    out = args.out or f"work/models/hog_seed{args.seed}.joblib"
    joblib.dump(model, out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"seed {args.seed}: saved {out} ({size_mb:.1f} MB)")
    with open("work/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tag": f"hog_seed{args.seed}", "config": vars(args), "val": acc, "size_mb": size_mb}) + "\n")


if __name__ == "__main__":
    main()

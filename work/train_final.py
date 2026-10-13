"""Train one submission-shaped model: a soft-voting ensemble of three MLPs
that share a single preprocessing block (standardise + PCA on clean images).

The three members differ in their training distribution, which is what makes
the ensemble better than any single member.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

sys.path.insert(0, "work")
from corruptions import CORRUPTION_NAMES, corrupt_batch  # noqa: E402


DATA = "CIFAR10_Data"
VAL = "validation_set_to_students"
N_FEATURES = 32 * 32 * 3


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


def build_preprocessing(pca_dim, seed):
    return Pipeline(
        [
            ("reshape", FunctionTransformer(np.reshape, kw_args={"shape": (-1, N_FEATURES)}, validate=False)),
            ("as_float32", FunctionTransformer(np.asarray, kw_args={"dtype": np.float32})),
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=pca_dim, random_state=seed)),
        ]
    )


def build_model(
    seed, hidden, epochs, alpha, n_clean, n_shift, n_corr, pca_dim, sev_lo, sev_hi,
    per_member_pre=False, verbose=True,
):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    X_all = np.load(f"{DATA}/x_train.npy")
    y_all = np.load(f"{DATA}/y_train.npy").ravel()
    X_pool, y_pool = X_all[:n_clean], y_all[:n_clean]

    shift_idx = rng.choice(n_clean, n_shift, replace=False)
    X_shift, y_shift = random_flip_shift(X_pool[shift_idx], rng), y_pool[shift_idx]
    corr_idx = rng.choice(n_clean, n_corr, replace=False)
    X_corr = make_corrupted(X_pool[corr_idx], rng, sev_lo, sev_hi)
    y_corr = y_pool[corr_idx]
    if verbose:
        print(f"  augmentation ready in {time.time() - t0:.0f}s", flush=True)

    pre = build_preprocessing(pca_dim, seed)
    pre.fit(X_pool)
    if verbose:
        print(f"  preprocessing fitted in {time.time() - t0:.0f}s", flush=True)

    recipes = {
        "clean+shift": (np.concatenate([X_pool, X_shift]), np.concatenate([y_pool, y_shift])),
        "clean": (X_pool, y_pool),
        "clean+corrupted": (np.concatenate([X_pool, X_corr]), np.concatenate([y_pool, y_corr])),
    }
    members = []
    for name, (Xm, ym) in recipes.items():
        tm = time.time()
        head = MLPClassifier(
            hidden_layer_sizes=(hidden,),
            alpha=alpha,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=epochs,
            random_state=seed,
        )
        if per_member_pre:
            pipe = Pipeline([("pre", build_preprocessing(pca_dim, seed)), ("mlp", head)])
            pipe.fit(Xm, ym)
            epochs_run = pipe.named_steps["mlp"].n_iter_
        else:
            Z = pre.transform(Xm).astype(np.float32)
            head.fit(Z, ym)
            epochs_run = head.n_iter_
            pipe = Pipeline([("pre", pre), ("mlp", head)])
            del Z
        members.append(pipe)
        if verbose:
            print(f"  member {name:16s} {Xm.shape[0]:6d} samples, {epochs_run} epochs, {time.time() - tm:.0f}s", flush=True)

    voter = VotingClassifier([(f"m{i}", m) for i, m in enumerate(members)], voting="soft")
    voter.estimators_ = members
    voter.le_ = LabelEncoder().fit(y_pool)
    voter.classes_ = voter.le_.classes_
    if verbose:
        print(f"  model built in {time.time() - t0:.0f}s", flush=True)
    return voter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-clean", type=int, default=50000)
    ap.add_argument("--n-shift", type=int, default=50000)
    ap.add_argument("--n-corr", type=int, default=50000)
    ap.add_argument("--sev-lo", type=int, default=1)
    ap.add_argument("--sev-hi", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--per-member-pre", action="store_true")
    args = ap.parse_args()

    model = build_model(
        args.seed, args.hidden, args.epochs, args.alpha,
        args.n_clean, args.n_shift, args.n_corr, args.pca, args.sev_lo, args.sev_hi,
        per_member_pre=args.per_member_pre,
    )

    X_val = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"seed {args.seed}: validation accuracy {acc:.4f}", flush=True)

    out = args.out or f"work/models/final_seed{args.seed}.joblib"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(model, out)
    size_mb = os.path.getsize(out) / 1e6
    print(f"seed {args.seed}: saved {out} ({size_mb:.1f} MB)")
    with open("work/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tag": f"final_seed{args.seed}", "config": vars(args), "val": acc, "size_mb": size_mb}) + "\n")


if __name__ == "__main__":
    main()

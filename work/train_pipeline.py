"""Train the submission-shaped sklearn Pipeline and evaluate it.

The pipeline takes raw (N, 32, 32, 3) uint8 images straight from the .npy
files, so whatever this script saves is exactly what Model_Evaluation.ipynb
would load.
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, "work")
from corruptions import CORRUPTION_NAMES, corrupt_batch  # noqa: E402


DATA = "CIFAR10_Data"
VAL = "validation_set_to_students"
N_FEATURES = 32 * 32 * 3
MODEL_DIR = "work/models"


def build_pipeline(hidden, pca_dims, seed, max_iter, alpha=1e-4):
    return Pipeline(
        [
            (
                "reshape",
                FunctionTransformer(
                    np.reshape, kw_args={"shape": (-1, N_FEATURES)}, validate=False
                ),
            ),
            # uint8 -> float32 (so the scaler and PCA keep working in float32)
            ("as_float32", FunctionTransformer(np.asarray, kw_args={"dtype": np.float32})),
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=pca_dims, random_state=seed)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    alpha=alpha,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def random_flip_shift(images, rng, max_shift=3, cutout=8):
    x = images.copy()
    flip = rng.random(len(x)) < 0.5
    if flip.any():
        x[flip] = x[flip][:, :, ::-1]
    dy = rng.integers(-max_shift, max_shift + 1, len(x))
    dx = rng.integers(-max_shift, max_shift + 1, len(x))
    rows = np.arange(len(x))[:, None, None]
    ys = np.clip(np.arange(32)[None, :, None] + dy[:, None, None], 0, 31)
    xs = np.clip(np.arange(32)[None, None, :] + dx[:, None, None], 0, 31)
    x = x[rows, ys, xs]
    if cutout:
        cy = rng.integers(0, 32, len(x))
        cx = rng.integers(0, 32, len(x))
        half = cutout // 2
        yy = np.arange(32)[None, :, None]
        xx = np.arange(32)[None, None, :]
        box = ((np.abs(yy - cy[:, None, None]) <= half) & (np.abs(xx - cx[:, None, None]) <= half))[..., None]
        x = np.where(box, 0, x)
    return x


def make_corrupted(images, rng, sev_lo=1, sev_hi=5, kinds=None):
    kinds = CORRUPTION_NAMES if kinds is None else kinds
    out = np.empty_like(images)
    choice = rng.integers(0, len(kinds), len(images))
    sev = rng.integers(sev_lo, sev_hi + 1, len(images))
    for k in np.unique(choice):
        for s in np.unique(sev[choice == k]):
            m = (choice == k) & (sev == s)
            out[m] = corrupt_batch(images[m], kinds[k], int(s), rng)
    return out


def evaluate(model, X_uint8, y):
    return float(accuracy_score(y, model.predict(X_uint8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-clean", type=int, default=45000)
    ap.add_argument("--n-corr", type=int, default=0)
    ap.add_argument("--n-shift", type=int, default=0)
    ap.add_argument("--cutout", type=int, default=0)
    ap.add_argument("--sev-lo", type=int, default=1)
    ap.add_argument("--sev-hi", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    X_all = np.load(f"{DATA}/x_train.npy")
    y_all = np.load(f"{DATA}/y_train.npy").ravel()
    X_val = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")

    order = np.random.default_rng(0).permutation(len(X_all))
    hold = order[: args.n_eval]
    pool = order[args.n_eval : args.n_eval + args.n_clean]
    X_base, y_base = X_all[pool], y_all[pool]
    X_hold, y_hold = X_all[hold], y_all[hold]

    parts_x, parts_y = [X_base], [y_base]
    if args.n_shift:
        idx = rng.choice(len(X_base), args.n_shift)
        parts_x.append(random_flip_shift(X_base[idx], rng, cutout=args.cutout))
        parts_y.append(y_base[idx])
    if args.n_corr:
        idx = rng.choice(len(X_base), args.n_corr)
        parts_x.append(make_corrupted(X_base[idx], rng, args.sev_lo, args.sev_hi))
        parts_y.append(y_base[idx])
    X_tr = np.concatenate(parts_x)
    y_tr = np.concatenate(parts_y)
    print(f"[{args.tag}] train {X_tr.shape} built in {time.time() - t0:.0f}s", flush=True)

    hidden = tuple([args.hidden] * args.layers)
    model = build_pipeline(hidden, args.pca, args.seed, args.max_iter, args.alpha)
    t1 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t1
    n_features = model.named_steps["pca"].n_components_
    print(f"[{args.tag}] fit {fit_s:.0f}s, {model.named_steps['mlp'].n_iter_} epochs", flush=True)

    res = {
        "tag": args.tag,
        "config": vars(args),
        "fit_s": round(fit_s),
        "epochs": int(model.named_steps["mlp"].n_iter_),
        "pca_dim": int(n_features),
    }
    res["clean_holdout"] = evaluate(model, X_hold, y_hold)
    res["val_provided"] = evaluate(model, X_val, y_val)
    for sev in (1, 3, 5):
        synth = make_corrupted(X_hold[:2000], np.random.default_rng(1000 + sev), sev, sev)
        res[f"synth_sev{sev}"] = evaluate(model, synth, y_hold[:2000])

    print(
        f"[{args.tag}] clean={res['clean_holdout']:.4f} val={res['val_provided']:.4f} "
        f"synth1={res['synth_sev1']:.4f} synth3={res['synth_sev3']:.4f} synth5={res['synth_sev5']:.4f}",
        flush=True,
    )
    path = os.path.join(MODEL_DIR, f"{args.tag}.joblib")
    joblib.dump(model, path)
    res["model_path"] = path
    res["model_mb"] = round(os.path.getsize(path) / 1e6, 2)
    print(f"[{args.tag}] saved {path} ({res['model_mb']} MB), total {time.time() - t0:.0f}s")
    with open("work/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()

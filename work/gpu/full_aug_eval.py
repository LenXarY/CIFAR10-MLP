"""Full-scale test of the augmented recipe with the baked CNN features.

Builds the augmented training mixture, extracts the baked convolution features
and the PCANet / oriented-filter features, then trains one scikit-learn head on
the concatenation and scores it against the validation set. Results are cached
so the classifier can be re-run without recomputing features.
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bake  # noqa: E402
from lab_lib import FEATURE_SPECS, build_features, build_mixture, load_raw  # noqa: E402


MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32)
SIZES = {"clean": 50000, "shift": 30000, "shift2": 0, "corr13": 30000, "corr15": 30000}


def cnn_steps(npz_path):
    z = np.load(npz_path)
    state = {k: z[k] for k in z.files if k.startswith("convs_")}
    widths = tuple(int(v) for v in z["args_widths"])
    pool = bool(z["args_pool"][0])
    res = tuple(32 // (2**i) for i in range(len(widths)))
    return bake.bake(state, widths, pool=pool, resolutions=res, norm=(MEAN, STD))


def chunked(fn, blocks, chunk):
    out = [fn(b[i : i + chunk]).astype(np.float32) for b in blocks for i in range(0, len(b), chunk)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="work/gpu/out/baked_weights.npz")
    ap.add_argument("--spec", default="pc_or")
    ap.add_argument("--which", default="both", choices=["cnn", "pcanet", "both"])
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--hidden", default="1024")
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--parts", default="clean,shift,corr13,corr15")
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--cache", default="work/gpu/out/cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-tag", default="")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    tag = args.cache_tag or f"{args.which}_{args.spec}_seed{args.seed}"
    tr_path, va_path, y_path = (f"{args.cache}/{tag}_train.npy",
                                f"{args.cache}/{tag}_val.npy",
                                f"{args.cache}/{tag}_y.npy")

    X, y, X_val, y_val = load_raw()
    if os.path.exists(tr_path):
        X_all, X_va, y_all = np.load(tr_path), np.load(va_path), np.load(y_path)
        print(f"cached features {X_all.shape} / {X_va.shape}", flush=True)
    else:
        parts = build_mixture(X, y, sizes=SIZES)
        keys = args.parts.split(",")
        blocks = [parts[k][0] for k in keys]
        y_all = np.concatenate([parts[k][1] for k in keys])
        print(f"training mixture {sum(len(b) for b in blocks)} images", flush=True)

        feats_tr, feats_va = [], []
        if args.which in ("cnn", "both"):
            t0 = time.time()
            steps = cnn_steps(args.npz)
            fn = lambda b: bake.apply_steps(steps, b)  # noqa: E731
            feats_tr.append(chunked(fn, blocks, args.chunk))
            feats_va.append(chunked(fn, [X_val], args.chunk))
            print(f"cnn features {feats_tr[-1].shape} in {time.time() - t0:.0f}s", flush=True)
        if args.which in ("pcanet", "both"):
            t0 = time.time()
            pc = build_features(FEATURE_SPECS[args.spec])
            fn = lambda b: pc.transform(b)  # noqa: E731
            feats_tr.append(chunked(fn, blocks, args.chunk))
            feats_va.append(chunked(fn, [X_val], args.chunk))
            print(f"pcanet features {feats_tr[-1].shape} in {time.time() - t0:.0f}s", flush=True)
        del blocks, parts
        X_all = np.concatenate(feats_tr, axis=1) if len(feats_tr) > 1 else feats_tr[0]
        X_va = np.concatenate(feats_va, axis=1) if len(feats_va) > 1 else feats_va[0]
        del feats_tr, feats_va
        np.save(tr_path, X_all)
        np.save(va_path, X_va)
        np.save(y_path, y_all)
        print(f"combined train {X_all.shape} val {X_va.shape}", flush=True)

    hidden = tuple(int(v) for v in args.hidden.split(","))
    clf = Pipeline([
        ("standardize", StandardScaler()),
        ("pca", PCA(n_components=args.pca, random_state=args.seed)),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256,
                              learning_rate_init=1e-3, max_iter=args.epochs,
                              random_state=args.seed)),
    ])
    t0 = time.time()
    clf.fit(X_all, y_all)
    acc = accuracy_score(y_val, clf.predict(X_va))
    print(f"[{args.which}] pca {args.pca} hidden {hidden} -> val {acc:.4f} "
          f"({clf.named_steps['mlp'].n_iter_} epochs, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

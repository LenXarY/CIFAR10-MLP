"""Score the final head design on cached features: shared projection, bagged
MLPs, horizontal-flip TTA, and the CNN-only vs CNN+PCANet comparison."""

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
from full_aug_eval import MEAN, STD, cnn_steps, chunked  # noqa: E402
from lab_lib import FEATURE_SPECS, build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="work/gpu/out/cache")
    ap.add_argument("--tag", default="both_pc_or_seed0")
    ap.add_argument("--spec", default="pc_or")
    ap.add_argument("--npz", default="work/gpu/out/baked_weights.npz")
    ap.add_argument("--which", default="both")
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--hidden", default="1024")
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    X_tr = np.load(f"{args.cache}/{args.tag}_train.npy")
    X_va = np.load(f"{args.cache}/{args.tag}_val.npy")
    y = np.load(f"{args.cache}/{args.tag}_y.npy")
    y_val = np.load("validation_set_to_students/labels.npy")
    print(f"train {X_tr.shape} val {X_va.shape}", flush=True)

    if args.tta:
        X_val = np.load("validation_set_to_students/images.npy")
        flipped = np.ascontiguousarray(X_val[:, :, ::-1])
        blocks_tr, blocks_va = [], []
        if args.which in ("cnn", "both"):
            steps = cnn_steps(args.npz)
            blocks_va.append(chunked(lambda b: bake.apply_steps(steps, b), [flipped], 5000))
        if args.which in ("pcanet", "both"):
            pc = build_features(FEATURE_SPECS[args.spec])
            blocks_va.append(chunked(lambda b: pc.transform(b), [flipped], 5000))
        X_va_flip = np.concatenate(blocks_va, axis=1) if len(blocks_va) > 1 else blocks_va[0]
        np.save(f"{args.cache}/{args.tag}_val_flip.npy", X_va_flip)
        print(f"flipped val features {X_va_flip.shape}", flush=True)

    hidden = tuple(int(v) for v in args.hidden.split(","))
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(X_tr), min(70000, len(X_tr)), replace=False)
    t0 = time.time()
    prep = Pipeline([
        ("standardize", StandardScaler()),
        ("pca", PCA(n_components=args.pca, random_state=args.seed)),
    ]).fit(X_tr[rows])
    Z_tr = prep.transform(X_tr)
    Z_va = prep.transform(X_va)
    print(f"shared projection fitted in {time.time() - t0:.0f}s -> {Z_tr.shape}", flush=True)

    proba_va, proba_flip = [], []
    for i in range(args.members):
        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256,
                            learning_rate_init=1e-3, max_iter=args.epochs,
                            random_state=args.seed + i).fit(Z_tr, y)
        p = mlp.predict_proba(Z_va)
        proba_va.append(p)
        print(f"  member {i}: val {accuracy_score(y_val, p.argmax(1)):.4f} "
              f"({mlp.n_iter_} epochs, {time.time() - t0:.0f}s)", flush=True)
        if args.tta:
            proba_flip.append(mlp.predict_proba(prep.transform(X_va_flip)))

    ens = np.mean(proba_va, axis=0)
    print(f"ensemble of {args.members}: {accuracy_score(y_val, ens.argmax(1)):.4f}")
    if args.tta:
        tta = np.mean(proba_va + proba_flip, axis=0)
        print(f"ensemble + hflip TTA: {accuracy_score(y_val, tta.argmax(1)):.4f}")


if __name__ == "__main__":
    main()

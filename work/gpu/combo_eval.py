"""Does concatenating the baked CNN features with the PCANet features help?"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lab_lib import FEATURE_SPECS, build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="work/gpu/out/baked")
    ap.add_argument("--spec", default="pc_or")
    ap.add_argument("--hidden", default="512")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    A_tr = np.load(f"{args.prefix}_train.npy")
    A_va = np.load(f"{args.prefix}_val.npy")
    y = np.load(f"{args.prefix}_y_train.npy")
    yv = np.load(f"{args.prefix}_y_val.npy")

    X = np.load("CIFAR10_Data/x_train.npy")[: len(y)]
    Xv = np.load("validation_set_to_students/images.npy")
    t0 = time.time()
    feats = build_features(FEATURE_SPECS[args.spec])
    B_tr = feats.transform(X).astype(np.float32)
    B_va = feats.transform(Xv).astype(np.float32)
    print(f"pcanet features {B_tr.shape} in {time.time() - t0:.0f}s")

    hidden = tuple(int(v) for v in args.hidden.split(","))
    combos = {
        "cnn only": (A_tr, A_va),
        "pcanet only": (B_tr, B_va),
        "cnn + pcanet": (np.hstack([A_tr, B_tr]), np.hstack([A_va, B_va])),
        "cnn + pcanet(std)": None,
    }
    for name, pair in combos.items():
        if name == "cnn + pcanet(std)":
            A2 = StandardScaler().fit_transform(A_tr)
            A2v = StandardScaler().fit_transform(A_va)
            pair = (np.hstack([A2, B_tr]), np.hstack([A2v, B_va]))
        Xtr, Xva = pair
        clf = Pipeline([
            ("standardize", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256,
                                  learning_rate_init=1e-3, max_iter=args.epochs,
                                  random_state=args.seed)),
        ]).fit(Xtr, y)
        acc = accuracy_score(yv, clf.predict(Xva))
        print(f"  {name:20s} dim {Xtr.shape[1]:6d} -> val {acc:.4f}")


if __name__ == "__main__":
    main()

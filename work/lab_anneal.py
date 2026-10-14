"""Learning-rate annealing with warm_start, on cached features."""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_lib import build_mixture, load_raw  # noqa: E402


STAGES = [(1e-3, 60), (5e-4, 40), (2e-4, 30), (1e-4, 20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="feature cache tag")
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--hidden", default="1024,512")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--modes", default="anneal,plain")
    args = ap.parse_args()

    X, y, X_val, y_val = load_raw()
    parts = build_mixture(X, y, sizes={
        "clean": 16000, "shift": 10000, "shift2": 7000, "corr13": 10000,
        "corr15": 10000, "corr_layered": 15000, "corr35": 0, "corr_layered15": 0,
        "rot": 0, "cutout": 0, "jitter": 0, "blur": 0,
    })
    import lab_lib

    feats, F = lab_lib.features_for(
        lab_lib.FEATURE_SPECS["pc_or"], parts, tag=args.tag
    )
    keys = ("clean", "shift", "shift2", "corr13", "corr15", "corr_layered")
    X_tr = np.concatenate([F[k] for k in keys if k in F])
    y_tr = np.concatenate([parts[k][1] for k in keys if k in F])
    F_va = feats.transform(X_val).astype(np.float32)
    F_flip = feats.transform(np.ascontiguousarray(X_val[:, :, ::-1])).astype(np.float32)
    print(f"train {X_tr.shape}", flush=True)

    rows = np.random.default_rng(0).choice(len(X_tr), min(50000, len(X_tr)), replace=False)
    prep = Pipeline([
        ("s", StandardScaler()),
        ("pca", PCA(n_components=args.pca, random_state=0)),
    ]).fit(X_tr[rows])
    Ztr, Zva, Zflip = prep.transform(X_tr), prep.transform(F_va), prep.transform(F_flip)

    hidden = tuple(int(v) for v in args.hidden.split(","))
    for mode in args.modes.split(","):
        t0 = time.time()
        if mode == "anneal":
            clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=args.alpha,
                                batch_size=args.batch, warm_start=True,
                                max_iter=1, random_state=0)
            for lr, ep in STAGES:
                clf.learning_rate_init = lr
                clf.max_iter = ep
                clf.fit(Ztr, y_tr)
            epochs = sum(e for _, e in STAGES)
        else:
            clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=args.alpha,
                                batch_size=args.batch, learning_rate_init=1e-3,
                                max_iter=150, random_state=0)
            clf.fit(Ztr, y_tr)
            epochs = clf.n_iter_
        p0 = clf.predict_proba(Zva)
        p1 = clf.predict_proba(Zflip)
        acc = accuracy_score(y_val, (p0 + p1).argmax(1))
        plain = accuracy_score(y_val, clf.predict(Zva))
        print(f"  {mode:7s} plain {plain:.4f}  TTA {acc:.4f}  "
              f"({epochs} epochs, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

"""Cheap check: does concatenating a second CNN's features beat one CNN?"""

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
from full_aug_eval import cnn_steps, chunked  # noqa: E402
from lab_lib import build_mixture, load_raw  # noqa: E402


def main():
    X, y, X_val, y_val = load_raw()
    sizes = {"clean": 50000, "shift": 30000, "shift2": 0, "corr13": 30000, "corr15": 30000}
    steps2 = cnn_steps("work/gpu/out/baked_s2_weights.npz")
    parts = build_mixture(X, y, sizes=sizes)
    blocks = [parts[k][0] for k in ("clean", "shift", "corr13", "corr15")]
    del parts

    t0 = time.time()
    cnn2_tr = chunked(lambda b: bake.apply_steps(steps2, b), blocks, 5000)
    cnn2_va = chunked(lambda b: bake.apply_steps(steps2, b), [X_val], 5000)
    print(f"second cnn features {cnn2_tr.shape} in {time.time() - t0:.0f}s", flush=True)
    del blocks

    base_tr = np.load("work/gpu/out/cache/both_big_train.npy")
    base_va = np.load("work/gpu/out/cache/both_big_val.npy")
    y_all = np.load("work/gpu/out/cache/both_big_y.npy")

    for name, Xtr, Xva in [
        ("one cnn + pcanet", base_tr, base_va),
        ("two cnn + pcanet", np.hstack([base_tr, cnn2_tr]), np.hstack([base_va, cnn2_va])),
    ]:
        rows = np.random.default_rng(0).choice(len(Xtr), 70000, replace=False)
        prep = Pipeline([
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=768, random_state=0)),
        ]).fit(Xtr[rows])
        Ztr, Zva = prep.transform(Xtr), prep.transform(Xva)
        mlp = MLPClassifier(hidden_layer_sizes=(1024,), alpha=1e-4, batch_size=256,
                            learning_rate_init=1e-3, max_iter=55, random_state=0).fit(Ztr, y_all)
        acc = accuracy_score(y_val, mlp.predict(Zva))
        print(f"  {name:20s} dim {Xtr.shape[1]:6d} -> single-member val {acc:.4f} "
              f"({mlp.n_iter_} epochs)", flush=True)


if __name__ == "__main__":
    main()

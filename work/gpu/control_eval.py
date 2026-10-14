"""Controls: PCANet bank alone, and the k-means bank with a wider projection."""

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


def fit_head(Xtr, y, Xva, yv, pca, hidden, epochs, seed=0, tag=""):
    rows = np.random.default_rng(seed).choice(len(Xtr), min(70000, len(Xtr)), replace=False)
    prep = Pipeline([
        ("standardize", StandardScaler()),
        ("pca", PCA(n_components=pca, random_state=seed)),
    ]).fit(Xtr[rows])
    Ztr, Zva = prep.transform(Xtr), prep.transform(Xva)
    mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256,
                        learning_rate_init=1e-3, max_iter=epochs,
                        random_state=seed).fit(Ztr, y)
    acc = accuracy_score(yv, mlp.predict(Zva))
    print(f"  {tag:28s} dim {Xtr.shape[1]:5d} pca {pca:5d} -> val {acc:.4f} "
          f"({mlp.n_iter_} epochs)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=55)
    args = ap.parse_args()
    hidden = (1024,)
    yv = np.load("validation_set_to_students/labels.npy")

    # control 1: the current PCANet + oriented bank, taken from the cached columns
    base_tr = np.load("work/gpu/out/cache/both_big_train.npy")[:, 192:]
    base_va = np.load("work/gpu/out/cache/both_big_val.npy")[:, 192:]
    y = np.load("work/gpu/out/cache/both_big_y.npy")
    fit_head(base_tr, y, base_va, yv, args.pca, hidden, args.epochs, tag="pcanet+oriented only")
    del base_tr, base_va

    # control 2: the k-means bank with a wider projection
    km_tr = np.load("work/gpu/out/cache/km24_train.npy")
    km_va = np.load("work/gpu/out/cache/km24_val.npy")
    km_y = np.load("work/gpu/out/cache/km24_y.npy")
    fit_head(km_tr, km_y, km_va, yv, 1536, hidden, args.epochs, tag="kmeans bank pca1536")


if __name__ == "__main__":
    main()

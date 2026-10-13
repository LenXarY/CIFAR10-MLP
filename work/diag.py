"""Diagnose where the validation errors come from.

The validation set is a clean + corrupted mixture. This script measures
per-image corruption indicators and reports the reference model's accuracy in
each stratum, so the next round of feature / augmentation work can target the
part of the set that actually costs accuracy.
"""

import os

import joblib
import numpy as np
from scipy import ndimage
from sklearn.metrics import accuracy_score


VAL = "validation_set_to_students"


def indicators(X):
    """Per-image corruption indicators, all cheap and corruption-agnostic."""
    xf = X.astype(np.float32)
    gray = xf.mean(3)

    # fully-saturated / clipped pixels: the signature of noise and contrast corruptions
    black = (X.max(3) == 0).mean((1, 2))
    white = (X.min(3) == 255).mean((1, 2))

    # JPEG blockiness: |diff| across the 8-pixel block borders vs everywhere
    d = np.abs(np.diff(gray, axis=2))
    border = d[:, :, 7::8][:, :, :3].mean((1, 2))
    overall = d.mean((1, 2)) + 1e-6
    blocky = border / overall

    # contrast and high-frequency energy
    std = gray.reshape(len(gray), -1).std(1)
    lap = np.abs(
        ndimage.convolve(gray, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)[None, :, :])
    ).mean((1, 2))

    sat = (xf.max(3) - xf.min(3)).mean((1, 2))
    return {
        "black": black,
        "white": white,
        "blocky": blocky,
        "std": std,
        "lap": lap,
        "sat": sat,
    }


def main():
    X_val = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")
    X_tr = np.load("CIFAR10_Data/x_train.npy", mmap_mode="r")

    model = joblib.load("reference/hog_trim_seed42.joblib")
    pred = model.predict(X_val)
    correct = pred == y_val
    print(f"overall validation accuracy: {correct.mean():.4f}")

    # how do the val images differ from clean training images?
    ind_val = indicators(X_val)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_tr), 20000, replace=False)
    ind_tr = indicators(np.asarray(X_tr[np.sort(idx)]))

    print("\nindicator            train(clean) med/p95     val med/p95")
    for k in ("black", "white", "blocky", "std", "lap", "sat"):
        t, v = ind_tr[k], ind_val[k]
        print(
            f"  {k:8s}  {np.median(t):10.4f} {np.percentile(t, 95):10.4f}"
            f"   {np.median(v):10.4f} {np.percentile(v, 95):10.4f}"
        )

    # stratify the validation set by a simple "does this look corrupted" flag
    thr_black = np.percentile(ind_tr["black"], 99.5)
    thr_blocky = np.percentile(ind_tr["blocky"], 99.5)
    thr_std = np.percentile(ind_tr["std"], 1.0)
    flag = (
        (ind_val["black"] > thr_black)
        | (ind_val["blocky"] > thr_blocky)
        | (ind_val["std"] < thr_std)
    )
    print(f"\nthresholds: black>{thr_black:.4f}  blocky>{thr_blocky:.3f}  std<{thr_std:.1f}")
    print(f"flagged as corrupted: {flag.mean():.1%} of the validation set")
    print(f"  accuracy on clean-like   ({1 - flag.mean():5.1%}): {correct[~flag].mean():.4f}")
    print(f"  accuracy on corrupted    ({flag.mean():5.1%}): {correct[flag].mean():.4f}")

    # accuracy by blockiness decile (compression strength proxy)
    print("\naccuracy by blockiness decile:")
    q = np.quantile(ind_val["blocky"], np.linspace(0, 1, 11))
    for i in range(10):
        m = (ind_val["blocky"] >= q[i]) & (ind_val["blocky"] <= q[i + 1])
        if m.sum():
            print(f"  d{i + 1} (blocky {q[i]:5.2f}-{q[i + 1]:5.2f}, n={m.sum():4d}): {correct[m].mean():.4f}")

    # accuracy by black-pixel fraction bucket (clipping severity)
    print("\naccuracy by clipped-pixel fraction:")
    edges = [0, 0.001, 0.005, 0.02, 0.05, 0.15, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ind_val["black"] >= lo) & (ind_val["black"] < hi)
        if m.sum():
            print(f"  [{lo:.3f}, {hi:.3f})  n={m.sum():4d}  acc={correct[m].mean():.4f}")


if __name__ == "__main__":
    main()

"""Estimate how much of the validation set is corrupted, and where the model loses.

A detector is trained with our own corruption suite as supervision: clean training
images are the negatives, the same images with random corruptions are the positives.
Applying it to the validation set gives a per-image corruption probability, which is
then used to stratify the model's accuracy.
"""

import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "work")
from lab_lib import FEATURE_SPECS, build_features, load_raw, make_corrupted  # noqa: E402


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model_seed42.joblib"
    X, y, X_val, y_val = load_raw()
    rng = np.random.default_rng(0)

    n = 6000
    idx = rng.choice(len(X), n, replace=False)
    clean = X[idx]
    corrupted = make_corrupted(clean.copy(), rng, 1, 5)

    feats = build_features(FEATURE_SPECS["pc_or"])
    F_clean = feats.transform(clean).astype(np.float32)
    F_corr = feats.transform(corrupted).astype(np.float32)
    F_val = feats.transform(X_val).astype(np.float32)
    print("detector features ready", F_clean.shape, F_val.shape, flush=True)

    Xd = np.vstack([F_corr, F_clean])
    yd = np.r_[np.ones(len(F_corr)), np.zeros(len(F_clean))]
    det = Pipeline([
        ("s", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=1.0)),
    ]).fit(Xd, yd)
    # sanity: how separable are they at all?
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(det, Xd, yd, cv=3, scoring="roc_auc").mean()
    print(f"detector separability (3-fold AUC, clean vs corrupted): {cv:.3f}")

    p_val = det.predict_proba(F_val)[:, 1]
    print(f"validation set estimated corrupted fraction: {(p_val > 0.5).mean():.1%}")

    m = joblib.load(model_path)
    pred = m.predict(X_val)
    correct = pred == y_val
    print(f"model {model_path}: overall {correct.mean():.4f}")
    q = np.quantile(p_val, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(5):
        m_i = (p_val >= q[i]) & (p_val <= q[i + 1])
        if m_i.sum():
            print(f"  corruption-prob quintile {i + 1} ({q[i]:.2f}-{q[i + 1]:.2f}, "
                  f"n={m_i.sum():4d}): acc {correct[m_i].mean():.4f}")
    flag = p_val > 0.5
    print(f"  clean-like  ({1 - flag.mean():5.1%}): {correct[~flag].mean():.4f}")
    print(f"  corrupted   ({flag.mean():5.1%}): {correct[flag].mean():.4f}")
    # accuracy by corruption severity band
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)):
        m_i = (p_val >= lo) & (p_val < hi)
        if m_i.sum():
            print(f"    p in [{lo:.1f},{hi:.1f}): n={m_i.sum():4d} acc {correct[m_i].mean():.4f}")


if __name__ == "__main__":
    main()

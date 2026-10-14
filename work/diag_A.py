"""Where does the pure-sklearn model lose accuracy: clean or corrupted images?"""

import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


def strata(X, Xtrain):
    """Split the validation set into clean-like and corrupted-like halves."""
    xf = X.astype(np.float32)
    gray = xf.mean(3)
    black = (X.max(3) == 0).mean((1, 2))
    d = np.abs(np.diff(gray, axis=2))
    blocky = d[:, :, 7::8][:, :, :3].mean((1, 2)) / (d.mean((1, 2)) + 1e-6)
    std = gray.reshape(len(gray), -1).std(1)
    sat = (xf.max(3) - xf.min(3)).mean((1, 2))

    t = Xtrain.astype(np.float32)
    g = t.mean(3)
    dt = np.abs(np.diff(g, axis=2))
    tb = np.percentile(dt[:, :, 7::8][:, :, :3].mean((1, 2)) / (dt.mean((1, 2)) + 1e-6), 99.5)
    ts = np.percentile(g.reshape(len(g), -1).std(1), 1.0)
    # a corruption is flagged if any indicator is far outside the clean distribution
    return (black > 0.05) | (blocky > tb) | (std < ts)


def main():
    X = np.load("validation_set_to_students/images.npy")
    y = np.load("validation_set_to_students/labels.npy")
    Xtr = np.load("CIFAR10_Data/x_train.npy")[:20000]
    flag = strata(X, Xtr)

    for path in sys.argv[1:]:
        m = joblib.load(path)
        p = m.predict(X)
        acc = accuracy_score(y, p)
        print(f"{path}: overall {acc:.4f}")
        print(f"    clean-like ({1 - flag.mean():5.1%}): {accuracy_score(y[~flag], p[~flag]):.4f}"
              f"   corrupted ({flag.mean():5.1%}): {accuracy_score(y[flag], p[flag]):.4f}")


if __name__ == "__main__":
    main()

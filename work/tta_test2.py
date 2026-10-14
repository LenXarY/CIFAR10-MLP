"""Test flip + shift test-time augmentation on the current model."""

import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


def shift(img, dy, dx, pad=4):
    p = np.pad(img, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    return p[:, pad + dy : pad + dy + 32, pad + dx : pad + dx + 32]


def main():
    path = sys.argv[1]
    X = np.load("validation_set_to_students/images.npy")
    y = np.load("validation_set_to_students/labels.npy")
    model = joblib.load(path)

    views = {"id": X, "flip": np.ascontiguousarray(X[:, :, ::-1])}
    for d in (1, 2, 3, 4):
        views[f"x{d}"] = shift(X, 0, -d)
        views[f"x{d}n"] = shift(X, 0, d)
        views[f"y{d}"] = shift(X, -d, 0)
        views[f"y{d}n"] = shift(X, d, 0)

    probs = {}
    for name, v in views.items():
        probs[name] = model.predict_proba(v)
        print(f"  {name:5s} alone {accuracy_score(y, probs[name].argmax(1)):.4f}", flush=True)

    def combo(names):
        p = sum(probs[n] for n in names)
        return accuracy_score(y, p.argmax(1))

    print()
    print(f"  flip only                 {combo(['id', 'flip']):.4f}")
    print(f"  flip + 4 shifts x4        "
          f"{combo(['id', 'flip'] + [f'{a}{d}{s}' for a in 'xy' for d in (1, 2, 3, 4) for s in ('', 'n')]):.4f}")
    print(f"  flip + shifts(1,2)        "
          f"{combo(['id', 'flip'] + [f'{a}{d}{s}' for a in 'xy' for d in (1, 2) for s in ('', 'n')]):.4f}")
    print(f"  flip + shifts(1,2,3)      "
          f"{combo(['id', 'flip'] + [f'{a}{d}{s}' for a in 'xy' for d in (1, 2, 3) for s in ('', 'n')]):.4f}")


if __name__ == "__main__":
    main()

"""Explore test-time-augmentation views cheaply.

The five ensemble members share one fitted feature block, so the oriented-filter
features can be extracted once per view and reused by every member.
"""

import numpy as np
import joblib
from sklearn.metrics import accuracy_score


def shift(img, dy, dx):
    rows = np.arange(32)[:, None]
    cols = np.arange(32)[None, :]
    ys = np.clip(rows + dy, 0, 31)
    xs = np.clip(cols + dx, 0, 31)
    return img[:, ys, xs]


X = np.load("validation_set_to_students/images.npy")
y = np.load("validation_set_to_students/labels.npy")
X, y = X[:1500], y[:1500]

model = joblib.load("reference/hog_trim_seed42.joblib")
feats = model.estimators_[0].named_steps["features"]
inners = [m.named_steps["inner"] for m in model.estimators_]

VIEWS = {
    "id": lambda a: a,
    "hflip": lambda a: a[:, :, ::-1],
    "vflip": lambda a: a[:, ::-1],
    "hvflip": lambda a: a[:, ::-1, ::-1],
    "yshift2": lambda a: shift(a, -2, 0),
    "yshift2n": lambda a: shift(a, 2, 0),
    "xshift2": lambda a: shift(a, 0, -2),
    "xshift2n": lambda a: shift(a, 0, 2),
    "xshift1": lambda a: shift(a, 0, -1),
    "xshift1n": lambda a: shift(a, 0, 1),
}

prob = {}
for name, fn in VIEWS.items():
    F = feats.transform(fn(X))
    prob[name] = np.mean([inner.predict_proba(F) for inner in inners], axis=0)
    print(f"view {name:10s} single-view accuracy {accuracy_score(y, np.argmax(prob[name], 1)):.4f}", flush=True)


def combo(names):
    p = sum(prob[n] for n in names)
    return accuracy_score(y, np.argmax(p, 1))


print()
for names in [
    ["id"],
    ["id", "hflip"],
    ["id", "hflip", "vflip", "hvflip"],
    ["id", "hflip", "yshift2", "yshift2n", "xshift2", "xshift2n"],
    ["id", "hflip", "xshift1", "xshift1n", "yshift2", "yshift2n"],
    ["id", "hflip", "vflip", "hvflip", "yshift2", "yshift2n", "xshift2", "xshift2n"],
]:
    print(f"{combo(names):.4f}  <- {'+'.join(names)}")

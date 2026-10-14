"""Compare TTA view sets by accuracy and by number of feature extractions."""

import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


def view(img, dy, dx, flip, pad=4):
    x = np.ascontiguousarray(img[:, :, ::-1]) if flip else img
    p = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    return p[:, pad + dy : pad + dy + 32, pad + dx : pad + dx + 32]


def main():
    path, members = sys.argv[1], int(sys.argv[2])
    X = np.load("validation_set_to_students/images.npy")
    y = np.load("validation_set_to_students/labels.npy")
    model = joblib.load(path)

    specs = [("id", 0, 0, False), ("flip", 0, 0, True)]
    for d in (1, 2, 3, 4):
        for s in (-1, 1):
            for f in (False, True):
                specs.append((f"{'f' if f else 'n'}x{d}{s}", 0, s * d, f))
                specs.append((f"{'f' if f else 'n'}y{d}{s}", s * d, 0, f))

    probs = {name: model.predict_proba(view(X, dy, dx, f)) for name, dy, dx, f in specs}

    def acc(names):
        return accuracy_score(y, sum(probs[n] for n in names).argmax(1))

    def sel(dmax, flips=True, plain=True):
        out = ["id"]
        if flips:
            out.append("flip")
        for d in range(1, dmax + 1):
            for s in (-1, 1):
                for f in ((False, True) if flips else (False,)):
                    if plain or f:
                        out += [f"{'f' if f else 'n'}x{d}{s}", f"{'f' if f else 'n'}y{d}{s}"]
        return sorted(set(out))

    for label, names in [
        ("flip only", ["id", "flip"]),
        ("shifts(1)", sel(1)),
        ("shifts(1,2)", sel(2)),
        ("shifts(1,2,3)", sel(3)),
        ("shifts(1..4)", sel(4)),
        ("shifts(1..4) no flip", sel(4, flips=False)),
    ]:
        n_views = len(names)
        print(f"  {label:22s} {n_views:3d} views  x{members} members = "
              f"{n_views * members:3d} passes   acc {acc(names):.4f}", flush=True)


if __name__ == "__main__":
    main()

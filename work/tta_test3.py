"""Flip + shift + flipped-shift TTA: pick the view set with the best accuracy."""

import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


VIEWS = [("id", 0, 0, False)]
for d in (1, 2, 3, 4):
    for sgn in (-1, 1):
        VIEWS.append((f"x{d}{'n' if sgn > 0 else ''}", 0, sgn * d, False))
        VIEWS.append((f"y{d}{'n' if sgn > 0 else ''}", sgn * d, 0, False))
        VIEWS.append((f"fx{d}{'n' if sgn > 0 else ''}", 0, sgn * d, True))
        VIEWS.append((f"fy{d}{'n' if sgn > 0 else ''}", sgn * d, 0, True))


def apply_view(img, dy, dx, flip, pad=4):
    x = np.ascontiguousarray(img[:, :, ::-1]) if flip else img
    p = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    return p[:, pad + dy : pad + dy + 32, pad + dx : pad + dx + 32]


def main():
    path = sys.argv[1]
    X = np.load("validation_set_to_students/images.npy")
    y = np.load("validation_set_to_students/labels.npy")
    model = joblib.load(path)
    probs = {}
    for name, dy, dx, flip in VIEWS:
        probs[name] = model.predict_proba(apply_view(X, dy, dx, flip))
    print(f"views computed: {len(probs)}", flush=True)

    def combo(names):
        return accuracy_score(y, sum(probs[n] for n in names).argmax(1))

    base = ["id"]
    flip = ["id"] + [n for n, _, _, f in VIEWS if f and n.startswith("fx") or n.startswith("fy")]
    flip_only = ["id"] + [n for n, dy, dx, f in VIEWS if f is False and (dy or dx)]
    allf = [n for n, _, _, f in VIEWS if f]
    alln = [n for n, _, _, f in VIEWS if not f]

    print(f"  id+flip                      {accuracy_score(y, (probs['id']+probs['fx1']).argmax(1)):.4f}")
    print(f"  id + all non-flip shifted    {combo(base + flip_only):.4f}  ({1+len(flip_only)} views)")
    print(f"  id + all flipped shifted     {combo(base + allf):.4f}")
    print(f"  everything                   {combo(alln + allf):.4f}  ({len(alln)+len(allf)} views)")


if __name__ == "__main__":
    main()

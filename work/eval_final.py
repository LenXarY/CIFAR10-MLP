"""Evaluate the three submitted models and their average, on the provided
validation set plus synthetic corruptions at several severities."""

import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, "work")
from corruptions import CORRUPTION_NAMES, corrupt_batch  # noqa: E402


def make_corrupted(images, sev, rng):
    out = np.empty_like(images)
    choice = rng.integers(0, len(CORRUPTION_NAMES), len(images))
    for k in np.unique(choice):
        m = choice == k
        out[m] = corrupt_batch(images[m], CORRUPTION_NAMES[k], sev, rng)
    return out


def main():
    X = np.load("CIFAR10_Data/x_train.npy")
    y = np.load("CIFAR10_Data/y_train.npy").ravel()
    hold = np.random.default_rng(0).permutation(len(X))[:3000]
    Xh, yh = X[hold], y[hold]

    X_val = np.load("validation_set_to_students/images.npy")
    y_val = np.load("validation_set_to_students/labels.npy")

    rng = np.random.default_rng(11)
    synth = {sev: make_corrupted(Xh, sev, rng) for sev in (1, 3, 5)}

    models = {s: joblib.load(f"model_seed{s}.joblib") for s in (42, 2026, 3102)}

    header = f"{'model':12s} {'clean':>7s} {'val':>7s} " + " ".join(f"{'sev' + str(s):>7s}" for s in (1, 3, 5))
    print(header)
    for seed, model in models.items():
        row = [accuracy_score(yh, model.predict(Xh)), accuracy_score(y_val, model.predict(X_val))]
        row += [accuracy_score(yh, model.predict(synth[s])) for s in (1, 3, 5)]
        print(f"seed {seed:<7d} " + " ".join(f"{v:7.4f}" for v in row))

    row = [
        accuracy_score(yh, np.mean([m.predict_proba(Xh) for m in models.values()], 0).argmax(1)),
        accuracy_score(y_val, np.mean([m.predict_proba(X_val) for m in models.values()], 0).argmax(1)),
    ]
    row += [
        accuracy_score(
            yh, np.mean([m.predict_proba(synth[s]) for m in models.values()], 0).argmax(1)
        )
        for s in (1, 3, 5)
    ]
    print(f"{'avg of 3':12s} " + " ".join(f"{v:7.4f}" for v in row))


if __name__ == "__main__":
    main()

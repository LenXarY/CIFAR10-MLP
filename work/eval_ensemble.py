"""Evaluate saved pipelines individually and as soft-voting ensembles."""

import itertools
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, "work")
from corruptions import corrupt_batch, CORRUPTION_NAMES  # noqa: E402


DATA = "CIFAR10_Data"
VAL = "validation_set_to_students"


def make_corrupted(images, rng, sev):
    out = np.empty_like(images)
    choice = rng.integers(0, len(CORRUPTION_NAMES), len(images))
    for k in np.unique(choice):
        m = choice == k
        out[m] = corrupt_batch(images[m], CORRUPTION_NAMES[k], sev, rng)
    return out


def main(tags):
    X_val = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")
    X_all = np.load(f"{DATA}/x_train.npy")
    y_all = np.load(f"{DATA}/y_train.npy").ravel()
    order = np.random.default_rng(0).permutation(len(X_all))
    hold = order[:5000]
    X_hold, y_hold = X_all[hold], y_all[hold]

    rng = np.random.default_rng(7)
    synth = {sev: make_corrupted(X_hold[:2000], rng, sev) for sev in (1, 3, 5)}

    models = {}
    for tag in tags:
        models[tag] = joblib.load(f"work/models/{tag}.joblib")

    def report(name, proba_list, y):
        p = np.mean(proba_list, axis=0)
        return accuracy_score(y, p.argmax(1))

    print(f"{'model':28s} {'clean':>7s} {'val':>7s} {'synth1':>7s} {'synth3':>7s} {'synth5':>7s}")
    for tag, m in models.items():
        row = [
            accuracy_score(y_hold[:2000], m.predict(X_hold[:2000])),
            accuracy_score(y_val, m.predict(X_val)),
        ]
        for sev in (1, 3, 5):
            row.append(accuracy_score(y_hold[:2000], m.predict(synth[sev])))
        print(f"{tag:28s} " + " ".join(f"{v:7.4f}" for v in row))

    print()
    for r in range(2, len(tags) + 1):
        for combo in itertools.combinations(tags, r):
            row = [
                report("", [models[t].predict_proba(X_hold[:2000]) for t in combo], y_hold[:2000]),
                report("", [models[t].predict_proba(X_val) for t in combo], y_val),
            ]
            for sev in (1, 3, 5):
                row.append(
                    report("", [models[t].predict_proba(synth[sev]) for t in combo], y_hold[:2000])
                )
            print(f"{'+'.join(combo):28s} " + " ".join(f"{v:7.4f}" for v in row))


if __name__ == "__main__":
    main(sys.argv[1:])

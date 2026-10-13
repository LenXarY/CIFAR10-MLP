"""Train a scikit-learn head on the dumped frozen-CNN features."""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--hidden", default="1024,512")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    X = np.load(f"{args.prefix}_train.npy")
    y = np.load(f"{args.prefix}_y_train.npy")
    Xv = np.load(f"{args.prefix}_val.npy")
    yv = np.load(f"{args.prefix}_y_val.npy")
    hidden = tuple(int(v) for v in args.hidden.split(","))
    print(f"train {X.shape} val {Xv.shape} hidden {hidden}")

    test_preds = []
    for s in range(args.seeds):
        clf = Pipeline([
            ("standardize", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256,
                                  learning_rate_init=1e-3, max_iter=args.epochs, random_state=s)),
        ])
        clf.fit(X, y)
        p = clf.predict_proba(Xv)
        test_preds.append(p)
        print(f"  seed {s}: val {accuracy_score(yv, p.argmax(1)):.4f} "
              f"({clf.named_steps['mlp'].n_iter_} epochs)")
    if len(test_preds) > 1:
        avg = np.mean(test_preds, axis=0)
        print(f"  ensemble of {len(test_preds)}: {accuracy_score(yv, avg.argmax(1)):.4f}")


if __name__ == "__main__":
    main()

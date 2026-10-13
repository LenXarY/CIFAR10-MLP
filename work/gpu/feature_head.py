"""Feasibility test: frozen CNN conv features + a scikit-learn MLP head.

If this lands close to the CNN's own accuracy, then baking the learned filters
into the submission pipeline is worth the engineering; if it collapses, it is not.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, Net, load_data  # noqa: E402


def extract(model, X, mean, std, device, chunk=2000):
    """Global-average-pooled features from the conv stack."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.from_numpy(X[i : i + chunk]).permute(0, 3, 1, 2).float().to(device)
            x = (x / 255.0 - mean) / std
            f = model.features
            for k in range(model.depth):
                x = f[k * 3 : k * 3 + 3](x)
                if k < 3:
                    x = F.max_pool2d(x, 2)
            out.append(x.mean((2, 3)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work/gpu/out/cnn_aug_w64.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", default="512,256")
    args = ap.parse_args()

    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, weights_only=False)
    a = ckpt["args"]
    model = Net(a["width"], a["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(device)
    print(f"cnn reported val: {ckpt['val_acc']:.4f}", flush=True)

    X, y, Xv, yv = load_data()
    hold = a.get("holdout", 5000)
    X, y = X[: len(X) - hold], y[: len(y) - hold]

    t0 = time.time()
    Ftr = extract(model, X, mean, std, device)
    Fva = extract(model, Xv, mean, std, device)
    print(f"features {Ftr.shape} / {Fva.shape} in {time.time() - t0:.0f}s", flush=True)

    hidden = tuple(int(v) for v in args.hidden.split(","))
    clf = Pipeline([
        ("standardize", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256,
                              learning_rate_init=1e-3, max_iter=args.epochs, random_state=0)),
    ])
    clf.fit(Ftr, y)
    acc = accuracy_score(yv, clf.predict(Fva))
    print(f"sklearn head on frozen cnn features: {acc:.4f}  ({clf.named_steps['mlp'].n_iter_} epochs)")

    # how much does the model's own 128->10 layer get? (upper bound for this head)
    with torch.no_grad():
        fv = torch.from_numpy(Fva).to(device)
        acc_fc = (model.fc(fv).argmax(1).cpu().numpy() == yv).mean()
    print(f"the cnn's own linear head      : {acc_fc:.4f}")


if __name__ == "__main__":
    main()

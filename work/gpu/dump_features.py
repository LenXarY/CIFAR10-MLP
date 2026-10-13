"""Dump the frozen conv-stack features to .npy so the scikit-learn side can be
run in the *submission* environment (no torch needed there)."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, Net, load_data  # noqa: E402


def extract(model, X, mean, std, device, chunk=2000, pool=True):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.from_numpy(X[i : i + chunk]).permute(0, 3, 1, 2).float().to(device)
            x = (x / 255.0 - mean) / std
            for k in range(model.depth):
                x = model.features[k * 3 : k * 3 + 3](x)
                if k < 3:
                    x = F.max_pool2d(x, 2)
            out.append((x.mean((2, 3)) if pool else x).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="prefix for the .npy files")
    ap.add_argument("--pool", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, weights_only=False)
    a = ckpt["args"]
    model = Net(a["width"], a["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(device)

    X, y, Xv, yv = load_data()
    hold = a.get("holdout", 5000)
    X, y = X[: len(X) - hold], y[: len(y) - hold]

    t0 = time.time()
    for name, Xi in (("train", X), ("val", Xv)):
        Fx = extract(model, Xi, mean, std, device, pool=bool(args.pool))
        np.save(f"{args.out}_{name}.npy", Fx)
        print(f"  {name} {Fx.shape} -> {args.out}_{name}.npy  ({time.time() - t0:.0f}s)", flush=True)
    np.save(f"{args.out}_y_train.npy", y)
    np.save(f"{args.out}_y_val.npy", yv)
    print(f"dump done in {time.time() - t0:.0f}s; cnn val was {ckpt['val_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()

"""Dump the baked (pure NumPy) convolution features for the whole dataset."""

import os
import sys
import time
import tracemalloc

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake import apply_steps, bake  # noqa: E402
from bake_net import BakeNet  # noqa: E402
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, load_data  # noqa: E402


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "work/gpu/out/bake_pool.pt"
    out = sys.argv[2] if len(sys.argv) > 2 else "work/gpu/out/baked"
    ckpt = torch.load(ckpt_path, weights_only=False)
    a = ckpt["args"]
    widths = tuple(int(v) for v in a["widths"].split(","))
    strides = tuple(int(v) for v in a["strides"].split(","))
    steps = bake(ckpt["state_dict"], widths, pool=a.get("pool", False),
                 resolutions=(32, 32 // 2, 32 // 4, 32 // 8)[: len(widths)],
                 norm=(CIFAR_MEAN, CIFAR_STD))

    X, y, Xv, yv = load_data()
    hold = a.get("holdout", 5000)
    X, y = X[: len(X) - hold], y[: len(y) - hold]

    t0 = time.time()
    np.save(f"{out}_train.npy", apply_steps(steps, X).astype(np.float32))
    np.save(f"{out}_val.npy", apply_steps(steps, Xv).astype(np.float32))
    np.save(f"{out}_y_train.npy", y)
    np.save(f"{out}_y_val.npy", yv)
    print(f"features dumped in {time.time() - t0:.0f}s (cnn val {ckpt['val_acc']:.4f})")

    # peak memory the *submission* pipeline needs for a 5,000-image prediction
    tracemalloc.start()
    _ = apply_steps(steps, Xv)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"peak python allocation for a 5000-image transform: {peak / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

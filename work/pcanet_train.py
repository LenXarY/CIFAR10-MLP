"""Learn the two-layer PCA filter banks and cache them to disk."""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcanet_lib import learn, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="cache id, e.g. pc8")
    ap.add_argument("--n-l1", type=int, default=8)
    ap.add_argument("--n-l2", type=int, default=8)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--n-patch", type=int, default=120000)
    ap.add_argument("--n-images", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("work/cache", exist_ok=True)
    X = np.load("CIFAR10_Data/x_train.npy")
    t0 = time.time()
    filters = learn(
        X,
        n_l1=args.n_l1,
        n_l2=args.n_l2,
        patch=args.patch,
        n_patch=args.n_patch,
        n_images=args.n_images,
        seed=args.seed,
    )
    path = f"work/cache/pcanet_{args.name}.npz"
    save(filters, path)
    print(
        f"learned {args.n_l1} stage-1 and {args.n_l1}x{args.n_l2} stage-2 filters "
        f"in {time.time() - t0:.0f}s -> {path}"
    )


if __name__ == "__main__":
    main()

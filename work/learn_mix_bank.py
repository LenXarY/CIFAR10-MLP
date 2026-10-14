"""Learn a stage-1 bank plus its 1x1 channel-mixing matrix, and cache them."""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix_lib  # noqa: E402
from pcanet_lib import kmeans_filters, pca_filters, sample_patches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mix16")
    ap.add_argument("--k1", type=int, default=16)
    ap.add_argument("--l", type=int, default=64)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--cells1", type=int, default=8)
    ap.add_argument("--n-patch", type=int, default=150000)
    ap.add_argument("--n-images", type=int, default=10000)
    ap.add_argument("--method", default="kmeans", choices=["kmeans", "pca"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("work/cache", exist_ok=True)
    X = np.load("CIFAR10_Data/x_train.npy")
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(X), args.n_images, replace=False))
    gray = X[idx].mean(3).astype(np.float32)

    t0 = time.time()
    patches = sample_patches(gray, args.patch, args.n_patch, rng)
    if args.method == "kmeans":
        kernels = kmeans_filters(patches, args.k1, seed=args.seed)
    else:
        kernels = pca_filters(patches, args.k1)
    mixing = mix_lib.learn_mixing(X, kernels, cells1=args.cells1, seed=args.seed,
                                  n_components=args.l)
    mix_lib.MIX_STORE[args.name] = {"kernels": kernels, "mixing": mixing}
    mix_lib.save(args.name, f"work/cache/{args.name}.npz")
    print(f"{args.k1} kernels + ({kernels.shape[0]},{mixing.shape[1]}) mixing in "
          f"{time.time() - t0:.0f}s -> work/cache/{args.name}.npz")


if __name__ == "__main__":
    main()

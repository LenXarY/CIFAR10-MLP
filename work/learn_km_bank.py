"""Learn the two-layer filter bank with scikit-learn k-means and cache it."""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcanet_lib import learn, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-l1", type=int, default=24)
    ap.add_argument("--n-l2", type=int, default=8)
    ap.add_argument("--patch", type=int, default=6)
    ap.add_argument("--n-patch", type=int, default=150000)
    ap.add_argument("--n-images", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmented", action="store_true",
                    help="also draw patches from corrupted copies of the images")
    args = ap.parse_args()

    os.makedirs("work/cache", exist_ok=True)
    X = np.load("CIFAR10_Data/x_train.npy")
    if args.augmented:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from corruptions import CORRUPTION_NAMES, corrupt_batch

        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), 12000, replace=False)
        extra = X[idx].copy()
        choice = rng.integers(0, len(CORRUPTION_NAMES), len(extra))
        sev = rng.integers(1, 4, len(extra))
        for k in np.unique(choice):
            for s in np.unique(sev[choice == k]):
                m = (choice == k) & (sev == s)
                extra[m] = corrupt_batch(extra[m], CORRUPTION_NAMES[k], int(s), rng)
        X = np.concatenate([X, extra])
        print(f"augmented pool: {X.shape}")

    t0 = time.time()
    filters = learn(
        X, n_l1=args.n_l1, n_l2=args.n_l2, patch=args.patch,
        n_patch=args.n_patch, n_images=args.n_images, seed=args.seed, method="kmeans",
    )
    path = f"work/cache/pcanet_{args.name}.npz"
    save(filters, path)
    print(f"learned {args.n_l1} stage-1 and {args.n_l1}x{args.n_l2} stage-2 "
          f"k-means filters in {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()

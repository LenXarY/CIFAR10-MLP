"""Sanity-check the course data before spending an hour on the real Run All."""

import collections
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ("CIFAR10_Data", "x_train.npy"),
    ("CIFAR10_Data", "y_train.npy"),
    ("validation_set_to_students", "images.npy"),
    ("validation_set_to_students", "labels.npy"),
    ("", "Model_Evaluation.ipynb"),
]


def main():
    for folder, name in FILES:
        path = os.path.join(ROOT, folder, name)
        if os.path.exists(path):
            print(f"OK      {os.path.join(folder, name):45s} {os.path.getsize(path) / 1e6:7.1f} MB")
        else:
            print(f"MISSING {os.path.join(folder, name)}")

    needed = [os.path.join(ROOT, f, n) for f, n in FILES[:4]]
    if not all(os.path.exists(p) for p in needed):
        print("\nsome data files are missing")
        return 1

    x_train = np.load(os.path.join(ROOT, "CIFAR10_Data", "x_train.npy"), mmap_mode="r")
    y_train = np.load(os.path.join(ROOT, "CIFAR10_Data", "y_train.npy"))
    x_val = np.load(os.path.join(ROOT, "validation_set_to_students", "images.npy"), mmap_mode="r")
    y_val = np.load(os.path.join(ROOT, "validation_set_to_students", "labels.npy"))

    print()
    print("x_train", x_train.shape, x_train.dtype)
    print("y_train", y_train.shape, y_train.dtype, "-> ravels to", y_train.ravel().shape)
    print("x_val  ", x_val.shape, x_val.dtype)
    print("y_val  ", y_val.shape, y_val.dtype)

    print("train class counts", sorted(collections.Counter(y_train.ravel().tolist()).items()))
    print("val   class counts", sorted(collections.Counter(y_val.tolist()).items()))

    sample = np.asarray(x_train[::100]).reshape(-1, 3)
    print("train per-channel mean", np.round(sample.mean(0), 2), "std", np.round(sample.std(0), 2))
    print("(reference CIFAR-10: mean [125.31 122.95 113.87] std [62.99 62.09 66.70])")
    return 0


if __name__ == "__main__":
    sys.exit(main())

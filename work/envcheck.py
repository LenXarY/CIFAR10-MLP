"""End-to-end environment check.

Builds a tiny synthetic dataset in work/envcheck/, executes a scaled-down copy
of train_models.ipynb with the registered aiaa3102 kernel, then reloads the
saved models. Proves that Python 3.14.6 + numpy 2.4.6 + scipy 1.18.0 +
scikit-learn 1.9.0 + joblib 1.5.3 can run the real pipeline on this machine,
without needing the course data.
"""

import os
import sys
import time

import nbformat
import numpy as np
from nbclient import NotebookClient
from sklearn.metrics import accuracy_score

from run_notebook import prepare_jupyter_env


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(ROOT, "work", "envcheck")
N_TRAIN, N_VAL = 3000, 500


def make_data():
    """Synthetic CIFAR-shaped data: class-dependent colour + noise."""
    rng = np.random.default_rng(0)
    colours = np.array(
        [
            [200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40],
            [200, 40, 200], [40, 200, 200], [140, 140, 140], [230, 230, 230],
            [90, 40, 10], [10, 90, 40],
        ],
        dtype=np.float32,
    )

    def batch(n):
        y = rng.integers(0, 10, n)
        x = rng.normal(0, 26, (n, 32, 32, 3)).astype(np.float32)
        x += colours[y][:, None, None, :] * 0.6
        return np.clip(x, 0, 255).astype(np.uint8), y

    x_tr, y_tr = batch(N_TRAIN)
    x_va, y_va = batch(N_VAL)
    np.save(os.path.join(SCRATCH, "CIFAR10_Data", "x_train.npy"), x_tr)
    np.save(os.path.join(SCRATCH, "CIFAR10_Data", "y_train.npy"), y_tr.reshape(-1, 1))
    np.save(os.path.join(SCRATCH, "validation_set_to_students", "images.npy"), x_va)
    np.save(os.path.join(SCRATCH, "validation_set_to_students", "labels.npy"), y_va.astype(np.int64))


def scaled_notebook():
    replacements = [
        ('"n_clean": 50000', f'"n_clean": {N_TRAIN}'),
        ('"n_shift": 30000', '"n_shift": 600'),
        ('"n_shift2": 20000', '"n_shift2": 300'),
        ('"n_corr": 30000', '"n_corr": 600'),
        ('"pca": 512', '"pca": 64'),
        ('"epochs": 55', '"epochs": 6'),
        (', (768,)),', ', (64,)),'),
        ('(512, 512)),', '(32, 32)),'),
        ('MODEL_FILES = {', 'MODEL_FILES = {  # envcheck'),
    ]
    nb = nbformat.read(os.path.join(ROOT, "train_models.ipynb"), as_version=4)
    hits = {old: 0 for old, _ in replacements}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for old, new in replacements:
            if old in cell.source:
                hits[old] += cell.source.count(old)
                cell.source = cell.source.replace(old, new)
    missing = [old for old, n in hits.items() if n == 0]
    if missing:
        raise SystemExit(f"patterns not found in the notebook: {missing}")
    return nb


def main():
    prepare_jupyter_env()
    for sub in ("CIFAR10_Data", "validation_set_to_students"):
        os.makedirs(os.path.join(SCRATCH, sub), exist_ok=True)

    print("building synthetic data ...", flush=True)
    make_data()

    nb = scaled_notebook()
    t0 = time.time()
    NotebookClient(
        nb,
        timeout=3600,
        kernel_name="aiaa3102",
        resources={"metadata": {"path": SCRATCH}},
    ).execute()
    print(f"\nnotebook executed in {time.time() - t0:.0f}s", flush=True)

    failed = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                print(f"[cell {i}] {out.text.rstrip()}")
            elif out.output_type == "error":
                failed = True
                print(f"[cell {i}] ERROR {out.ename}: {out.evalue}")

    if failed:
        return 1

    import joblib

    y_va = np.load(os.path.join(SCRATCH, "validation_set_to_students", "labels.npy"))
    x_va = np.load(os.path.join(SCRATCH, "validation_set_to_students", "images.npy"))
    print("\nreload check")
    for seed in (42, 2026, 3102):
        path = os.path.join(SCRATCH, f"model_seed{seed}.joblib")
        model = joblib.load(path)
        acc = accuracy_score(y_va, model.predict(x_va))
        print(f"  model_seed{seed}.joblib  {acc:.4f}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    print("\nenvcheck finished without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

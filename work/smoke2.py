"""Execute a scaled-down copy of train_models.ipynb to catch cell-level errors
before spending an hour on the real Run All."""

import os
import sys

import nbformat
from nbclient import NotebookClient

from run_notebook import prepare_jupyter_env, pick_kernel


REPLACEMENTS = [
    ('"n_clean": 50000,', '"n_clean": 3000,'),
    ('"n_shift": 30000,', '"n_shift": 600,'),
    ('"n_shift2": 20000,', '"n_shift2": 300,'),
    ('"n_corr13": 30000,', '"n_corr13": 600,'),
    ('"n_corr15": 30000,', '"n_corr15": 600,'),
    ('"pca": 768,', '"pca": 64,'),
    ('"hidden": (1024,),', '"hidden": (64,),'),
    ('"epochs": 55,', '"epochs": 3,'),
    ('"n_members": 3,', '"n_members": 2,'),
    ('"prep_fit_rows": 60000,', '"prep_fit_rows": 4000,'),
    ('n_patch=120000', 'n_patch=20000'),
    ('n_images=8000', 'n_images=2000'),
    ('"model_seed42.joblib"', '"work/models/smoke_seed42.joblib"'),
    ('"model_seed2026.joblib"', '"work/models/smoke_seed2026.joblib"'),
    ('"model_seed3102.joblib"', '"work/models/smoke_seed3102.joblib"'),
]


def main():
    prepare_jupyter_env()
    nb = nbformat.read("train_models.ipynb", as_version=4)
    hits = {old: 0 for old, _ in REPLACEMENTS}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for old, new in REPLACEMENTS:
            if old in cell.source:
                hits[old] += cell.source.count(old)
                cell.source = cell.source.replace(old, new)
    missing = [old for old, n in hits.items() if n == 0]
    if missing:
        print("patterns not found:", missing)
        return 1

    client = NotebookClient(nb, timeout=3600, kernel_name=pick_kernel())
    client.execute()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                print(f"[cell {i}] {out.text.strip()[:600]}")
            elif out.output_type == "error":
                print(f"[cell {i}] ERROR {out.ename}: {out.evalue}")
                return 1
    print("\nsmoke run finished without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Execute a scaled-down copy of train_models.ipynb to catch cell-level errors
before spending an hour on the real Run All."""

import json
import os
import sys

import nbformat
from nbclient import NotebookClient


def main():
    nb = nbformat.read("train_models.ipynb", as_version=4)
    replacements = [
        ('"n_clean": 50000', '"n_clean": 3000'),
        ('"n_shift": 30000', '"n_shift": 300'),
        ('"n_shift2": 20000', '"n_shift2": 150'),
        ('"n_corr": 30000', '"n_corr": 300'),
        ('"epochs": 55', '"epochs": 3'),
        ('"model_seed42.joblib"', '"work/models/smoke_seed42.joblib"'),
        ('"model_seed2026.joblib"', '"work/models/smoke_seed2026.joblib"'),
        ('"model_seed3102.joblib"', '"work/models/smoke_seed3102.joblib"'),
        (', (768,)),', ', (64,)),'),
        ('(512, 512)),', '(64, 32)),'),
    ]
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
        print("patterns not found:", missing)
        return 1

    client = NotebookClient(nb, timeout=1200, kernel_name="aiaa3102")
    client.execute()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                print(f"[cell {i}] {out.text.strip()[:400]}")
            elif out.output_type == "error":
                print(f"[cell {i}] ERROR {out.ename}: {out.evalue}")
                return 1
    print("\nsmoke run finished without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())

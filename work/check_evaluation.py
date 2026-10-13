"""Run the provided Model_Evaluation.ipynb exactly as the course would,
without modifying the file."""

import os
import sys

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from run_notebook import pick_kernel


def main():
    kernel = pick_kernel()
    print(f"using kernel: {kernel}", flush=True)
    nb = nbformat.read("Model_Evaluation.ipynb", as_version=4)
    client = NotebookClient(
        nb, timeout=7200, kernel_name=kernel, resources={"metadata": {"path": "."}}
    )
    client.execute()
    failed = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                print(out.text.rstrip())
            elif out.output_type == "error":
                failed = True
                print(f"[cell {i}] ERROR {out.ename}: {out.evalue}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Execute train_models.ipynb in place (Run All) and keep the outputs."""

import os
import sys
import time

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def prepare_jupyter_env():
    """Keep Jupyter/IPython runtime files inside the project directory.

    The default locations (``%APPDATA%\\jupyter`` for connection files,
    ``~/.ipython`` for the IPython profile) are not writable on a locked-down
    machine, and the kernel dies before answering ``kernel_info`` when it
    cannot create its profile directory. The kernelspec itself stays in the
    default location, so ``JUPYTER_DATA_DIR`` is deliberately left alone.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(root, ".jupyter")
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", os.path.join(local, "runtime"))
    os.environ.setdefault("IPYTHONDIR", os.path.join(local, "ipython"))
    for path in (os.environ["JUPYTER_RUNTIME_DIR"], os.environ["IPYTHONDIR"]):
        os.makedirs(path, exist_ok=True)


prepare_jupyter_env()


def pick_kernel(preferred=("aiaa3102", "python3")):
    """Find an installed kernel, falling back to whatever Python 3 kernel exists."""
    specs = KernelSpecManager().get_all_specs()
    for name in preferred:
        if name in specs:
            return name
    for name, spec in specs.items():
        if spec.get("spec", {}).get("language") == "python":
            return name
    raise SystemExit(f"no python kernel installed; available: {sorted(specs)}")


def main():
    kernel = pick_kernel()
    print(f"using kernel: {kernel}", flush=True)
    nb = nbformat.read("train_models.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=21600, kernel_name=kernel)
    t0 = time.time()
    client.execute()
    nbformat.write(nb, "train_models.ipynb")
    print(f"executed in {time.time() - t0:.0f}s")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.output_type == "error":
                print(f"[cell {i}] ERROR {out.ename}: {out.evalue}")
                return 1
            if out.output_type == "stream":
                print(f"[cell {i}] {out.text.strip()[-300:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

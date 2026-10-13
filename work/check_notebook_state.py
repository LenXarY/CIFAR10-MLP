"""Print the state of train_models.ipynb (cells, outputs, recipe values)."""

import os
import time

import nbformat


def main():
    path = "train_models.ipynb"
    nb = nbformat.read(path, as_version=4)
    print("cells:", len(nb.cells))
    print("outputs per code cell:", [len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code"])
    print("mtime:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))))
    src = "".join(c.source for c in nb.cells if c.cell_type == "code")
    for key in ["'n_shift': 30000", "'n_corr': 30000", "'n_shift2': 20000", "'epochs': 55"]:
        alt = key.replace("'", '"')
        print(f"{key} present:", key in src or alt in src)


if __name__ == "__main__":
    main()

"""Append the measured full-scale result to the notebook's method write-up.

Only the markdown cell is touched; the executed code cells and their stored
outputs are left exactly as they came out of Run All.
"""

import nbformat as nbf


MEASURED = """

## Measured result

Run All on this machine (20 logical cores, no GPU) took 1.9 hours and wrote three 46 MB
model files. Re-running the unmodified `Model_Evaluation.ipynb` against them gives:

| Model file | Validation accuracy |
| --- | --- |
| `model_seed42.joblib` | 0.6356 |
| `model_seed2026.joblib` | 0.6366 |
| `model_seed3102.joblib` | 0.6352 |
| **average** | **0.6358** |

The screening numbers quoted in the tables above were measured on a 16,000-image budget;
these are the numbers from the full 50,000-image run.
"""


def main():
    nb = nbf.read("train_models.ipynb", as_version=4)
    before = [len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code"]
    assert nb.cells[0].cell_type == "markdown"
    if "## Measured result" not in nb.cells[0].source:
        nb.cells[0].source = nb.cells[0].source.replace(
            "and trains one MLP per member per seed on a CPU. Nothing here uses a GPU.",
            "and trains one MLP per member per seed on a CPU. Nothing here uses a GPU.\n"
            "Expect roughly 40 minutes per seed.",
        ) + MEASURED
    nbf.write(nb, "train_models.ipynb")
    nb2 = nbf.read("train_models.ipynb", as_version=4)
    after = [len(c.get("outputs", [])) for c in nb2.cells if c.cell_type == "code"]
    print("code outputs before:", before)
    print("code outputs after :", after)
    print("unchanged:", before == after)


if __name__ == "__main__":
    main()

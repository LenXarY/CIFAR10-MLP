"""Add the measured results and the TTA write-up to the notebook markdown."""

import nbformat as nbf

P1 = "Each member is also present a second time\nbehind a horizontal-flip transform: because the flipped member shares the *same fitted*\nhead object, the flip view is test-time augmentation that costs no extra model size."
P2 = "Every member is then evaluated once per *view*: the identity, each of the eight shifts of\n1, 2, 3 and 4 pixels in both axes, and the same 16 shifted views again after a horizontal\nflip. That is 33 views per member and 99 voting members per model. Each view shares the\n*same fitted* head object, so the whole view bank is test-time augmentation and adds no\nmodel size at all - the view transform is merely a padded shift, built from module-level\nNumPy callables (`np.pad` + `np.take`) so the model still unpickles with nothing but\nnumpy, joblib and scikit-learn. Views are cheap on disk and expensive at prediction time,\nwhich is why the evaluation cell takes minutes rather than seconds; the accuracy they buy\nis large, and it is the biggest single lever in this solution."

MEASURED = "\n\n## Measured result\n\nRun All on this machine (20 logical cores, no GPU) took 3.3 hours and wrote three 46 MB\nmodel files. Re-running the unmodified `Model_Evaluation.ipynb` against them gives:\n\n| Model file | Validation accuracy |\n| --- | --- |\n| `model_seed42.joblib` | 0.6642 |\n| `model_seed2026.joblib` | 0.6678 |\n| `model_seed3102.joblib` | 0.6684 |\n| **average** | **0.6668** |\n\nThe screening numbers quoted in the tables above were measured on a reduced training\nbudget; these are the numbers from the full 50,000-image run.\n"


def main():
    nb = nbf.read("train_models.ipynb", as_version=4)
    before = [len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code"]
    src = nb.cells[0].source
    if P1 in src:
        src = src.replace(P1, P2)
        print("replaced the ensemble paragraph")
    elif P2 in src:
        print("paragraph already updated")
    else:
        raise SystemExit("anchor paragraph not found")
    if "## Measured result" not in src:
        src += MEASURED
    nb.cells[0].source = src
    nbf.write(nb, "train_models.ipynb")
    nb2 = nbf.read("train_models.ipynb", as_version=4)
    after = [len(c.get("outputs", [])) for c in nb2.cells if c.cell_type == "code"]
    print("outputs before:", before)
    print("outputs after :", after)
    print("outputs preserved:", before == after)


if __name__ == "__main__":
    main()

"""Fast structural check of the notebook's TTA view ensemble."""

import os

import joblib
import nbformat as nbf
import numpy as np


def main():
    nb = nbf.read("train_models.ipynb", as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    # everything except the training loop and the verification cell
    src = "\n".join(c.source for c in code_cells[:-2])
    g = {"__name__": "nb"}
    exec(compile("import os, time, io\n" + src, "nb", "exec"), g)

    X = np.load("CIFAR10_Data/x_train.npy")[:400]
    y = np.load("CIFAR10_Data/y_train.npy").ravel()[:400]
    Xv = np.load("validation_set_to_students/images.npy")[:60]
    yv = np.load("validation_set_to_students/labels.npy")[:60]

    recipe = dict(g["RECIPE"])
    recipe.update(n_clean=400, n_shift=100, n_shift2=60, n_corr13=100, n_corr15=100,
                  n_corr_layered=100, pca=64, hidden=(64,), epochs=3, n_members=2,
                  prep_fit_rows=300, seeds=[42])
    filters = g["learn_two_layer_filters"](X, seed=42, n_l1=4, n_l2=4, patch=5,
                                           n_patch=5000, n_images=200)
    feats = g["build_feature_block"](filters)
    model = g["build_model"](42, X, y, feats, recipe)
    pred = model.predict(Xv)
    print("ensemble voters:", len(model.estimators_))
    print("predict ok:", pred.shape, "acc", (pred == yv).mean())

    path = "work/models/ttacheck.joblib"
    joblib.dump(model, path)
    print("size KB", round(os.path.getsize(path) / 1024))
    again = joblib.load(path)
    print("reload identical:", bool((again.predict(Xv) == pred).all()))


if __name__ == "__main__":
    main()

# AIAA 3102 Assignment 1 — migration package

Everything needed to finish the assignment on another machine. The heavy part
(training the three submitted models) is done by running one notebook.

## 0. Status on this machine (desktop, `D:\AIAA3102-migration`)

**Finished.** Environment is installed, the models are trained, verified and
ready to package.

| Item | State |
| --- | --- |
| Python 3.14.6 | `C:\Users\Mayn\AppData\Local\Programs\Python\Python314` |
| `.venv314` | NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3 |
| Jupyter kernel | `aiaa3102` registered |
| Data | `CIFAR10_Data/`, `validation_set_to_students/`, `Model_Evaluation.ipynb` all in place |
| `train_models.ipynb` | Run All completed in 3.3 h, outputs retained |
| `model_seed{42,2026,3102}.joblib` | 46.2 MB each, all below the 100 MB limit |
| Verification | the unmodified course `Model_Evaluation.ipynb` reports **0.6668 average** |

Per-seed validation accuracy: 0.6642 / 0.6678 / 0.6684.

Two changes produced the whole gain over the earlier 0.6358 version:

* **Test-time augmentation over a shift grid.** Every member is scored on 33 views
  (identity, +/-1..4 pixel shifts in both axes, and the same 16 shifted views after a
  horizontal flip). The views share one fitted head, so they add no model size at all.
  +2.2 points, and by far the biggest single lever.
* **A double-corruption component in the training mixture** (45,000 images carrying two
  CIFAR-10-C corruptions at once). +1.0 point.

Everything else tried was neutral or worse - wider MLPs, higher PCA, multi-scale or
k-means filter banks, max pooling, channel mixing, learning-rate annealing. Screening
numbers and the rejected variants are logged in `work/lab_results.jsonl`.

**Still to do:** build the submission ZIP with the student's name and number
(section 7) — `work/package.py "NAME, Givenname-STUDENTID"`.

## 1. What is in this package

| Path | What it is |
| --- | --- |
| `train_models.ipynb` | The final training notebook. Run All = data processing + training + saving the three models, with outputs kept. |
| `reference/hog_trim_seed42.joblib` | A validated model from this exact recipe (seed 42, 59.30% on the provided validation set). Use it to smoke-test `Model_Evaluation.ipynb` before the long run. |
| `requirements.txt` | The pinned package versions. |
| `work/` | The experiment toolchain used to pick the recipe (feature sweeps, seed comparisons, verification scripts). Optional, not needed for the submission. |

## 2. Environment (must match the course requirement exactly)

The submitted `.joblib` files must be produced with:

* Python 3.14.6
* NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3

Setup on the new machine:

```powershell
# 1. install Python 3.14.6 (python.org installer), then:
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt

# 2. register a kernel so the notebook can use this interpreter
.\.venv314\Scripts\python.exe -m ipykernel install --user --name aiaa3102 `
    --display-name "Python 3.14.6 (AIAA3102)"
```

`requirements.txt` installs the four pinned packages plus `ipykernel`,
`nbclient` and `nbconvert` (the last two only to run the notebook headlessly).
If the machine has no internet, copy the wheels over instead.

## 3. Data layout

Put the course data next to the notebook:

```
train_models.ipynb
CIFAR10_Data/x_train.npy            # (50000, 32, 32, 3) uint8
CIFAR10_Data/y_train.npy            # (50000, 1) uint8  (code ravels it)
validation_set_to_students/images.npy   # (5000, 32, 32, 3) uint8
validation_set_to_students/labels.npy   # (5000,) int64
Model_Evaluation.ipynb              # the course file, unmodified
```

## 4. How to run

```powershell
# headless (recommended, keeps the outputs in the notebook file)
.\.venv314\Scripts\python.exe work\run_notebook.py
```

or open `train_models.ipynb` in Jupyter, select the `aiaa3102` kernel, and use
**Run All**. Either way the notebook writes `model_seed42.joblib`,
`model_seed2026.joblib` and `model_seed3102.joblib` into the working directory
with the execution outputs retained.

**Runtime:** about 33 minutes per seed on an 18-core / no-GPU laptop
(~100 minutes total). A desktop CPU with more cores will be faster. Nothing here
uses a GPU.

**Expected result** (measured on this machine, seed 42, same recipe):
validation accuracy ≈ 59.3%, each model file ≈ 48 MB.

## 5. Verifying before submission

```powershell
# runs the course notebook unmodified and prints its output
.\.venv314\Scripts\python.exe work\check_evaluation.py
```

It must print a validation accuracy for all three files and the average. Each
file must stay below 100 MB.

## 6. Why the solution looks like this (short version, for the report)

* **Only scikit-learn is allowed**, and `Model_Evaluation.ipynb` imports nothing
  but `numpy`, `joblib` and `scikit-learn`. A pickled `Pipeline`
  therefore may not contain any custom class — so the preprocessing is built
  only from `scipy.ndimage.convolve` plus NumPy ops, whose `FunctionTransformer`
  wrappers pickle by module reference.
* **The provided validation set is a clean + corrupted mixture.** ~24% of its
  images have >2% pure black pixels versus 2% for clean training images, i.e. it
  contains pixel-clipping corruptions in the style of CIFAR-10-C. Training with
  CIFAR-10-C style augmentation (20 corruption types, severities 1-3) is
  therefore part of the recipe.
* **Features matter more than the classifier.** Raw pixels top out around 51%
  clean / 43% validation with this budget. Replacing them with HOG-style
  oriented-filter features (8 orientations x 2 scales + 2 Laplacians + colour
  cell means, pooled over an 8x8 grid of 4x4-pixel cells, then PCA-512) lifts a
  single model to 66% clean / 54% validation. Measured feature variants:
  raw+PCA 43.3% val, HOG+PCA 52.5% val, finer 2x2-pixel cells 48.5%,
  HOG+raw-pixels concatenation 49.9%, more filter scales 52.2%.
* **Ensembling on top.** Each submitted model is a soft-voting ensemble of five
  MLPs trained on five different mixtures (clean, clean+shifted,
  clean+corrupted at severity 1-3, clean+corrupted at severity 1-5, clean+shifted
  with a 2-layer head). Members score 53-55% individually, the ensemble 59.3%.
* **Seeds.** Pipeline, mixtures and hyper-parameters are fixed; the three models
  differ only by the training seed (42 / 2026 / 3102), as required.

## 7. Building the final submission ZIP

```powershell
.\.venv314\Scripts\python.exe work\package.py "NAME, Givenname-STUDENTID"
```

creates `NAME, Givenname-STUDENTID-AIAA3102-Assignment1.zip` containing exactly
the four required files: `train_models.ipynb` + the three `model_seed*.joblib`.

## 8. Fallback material

The earlier, weaker recipes are kept for comparison in `work/backup_5member_raw/`
(raw-pixel ensemble, 49.9% average) and `work/models/` (experiment models).
They are not part of the submission.

## 9. What to copy to the new machine

Copy only the source material — the environment and the data are machine-specific:

| Copy | Do not copy |
| --- | --- |
| `train_models.ipynb` | `.venv314/` (400+ MB, recreate with step 2) |
| `requirements.txt`, `work/` | `.jupyter/` (kernel registration, recreate) |
| `Model_Evaluation.ipynb` | `work/models/`, `work/backup_*/`, `work/partial_run_*/` |
| `CIFAR10_Data/`, `validation_set_to_students/` (copy by hand or over the network: `x_train.npy` is ~154 MB, above GitHub's 100 MB file limit, so do not push it to a plain git repo) | any `*.joblib` in the working directory (regenerated by the notebook) |

The working directory of this package currently contains **no** `.joblib` files:
the last run here was interrupted during seed 3102, so its outputs were moved to
`work/partial_run_seed42_2026/` (two HOG models and one older raw-pixel model —
inconsistent set, do not submit it). Running the notebook regenerates a
consistent set of three.

"""Training experiments: clean vs. augmented vs. corruption-augmented.

Usage example:
    python work/experiment.py --n-clean 30000 --n-corr 30000 --hidden 512 --max-iter 40
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, "work")
from corruptions import CORRUPTION_NAMES, corrupt_batch  # noqa: E402


DATA = "CIFAR10_Data"
VAL = "validation_set_to_students"


def flatten(x):
    return x.reshape(len(x), -1).astype(np.float32) / 255.0


def random_flip_shift(images, rng, max_shift=3, cutout=8):
    """Vectorised horizontal flip + translation + cutout on uint8 batches."""
    x = images.copy()
    flip = rng.random(len(x)) < 0.5
    if flip.any():
        x[flip] = x[flip][:, :, ::-1]
    dy = rng.integers(-max_shift, max_shift + 1, len(x))
    dx = rng.integers(-max_shift, max_shift + 1, len(x))
    ys = np.clip(np.arange(32)[None, :, None] + dy[:, None, None], 0, 31)
    xs = np.clip(np.arange(32)[None, None, :] + dx[:, None, None], 0, 31)
    rows = np.arange(len(x))[:, None, None]
    x = x[rows, ys, xs]
    if cutout:
        cy = rng.integers(0, 32, len(x))
        cx = rng.integers(0, 32, len(x))
        half = cutout // 2
        yy = np.arange(32)[None, :, None]
        xx = np.arange(32)[None, None, :]
        box = (
            (np.abs(yy - cy[:, None, None]) <= half) & (np.abs(xx - cx[:, None, None]) <= half)
        )[..., None]
        x = np.where(box, 0, x)
    return x


def make_corrupted(images, rng, kinds=None, sev_lo=1, sev_hi=5):
    kinds = CORRUPTION_NAMES if kinds is None else kinds
    out = np.empty_like(images)
    choice = rng.choice(len(kinds), len(images))
    sev = rng.integers(sev_lo, sev_hi + 1, len(images))
    for k in np.unique(choice):
        for s in np.unique(sev[choice == k]):
            m = (choice == k) & (sev == s)
            out[m] = corrupt_batch(images[m], kinds[k], int(s), rng)
    return out


def make_mild(images, rng):
    """Mild, mostly invisible degradations: light JPEG and slight blur.

    These model "the whole set was saved/re-encoded once" style shift.
    """
    out = images.copy()
    choice = rng.integers(0, 3, len(images))

    m = choice == 0
    if m.any():
        from corruptions import jpeg_compression

        x = out[m].astype(np.float32) / 255.0
        # linear quality ramp inside jpeg_compression is not available, so call
        # it at the two mildest severities and interpolate by mixing.
        q_hi = jpeg_compression(x, 1, rng) * 255.0
        q_lo = jpeg_compression(x, 2, rng) * 255.0
        w = rng.random((len(x), 1, 1, 1)).astype(np.float32)
        out[m] = np.clip(q_lo * w + q_hi * (1 - w), 0, 255).astype(np.uint8)

    m = choice == 1
    if m.any():
        sigma = rng.uniform(0.3, 0.7, len(images[m]))
        blurred = np.stack(
            [
                ndimage.gaussian_filter(img.astype(np.float32), sigma=(s, s, 0), mode="reflect")
                for img, s in zip(images[m], sigma)
            ]
        )
        out[m] = np.clip(blurred, 0, 255).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=30000)
    ap.add_argument("--n-corr", type=int, default=0)
    ap.add_argument("--n-shift", type=int, default=0, help="flip/shift/cutout copies of clean images")
    ap.add_argument("--n-mild", type=int, default=0, help="mild re-encoding style degradation copies")
    ap.add_argument("--corr-only", action="store_true", help="drop the clean part of the training set")
    ap.add_argument("--sev-lo", type=int, default=1)
    ap.add_argument("--sev-hi", type=int, default=5)
    ap.add_argument("--pca-fit-n", type=int, default=30000, help="subsample used to fit PCA")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--per-type-eval", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    X_all = np.load(f"{DATA}/x_train.npy")
    y_all = np.load(f"{DATA}/y_train.npy").ravel()
    X_val_img = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")

    order = np.random.default_rng(0).permutation(len(X_all))
    hold_idx = order[: args.n_eval]
    pool_idx = order[args.n_eval : args.n_eval + args.n_clean]

    X_base = X_all[pool_idx]
    y_base = y_all[pool_idx]
    X_hold = X_all[hold_idx]
    y_hold = y_all[hold_idx]

    # ---- training set assembly -------------------------------------------
    parts_x = [] if args.corr_only else [X_base]
    parts_y = [] if args.corr_only else [y_base]
    if args.n_shift:
        idx = rng.choice(len(X_base), args.n_shift)
        parts_x.append(random_flip_shift(X_base[idx], rng))
        parts_y.append(y_base[idx])
    if args.n_corr:
        idx = rng.choice(len(X_base), args.n_corr)
        parts_x.append(make_corrupted(X_base[idx], rng, sev_lo=args.sev_lo, sev_hi=args.sev_hi))
        parts_y.append(y_base[idx])
    if args.n_mild:
        idx = rng.choice(len(X_base), args.n_mild)
        parts_x.append(make_mild(X_base[idx], rng))
        parts_y.append(y_base[idx])
    if not parts_x:
        raise SystemExit("empty training set")
    X_tr_img = np.concatenate(parts_x)
    y_tr = np.concatenate(parts_y)
    del parts_x, parts_y
    print(f"[{args.tag}] train images {X_tr_img.shape} built in {time.time() - t0:.0f}s", flush=True)

    # ---- preprocessing ----------------------------------------------------
    # The scaler / PCA are always fitted on CLEAN images: fitting them on a
    # corrupted mixture makes the components track the corruption instead of
    # the content.
    fit_pool = flatten(X_base)
    fit_idx = rng.choice(len(fit_pool), min(args.pca_fit_n, len(fit_pool)), replace=False)
    fit_pool = fit_pool[fit_idx]
    mean = fit_pool.mean(0)
    std = fit_pool.std(0) + 1e-8
    fit_pool = (fit_pool - mean) / std

    X_tr_raw = flatten(X_tr_img)
    X_tr_raw = (X_tr_raw - mean) / std

    if args.pca:
        t1 = time.time()
        pca = PCA(n_components=args.pca, random_state=args.seed)
        pca.fit(fit_pool)
        X_tr = pca.transform(X_tr_raw).astype(np.float32)
        print(
            f"[{args.tag}] PCA{args.pca} fitted on {len(fit_pool)} clean rows in {time.time() - t1:.0f}s, "
            f"explained var {pca.explained_variance_ratio_.sum():.3f}",
            flush=True,
        )
        del X_tr_raw
    else:
        pca = None
        X_tr = X_tr_raw

    def prepare(images):
        z = (flatten(images) - mean) / std
        return pca.transform(z).astype(np.float32) if pca is not None else z

    # ---- model ------------------------------------------------------------
    clf = MLPClassifier(
        hidden_layer_sizes=(args.hidden,),
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=False,
        random_state=args.seed,
        verbose=False,
    )
    t1 = time.time()
    clf.fit(X_tr, y_tr)
    fit_time = time.time() - t1
    print(f"[{args.tag}] fit {fit_time:.0f}s for {clf.n_iter_} epochs", flush=True)

    # ---- evaluation -------------------------------------------------------
    results = {"tag": args.tag, "config": vars(args), "fit_time_s": round(fit_time), "epochs": int(clf.n_iter_)}
    results["clean_holdout"] = float(accuracy_score(y_hold, clf.predict(prepare(X_hold))))
    results["val_provided"] = float(accuracy_score(y_val, clf.predict(prepare(X_val_img))))

    synth = make_corrupted(X_hold, np.random.default_rng(1234), sev_lo=3, sev_hi=5)
    results["synth_sev3_5"] = float(accuracy_score(y_hold, clf.predict(prepare(synth))))
    synth5 = make_corrupted(X_hold, np.random.default_rng(99), sev_lo=5, sev_hi=5)
    results["synth_sev5"] = float(accuracy_score(y_hold, clf.predict(prepare(synth5))))

    if args.per_type_eval:
        per_type = {}
        for name in CORRUPTION_NAMES:
            c = corrupt_batch(X_hold, name, 5, np.random.default_rng(5))
            per_type[name] = round(float(accuracy_score(y_hold, clf.predict(prepare(c)))), 4)
        results["per_type_sev5"] = per_type
        print(f"[{args.tag}] per-type sev5: {per_type}")

    print(
        f"[{args.tag}] clean={results['clean_holdout']:.4f} val={results['val_provided']:.4f} "
        f"synth3-5={results['synth_sev3_5']:.4f} synth5={results['synth_sev5']:.4f}"
    )
    with open("work/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(results) + "\n")
    print(f"[{args.tag}] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

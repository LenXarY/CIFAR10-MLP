"""Compare feature sets on the same data split / budget (seed 42)."""

import argparse
import json
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, "work")
from feat_lib import build_hog_model, hog_feature_dim  # noqa: E402
from train_pipeline import build_pipeline, make_corrupted, random_flip_shift  # noqa: E402


DATA = "CIFAR10_Data"
VAL = "validation_set_to_students"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True, choices=["raw", "hog", "hog_pca"])
    ap.add_argument("--n-clean", type=int, default=20000)
    ap.add_argument("--n-shift", type=int, default=20000)
    ap.add_argument("--n-corr", type=int, default=0)
    ap.add_argument("--sev-lo", type=int, default=1)
    ap.add_argument("--sev-hi", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--angles", type=int, default=8)
    ap.add_argument("--sigmas", type=str, default="1.2,2.2")
    ap.add_argument("--laplacians", type=str, default="1.6,3.0")
    ap.add_argument("--pyramid", action="store_true")
    ap.add_argument("--raw-union", action="store_true")
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    tag = args.tag or args.feature
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    X_all = np.load(f"{DATA}/x_train.npy")
    y_all = np.load(f"{DATA}/y_train.npy").ravel()
    X_val = np.load(f"{VAL}/images.npy")
    y_val = np.load(f"{VAL}/labels.npy")

    order = np.random.default_rng(0).permutation(len(X_all))
    hold = order[:5000]
    pool = order[5000 : 5000 + args.n_clean]
    X_base, y_base = X_all[pool], y_all[pool]
    X_hold, y_hold = X_all[hold], y_all[hold]

    parts_x, parts_y = [X_base], [y_base]
    if args.n_shift:
        idx = rng.choice(len(X_base), args.n_shift)
        parts_x.append(random_flip_shift(X_base[idx], rng))
        parts_y.append(y_base[idx])
    if args.n_corr:
        idx = rng.choice(len(X_base), args.n_corr)
        parts_x.append(make_corrupted(X_base[idx], rng, args.sev_lo, args.sev_hi))
        parts_y.append(y_base[idx])
    X_tr = np.concatenate(parts_x)
    y_tr = np.concatenate(parts_y)
    print(f"[{tag}] train {X_tr.shape} built in {time.time() - t0:.0f}s", flush=True)

    if args.feature == "raw":
        model = build_pipeline((args.hidden,), args.pca, args.seed, args.epochs)
    else:
        model = build_hog_model(
            cells=args.cells,
            angles=args.angles,
            sigmas=tuple(float(v) for v in args.sigmas.split(",")),
            laplacians=tuple(float(v) for v in args.laplacians.split(",")),
            hidden=(args.hidden,),
            epochs=args.epochs,
            seed=args.seed,
            pca_dim=args.pca if args.feature == "hog_pca" else None,
            pyramid=args.pyramid,
            raw_union=args.raw_union,
            log1p=args.log1p,
            gray=args.gray,
        )
        print(f"[{tag}] feature dim {hog_feature_dim(cells=args.cells, angles=args.angles)}", flush=True)

    t1 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t1
    res = {
        "tag": tag,
        "feature": args.feature,
        "config": vars(args),
        "fit_s": round(fit_s),
        "clean": round(float(accuracy_score(y_hold, model.predict(X_hold))), 4),
        "val": round(float(accuracy_score(y_val, model.predict(X_val))), 4),
    }
    print(f"[{tag}] fit {fit_s:.0f}s -> clean {res['clean']:.4f}  val {res['val']:.4f}", flush=True)
    if args.save:
        joblib.dump(model, args.save)
        import os

        res["model_mb"] = round(os.path.getsize(args.save) / 1e6, 2)
        print(f"[{tag}] saved {args.save} ({res['model_mb']} MB)")
    with open("work/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[{tag}] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

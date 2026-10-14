"""Run one recipe experiment: features -> PCA -> MLP -> validation accuracy."""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_lib import FEATURE_SPECS, build_mixture, feature_dim, features_for, load_raw  # noqa: E402


RESULTS = "work/lab_results.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--variant", default="",
                    help="cache-key suffix; must change whenever the training set changes")
    ap.add_argument("--parts", default="clean,shift,corr13,corr15",
                    help="which mixture components to train on")
    ap.add_argument("--aug", default="",
                    help="comma list of extra augmentation levers: cutout8,jitter,double,sev_hi15=5")
    ap.add_argument("--sizes", default=None,
                    help="mixture sizes as key=value pairs, e.g. clean=16000,corr15=15000")
    ap.add_argument("--cache-tag", default="")
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--hidden", default="768")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.spec
    hidden = tuple(int(v) for v in args.hidden.split(","))

    t0 = time.time()
    X, y, X_val, y_val = load_raw()
    aug = {}
    for item in filter(None, (s.strip() for s in args.aug.split(","))):
        if "=" in item:
            k, v = item.split("=", 1)
            aug[k] = int(v)
        elif item.startswith("cutout"):
            aug["cutout"] = int(item[len("cutout"):] or 8)
        else:
            aug[item] = True
    if args.sizes:
        sizes = {}
        for item in args.sizes.split(","):
            k, v = item.split("=")
            sizes[k.strip()] = int(v)
        sizes.setdefault("clean", 16000)
        sizes.setdefault("shift", 10000)
        sizes.setdefault("shift2", 7000)
        for k in ("corr13", "corr15", "corr35", "corr_layered", "corr_layered15",
                  "rot", "cutout", "jitter", "blur"):
            sizes.setdefault(k, 0)
    else:
        sizes = None
    parts = build_mixture(X, y, aug=aug, sizes=sizes)
    cfg = FEATURE_SPECS[args.spec]
    print(f"[{tag}] feature dim {feature_dim(cfg)} | aug {aug}", flush=True)

    t1 = time.time()
    variant = args.cache_tag or args.variant or (
        args.spec + ("__" + args.aug.replace(",", "_") if args.aug else "")
    )
    feats, F = features_for(cfg, parts, tag=variant)
    print(f"[{tag}] features in {time.time() - t1:.0f}s", flush=True)

    keys = [k for k in args.parts.split(",") if k in F]
    X_tr = np.concatenate([F[k] for k in keys])
    y_tr = np.concatenate([parts[k][1] for k in keys])
    del F
    print(f"[{tag}] train matrix {X_tr.shape} built in {time.time() - t0:.0f}s", flush=True)

    inner = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("pca", PCA(n_components=args.pca, random_state=args.seed)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    alpha=args.alpha,
                    batch_size=args.batch,
                    learning_rate_init=args.lr,
                    max_iter=args.epochs,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    t2 = time.time()
    inner.fit(X_tr, y_tr)
    fit_s = time.time() - t2
    n_iter = inner.named_steps["mlp"].n_iter_

    # the saved object is the same shape as a submission member: the (stateless)
    # feature block in front of the fitted scaler/PCA/MLP head
    model = Pipeline([("features", feats), ("inner", inner)])

    t3 = time.time()
    X_val_f = feats.transform(X_val).astype(np.float32)
    X_val_flip = feats.transform(np.ascontiguousarray(X_val[:, :, ::-1])).astype(np.float32)
    plain = accuracy_score(y_val, inner.predict(X_val_f))
    p0 = inner.predict_proba(X_val_f)
    p1 = inner.predict_proba(X_val_flip)
    tta = accuracy_score(y_val, inner.named_steps["mlp"].classes_[np.argmax(p0 + p1, axis=1)])
    print(
        f"[{tag}] fit {fit_s:.0f}s ({n_iter} epochs) | predict {time.time() - t3:.0f}s | "
        f"plain {plain:.4f}  hflip-TTA {tta:.4f}",
        flush=True,
    )

    res = {
        "tag": tag,
        "spec": args.spec,
        "aug": aug,
        "variant": variant,
        "parts": args.parts,
        "pca": args.pca,
        "hidden": list(hidden),
        "epochs": args.epochs,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_train": int(len(y_tr)),
        "fit_s": round(fit_s),
        "plain": round(float(plain), 4),
        "tta": round(float(tta), 4),
    }
    out = f"work/models/lab_{tag}_seed{args.seed}.joblib"
    os.makedirs("work/models", exist_ok=True)
    joblib.dump(model, out)
    res["model_mb"] = round(os.path.getsize(out) / 1e6, 2)
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[{tag}] total {time.time() - t0:.0f}s | saved {out} ({res['model_mb']} MB)", flush=True)


if __name__ == "__main__":
    main()

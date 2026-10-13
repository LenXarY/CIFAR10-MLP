"""Train and score a full multi-member ensemble exactly like the submission
notebook does, at lab scale, including the horizontal-flip TTA members."""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_lib import (  # noqa: E402
    FEATURE_SPECS,
    _ft,
    build_mixture,
    feature_dim,
    features_for,
    load_raw,
)


MEMBERS = [
    ("clean+shift", ("clean", "shift"), (768,)),
    ("clean", ("clean",), (768,)),
    ("clean+corr13", ("clean", "corr13"), (768,)),
    ("clean+corr15", ("clean", "corr15"), (768,)),
    ("clean+shift2-2layer", ("clean", "shift2"), (512, 512)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--aug", default="")
    ap.add_argument("--pca", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shared-prep", action="store_true",
                    help="fit one StandardScaler+PCA and share it across members (pickles once)")
    ap.add_argument("--bagged", type=int, default=0,
                    help="instead of one member per data mixture, train N members on the "
                         "union of all mixtures, differing only by random seed")
    ap.add_argument("--bag-hidden", default="1024",
                    help="hidden layer sizes for bagged members")
    ap.add_argument("--tta", action="store_true", help="add horizontal-flip twin members")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.spec

    aug = {}
    for item in filter(None, (s.strip() for s in args.aug.split(","))):
        if "=" in item:
            k, v = item.split("=", 1)
            aug[k] = int(v)
        elif item.startswith("cutout"):
            aug["cutout"] = int(item[len("cutout"):] or 8)
        else:
            aug[item] = True

    t0 = time.time()
    X, y, X_val, y_val = load_raw()
    parts = build_mixture(X, y, aug=aug)
    cfg = FEATURE_SPECS[args.spec]
    variant = args.variant or (args.spec + ("__" + args.aug.replace(",", "_") if args.aug else ""))
    feats, F = features_for(cfg, parts, tag=variant)
    print(f"[{tag}] features ready in {time.time() - t0:.0f}s (dim {feature_dim(cfg)})", flush=True)

    prep = None
    if args.shared_prep:
        X_all = np.concatenate([F[k] for k in ("clean", "shift", "shift2", "corr13", "corr15")])
        t1 = time.time()
        prep = Pipeline(
            [
                ("standardize", StandardScaler()),
                ("pca", PCA(n_components=args.pca, random_state=args.seed)),
            ]
        ).fit(X_all)
        del X_all
        print(f"  shared prep fitted in {time.time() - t1:.0f}s", flush=True)

    if args.bagged:
        all_keys = ("clean", "shift", "shift2", "corr13", "corr15")
        bag_hidden = tuple(int(v) for v in args.bag_hidden.split(","))
        specs = [(f"bag{i}", all_keys, bag_hidden, args.seed + i) for i in range(args.bagged)]
    else:
        specs = [(name, keys, hidden, args.seed) for name, keys, hidden in MEMBERS]

    members = []
    for name, keys, hidden, member_seed in specs:
        tm = time.time()
        X_m = np.concatenate([F[k] for k in keys])
        y_m = np.concatenate([parts[k][1] for k in keys])
        head = MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=args.alpha,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=args.epochs,
            random_state=member_seed,
        )
        if args.shared_prep:
            inner = Pipeline([("prep", prep), ("mlp", head)]).fit(X_m, y_m)
        else:
            inner = Pipeline(
                [
                    ("standardize", StandardScaler()),
                    ("pca", PCA(n_components=args.pca, random_state=member_seed)),
                    ("mlp", head),
                ]
            ).fit(X_m, y_m)
        members.append(Pipeline([("features", feats), ("inner", inner)]))
        print(
            f"  member {name:22s} {len(y_m):6d} samples, {inner.named_steps['mlp'].n_iter_} epochs, "
            f"{time.time() - tm:.0f}s",
            flush=True,
        )
        del X_m, y_m

    all_members = list(members)
    if args.tta:
        flip = Pipeline([("flip", _ft(np.flip, axis=2)), ("features", feats)])
        for m in members:
            all_members.append(Pipeline([("flip", flip), ("inner", m.named_steps["inner"])]))

    ensemble = VotingClassifier([(f"m{i}", m) for i, m in enumerate(all_members)], voting="soft")
    ensemble.estimators_ = all_members
    ensemble.le_ = LabelEncoder().fit(y_val)
    ensemble.classes_ = ensemble.le_.classes_

    t2 = time.time()
    acc = accuracy_score(y_val, ensemble.predict(X_val))
    print(f"[{tag}] ensemble ({len(all_members)} members) val {acc:.4f} | predict {time.time() - t2:.0f}s", flush=True)

    plain = VotingClassifier([(f"m{i}", m) for i, m in enumerate(members)], voting="soft")
    plain.estimators_ = members
    plain.le_ = ensemble.le_
    plain.classes_ = ensemble.classes_
    acc_plain = accuracy_score(y_val, plain.predict(X_val))
    print(f"[{tag}] without TTA: {acc_plain:.4f}  (TTA gain {acc - acc_plain:+.4f})", flush=True)

    out = f"work/models/lab_ens_{tag}_seed{args.seed}.joblib"
    os.makedirs("work/models", exist_ok=True)
    joblib.dump(ensemble, out)
    mb = os.path.getsize(out) / 1e6
    res = {
        "tag": tag, "spec": args.spec, "aug": aug, "pca": args.pca, "epochs": args.epochs,
        "seed": args.seed, "shared_prep": args.shared_prep, "tta": args.tta,
        "members": len(all_members), "val_no_tta": round(float(acc_plain), 4),
        "val": round(float(acc), 4), "model_mb": round(mb, 2), "total_s": round(time.time() - t0),
    }
    with open("work/lab_results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[{tag}] saved {out} ({mb:.1f} MB) | total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

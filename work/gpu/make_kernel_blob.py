"""Pack the learned convolution kernels into a base64 blob small enough to sit
inside the notebook, so the submission ZIP stays notebook + three model files."""

import argparse
import base64
import io
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="one or more exported .npz")
    ap.add_argument("--out", default="work/gpu/out/kernel_blob.txt")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    payload = {}
    for idx, path in enumerate(args.npz):
        z = np.load(path)
        for k in z.files:
            arr = z[k]
            if k.startswith("convs_") or k.startswith("args_") or k == "val_acc":
                key = k if len(args.npz) == 1 else f"m{idx}_{k}"
                payload[key] = arr.astype(np.dtype(args.dtype)) if k.startswith("convs_") else arr

    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("ascii")
    with open(args.out, "w", encoding="ascii") as fh:
        fh.write(b64)
    print(f"{len(payload)} arrays, {len(b64)} base64 chars ({len(b64) / 1e6:.2f} MB) -> {args.out}")


if __name__ == "__main__":
    main()

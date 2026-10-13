"""Copy the trained conv stack out of the torch checkpoint as plain NumPy
arrays, so the submission-side code never needs torch."""

import sys

import numpy as np
import torch


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "work/gpu/out/bake_pool.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "work/gpu/out/baked_weights.npz"
    ckpt = torch.load(src, weights_only=False)
    out = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("convs."):
            out[k.replace(".", "_")] = v.cpu().numpy()
    out["args_widths"] = np.array([int(v) for v in ckpt["args"]["widths"].split(",")])
    out["args_strides"] = np.array([int(v) for v in ckpt["args"]["strides"].split(",")])
    out["args_pool"] = np.array([1 if ckpt["args"].get("pool") else 0])
    out["val_acc"] = np.array([ckpt["val_acc"]])
    np.savez_compressed(dst, **out)
    print(f"exported {len(out)} arrays -> {dst}")
    print(f"  widths {out['args_widths']} strides {out['args_strides']} pool {out['args_pool'][0]}")


if __name__ == "__main__":
    main()

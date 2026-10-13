"""Check that the baked NumPy pipeline reproduces the trained CNN exactly."""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake import apply_steps, bake  # noqa: E402
from bake_net import BakeNet  # noqa: E402
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, load_data  # noqa: E402


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "work/gpu/out/bake_pool.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ckpt = torch.load(ckpt_path, weights_only=False)
    a = ckpt["args"]
    widths = tuple(int(v) for v in a["widths"].split(","))
    strides = tuple(int(v) for v in a["strides"].split(","))

    model = BakeNet(widths, strides, pool=a.get("pool", False))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    _, _, Xv, yv = load_data()
    X = Xv[:n]
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(device)
    with torch.no_grad():
        x = torch.from_numpy(X).permute(0, 3, 1, 2).float().to(device)
        x = (x / 255.0 - mean) / std
        ref = model(x, features_only=True).cpu().numpy()

    steps = bake(ckpt["state_dict"], widths, pool=a.get("pool", False),
                 resolutions=(32, 32 // 2, 32 // 4, 32 // 8)[: len(widths)],
                 norm=(CIFAR_MEAN, CIFAR_STD))
    t0 = time.time()
    got = apply_steps(steps, X)
    print(f"baked transform of {n} images in {time.time() - t0:.1f}s -> {got.shape}")
    print(f"torch reference shape {ref.shape}")
    diff = np.abs(got - ref)
    scale = np.abs(ref).mean() + 1e-6
    print(f"max abs diff {diff.max():.3e}  mean abs diff {diff.mean():.3e}  "
          f"(feature scale {scale:.3f}, max relative {diff.max() / scale:.2e})")

    # accuracy with the model's own linear head on the baked features (sanity)
    W = ckpt["state_dict"]["fc.weight"].cpu().numpy()
    b = ckpt["state_dict"]["fc.bias"].cpu().numpy()
    pred_baked = (got @ W.T + b).argmax(1)
    pred_ref = (ref @ W.T + b).argmax(1)
    same = (pred_baked == pred_ref).mean()
    print(f"baked vs torch predicted labels agree on {same:.4%} of the {n} images")
    print(f"linear-head accuracy: baked {(pred_baked == yv[:n]).mean():.4f}  "
          f"torch {(pred_ref == yv[:n]).mean():.4f}")
    print("MATCH" if same > 0.999 else "MISMATCH")


if __name__ == "__main__":
    main()

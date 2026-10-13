"""CNN trained with CIFAR-10-C style corruption augmentation.

The validation set is dominated by corrupted images (a plain CNN reaches 85% on
clean CIFAR but only 52% here), so the probe has to be trained under the same
distribution the model will be scored on. This measures the ceiling for a
"convolutional features + classifier" solution.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, Net, load_data  # noqa: E402


def gaussian_kernel(size, sigma, device):
    a = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    k = torch.exp(-(a**2) / (2 * sigma**2))
    k = k / k.sum()
    return (k[:, None] * k[None, :]).view(1, 1, size, size)


def corrupt(x, gen):
    """Per-sample random corruption in normalised space; mirrors the CIFAR-10-C mix."""
    n = x.shape[0]
    dev = x.device
    r = lambda: torch.rand(n, generator=gen, device=dev)  # noqa: E731
    out = x.clone()

    m = r() < 0.35  # additive noise
    if m.any():
        s = (0.05 + torch.rand(n, generator=gen, device=dev) * 0.35)[m].view(-1, 1, 1, 1)
        out[m] = out[m] + torch.randn_like(out[m]) * s

    m = r() < 0.35  # brightness
    if m.any():
        d = (torch.rand(n, generator=gen, device=dev) * 2.0 - 1.0)[m].view(-1, 1, 1, 1)
        out[m] = out[m] + d

    m = r() < 0.35  # contrast
    if m.any():
        f = (0.25 + torch.rand(n, generator=gen, device=dev) * 1.3)[m].view(-1, 1, 1, 1)
        mu = out[m].mean((1, 2, 3), keepdim=True)
        out[m] = (out[m] - mu) * f + mu

    m = r() < 0.30  # blur (gaussian or defocus-ish box)
    if m.any():
        sel = out[m]
        idx = torch.nonzero(m).ravel()
        half = len(sel) // 2
        if half:
            k = gaussian_kernel(5, 1.2, dev).repeat(3, 1, 1, 1)
            sel[:half] = F.conv2d(sel[:half], k, padding=2, groups=3)
        if len(sel) - half:
            sel[half:] = F.avg_pool2d(sel[half:], 3, stride=1, padding=1)
        out[idx] = sel

    m = r() < 0.20  # saturation
    if m.any():
        sel = out[m]
        gray = (sel * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        f = (0.2 + torch.rand(len(sel), generator=gen, device=dev) * 1.6).view(-1, 1, 1, 1)
        out[torch.nonzero(m).ravel()] = gray + (sel - gray) * f

    m = r() < 0.20  # pixelate
    if m.any():
        sel = out[m]
        s = torch.randint(8, 17, (len(sel),), generator=gen, device=dev)
        small = F.interpolate(sel, size=int(s.min()), mode="nearest")
        sel = F.interpolate(small, size=32, mode="nearest")
        out[torch.nonzero(m).ravel()] = sel

    m = r() < 0.20  # cutout
    if m.any():
        sel = out[m].clone()
        for i in range(len(sel)):
            sz = int(torch.randint(6, 14, (1,), generator=gen, device=dev))
            y = int(torch.randint(0, 33 - sz, (1,), generator=gen, device=dev))
            xx = int(torch.randint(0, 33 - sz, (1,), generator=gen, device=dev))
            sel[i, :, y : y + sz, xx : xx + sz] = 0.0
        out[torch.nonzero(m).ravel()] = sel

    return out.clamp(-3.0, 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--tag", default="cnn_aug")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    X, y, Xv, yv = load_data()
    Xh, yh = X[-args.holdout :], y[-args.holdout :]
    X, y = X[: -args.holdout], y[: -args.holdout]

    xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y.astype(np.int64))
    xv = torch.from_numpy(Xv).permute(0, 3, 1, 2).contiguous()
    xh = torch.from_numpy(Xh).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(device)

    def to_float(x):
        return (x.float() / 255.0 - mean) / std

    model = Net(args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    steps = args.epochs * ((len(xt) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def evaluate(t):
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(t), 1000):
                out.append(model(to_float(t[i : i + 1000].to(device))).argmax(1).cpu())
        return torch.cat(out).numpy()

    t0 = time.time()
    acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        perm = torch.randperm(len(xt), generator=torch.Generator().manual_seed(args.seed + epoch))
        for i in range(0, len(xt), args.bs):
            idx = perm[i : i + args.bs]
            xb = to_float(xt[idx].to(device))
            xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
            crops = torch.empty((len(idx), 3, 32, 32), device=device)
            for k in range(len(idx)):
                dy, dx = torch.randint(0, 9, (2,), generator=gen, device=device).tolist()
                crops[k] = xb[k, :, dy : dy + 32, dx : dx + 32]
            flip = torch.rand(len(idx), generator=gen, device=device) < 0.5
            crops[flip] = crops[flip].flip(-1)
            xb = corrupt(crops, gen)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.05)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            acc = (evaluate(xv) == yv).mean()
            clean = (evaluate(xh) == yh).mean()
            print(
                f"epoch {epoch + 1:3d}  loss {total / len(xt):.4f}  "
                f"val {acc:.4f}  clean-holdout {clean:.4f}  {time.time() - t0:.0f}s",
                flush=True,
            )

    os.makedirs("work/gpu/out", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "val_acc": float(acc)},
               f"work/gpu/out/{args.tag}.pt")
    with open("work/gpu/out/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tag": args.tag, **vars(args), "val": round(float(acc), 4),
                             "seconds": round(time.time() - t0)}) + "\n")
    print(f"final val {acc:.4f} | saved work/gpu/out/{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()

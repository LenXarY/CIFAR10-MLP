"""A CNN whose topology is deliberately chosen so that its convolution stack can
be baked into the scikit-learn pipeline as constant kernels.

Per stage:  depthwise 3x3 -> ReLU -> maxpool -> pointwise 1x1 -> ReLU

Why this order: at prediction time the pipeline cannot chunk, so the largest
intermediate it materialises must stay small. Baking a *pointwise* layer is a
plain matrix product whose working set is (n_pixels, C_in); baking a *depthwise*
layer is one independent 2-D convolution per channel, and doing the pooling
before the pointwise step keeps every intermediate in the hundreds of megabytes
rather than gigabytes. A dense 3x3x64->64 convolution would need an im2col
buffer of ~3 GB for 5,000 images; this topology peaks around 300 MB.
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
from cnn_aug import corrupt  # noqa: E402
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, load_data  # noqa: E402


class SepNet(nn.Module):
    def __init__(self, widths=(32, 64, 128), kernel=3):
        super().__init__()
        self.widths = widths
        self.kernel = kernel
        self.blocks = nn.ModuleList()
        cin = 3
        for w in widths:
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "dw": nn.Conv2d(cin, cin, kernel, padding=kernel // 2, groups=cin, bias=False),
                        "dwbn": nn.BatchNorm2d(cin),
                        "pw": nn.Conv2d(cin, w, 1, bias=False),
                        "pwbn": nn.BatchNorm2d(w),
                    }
                )
            )
            cin = w
        self.fc = nn.Linear(cin, 10)
        self.channels = cin

    def forward(self, x, features_only=False):
        for blk in self.blocks:
            x = F.relu(blk["dwbn"](blk["dw"](x)), inplace=True)
            x = F.max_pool2d(x, 2)
            x = F.relu(blk["pwbn"](blk["pw"](x)), inplace=True)
        x = x.mean((2, 3))
        if features_only:
            return x
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--widths", default="32,64,128")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=5000)
    ap.add_argument("--tag", default="sepnet")
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

    widths = tuple(int(v) for v in args.widths.split(","))
    model = SepNet(widths, args.kernel).to(device)
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
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            acc = (evaluate(xv) == yv).mean()
            clean = (evaluate(xh) == yh).mean()
            print(
                f"epoch {epoch + 1:3d}  loss {total / len(xt):.4f}  val {acc:.4f}  "
                f"clean-holdout {clean:.4f}  {time.time() - t0:.0f}s",
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

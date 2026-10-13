"""GPU probe: how much accuracy is reachable on this exact validation set?

This is an *experiment only*. The submission has to stay pure scikit-learn, so
the point of the probe is to (a) measure the headroom above the current 63.6%
and (b) produce a convolutional filter stack that can later be frozen and baked
into the scikit-learn feature block as plain constant kernels.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], np.float32)


def load_data():
    X = np.load("CIFAR10_Data/x_train.npy")
    y = np.load("CIFAR10_Data/y_train.npy").ravel()
    Xv = np.load("validation_set_to_students/images.npy")
    yv = np.load("validation_set_to_students/labels.npy")
    return X, y, Xv, yv


class Net(nn.Module):
    """conv-pool x3 -> global average pool -> linear. The conv stack is what we
    would later freeze and bake into the scikit-learn pipeline."""

    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = [nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        ch = width
        for i in range(depth - 1):
            out = width * (2 if i == depth - 2 else 1)
            layers += [
                nn.Conv2d(ch, out, 3, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            ch = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(ch, 10)
        self.channels = ch
        self.depth = depth

    def forward(self, x):
        n_pools = 3
        for i in range(self.depth):
            x = self.features[i * 3 : i * 3 + 3](x)
            if i < n_pools:
                x = self.pool(x)
        x = x.mean((2, 3))
        return self.fc(x)


def batches(n, bs, shuffle=True, generator=None):
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def augment(x, pad=4):
    """Random crop with reflection padding, horizontal flip, cutout."""
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    n = x.shape[0]
    out = torch.empty((n, 3, 32, 32), device=x.device, dtype=x.dtype)
    for i in range(n):
        dy, dx = torch.randint(0, 2 * pad + 1, (2,)).tolist()
        out[i] = x[i, :, dy : dy + 32, dx : dx + 32]
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def cutout(x, size=8):
    n = x.shape[0]
    ys = torch.randint(0, 32 - size + 1, (n,), device=x.device)
    xs = torch.randint(0, 32 - size + 1, (n,), device=x.device)
    for i in range(n):
        x[i, :, ys[i] : ys[i] + size, xs[i] : xs[i] + size] = 0.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="0 = all 50k images")
    ap.add_argument("--holdout", type=int, default=0,
                    help="hold these training images out as a *clean* test set")
    ap.add_argument("--tag", default="cnn")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)
    torch.manual_seed(args.seed)

    X, y, Xv, yv = load_data()
    Xh, yh = None, None
    if args.holdout:
        Xh, yh = X[-args.holdout :], y[-args.holdout :]
        X, y = X[: -args.holdout], y[: -args.holdout]
    if args.subset:
        X, y = X[: args.subset], y[: args.subset]
    xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y.astype(np.int64))
    xv = torch.from_numpy(Xv).permute(0, 3, 1, 2).contiguous()

    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(device)

    def to_float(x):
        return (x.float() / 255.0 - mean) / std

    model = Net(args.width, args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * ((len(xt) + args.bs - 1) // args.bs)
    )
    gen = torch.Generator().manual_seed(args.seed)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for idx in batches(len(xt), args.bs, generator=gen):
            xb = to_float(xt[idx].to(device))
            xb = cutout(augment(xb))
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(xv), 500):
                    preds.append(model(to_float(xv[i : i + 500].to(device))).argmax(1).cpu())
                acc = (torch.cat(preds).numpy() == yv).mean()
                msg = ""
                if Xh is not None:
                    xht = torch.from_numpy(Xh).permute(0, 3, 1, 2).contiguous()
                    hp = []
                    for i in range(0, len(xht), 500):
                        hp.append(model(to_float(xht[i : i + 500].to(device))).argmax(1).cpu())
                    acc_clean = (torch.cat(hp).numpy() == yh).mean()
                    msg = f"  clean-holdout {acc_clean:.4f}"
            print(
                f"epoch {epoch + 1:3d}  loss {total / len(xt):.4f}  val {acc:.4f}  "
                f"{time.time() - t0:.0f}s{msg}",
                flush=True,
            )

    os.makedirs("work/gpu/out", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "args": vars(args), "val_acc": float(acc)},
        f"work/gpu/out/{args.tag}.pt",
    )
    with open("work/gpu/out/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tag": args.tag, **vars(args), "val": round(float(acc), 4),
                             "seconds": round(time.time() - t0)}) + "\n")
    print(f"final val {acc:.4f} | saved work/gpu/out/{args.tag}.pt", flush=True)


if __name__ == "__main__":
    main()

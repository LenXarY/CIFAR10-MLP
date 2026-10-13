"""Where is the CNN losing accuracy: clean vs corrupted, and train vs val?"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "work/gpu")
from cnn_ceiling import CIFAR_MEAN, CIFAR_STD, Net  # noqa: E402


def main():
    ckpt = torch.load("work/gpu/out/cnn_w64_d3.pt", weights_only=False)
    args = ckpt["args"]
    model = Net(args["width"], args["depth"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(device)

    def predict(X):
        x = torch.from_numpy(X).permute(0, 3, 1, 2).float().to(device)
        x = (x / 255.0 - mean) / std
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 1000):
                out.append(model(x[i : i + 1000]).argmax(1).cpu().numpy())
        return np.concatenate(out)

    X = np.load("CIFAR10_Data/x_train.npy")
    y = np.load("CIFAR10_Data/y_train.npy").ravel()
    Xv = np.load("validation_set_to_students/images.npy")
    yv = np.load("validation_set_to_students/labels.npy")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), 5000, replace=False)
    print(f"train images  (in-sample)  : {(predict(X[idx]) == y[idx]).mean():.4f}")

    pv = predict(Xv)
    print(f"validation set (all)       : {(pv == yv).mean():.4f}")

    xf = Xv.astype(np.float32)
    black = (Xv.max(3) == 0).mean((1, 2))
    gray = xf.mean(3)
    d = np.abs(np.diff(gray, axis=2))
    blocky = d[:, :, 7::8][:, :, :3].mean((1, 2)) / (d.mean((1, 2)) + 1e-6)
    std_img = gray.reshape(len(gray), -1).std(1)

    xt = X.astype(np.float32)
    itr = np.random.default_rng(1).choice(len(X), 20000, replace=False)
    g = xt[itr].mean(3)
    dt = np.abs(np.diff(g, axis=2))
    tb = np.percentile(dt[:, :, 7::8][:, :, :3].mean((1, 2)) / (dt.mean((1, 2)) + 1e-6), 99.5)
    ts = np.percentile(g.reshape(len(g), -1).std(1), 1.0)
    flag = (black > 0.05) | (blocky > tb) | (std_img < ts)
    print(f"  clean-like   ({1 - flag.mean():5.1%})   : {(pv[~flag] == yv[~flag]).mean():.4f}")
    print(f"  corrupted    ({flag.mean():5.1%})   : {(pv[flag] == yv[flag]).mean():.4f}")


if __name__ == "__main__":
    main()

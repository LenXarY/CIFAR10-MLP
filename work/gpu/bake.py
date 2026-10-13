"""Bake a trained CNN's convolution stack into a plain NumPy/scikit-learn pipeline.

Each convolution becomes:  zero-pad -> append a constant-1 channel -> im2col ->
matrix product -> ReLU (-> max-pool). The constant channel carries the bias: its
kernel is b/9, so summing the nine taps adds b exactly once per output position.
That keeps the whole thing expressible as module-level NumPy callables, which is
what the submission requires -- nothing custom ends up in the pickle.
"""

import numpy as np


def _ft(func, **kw):
    """A step descriptor: just the module-level callable plus its constant kwargs."""
    return (func, kw or None)


def reshape(shape):
    return _ft(np.reshape, shape=shape)


def fold_bn(conv_w, bn):
    """Fold a BatchNorm into the preceding convolution."""
    gamma = np.asarray(bn["weight"], np.float64)
    beta = np.asarray(bn["bias"], np.float64)
    mean = np.asarray(bn["running_mean"], np.float64)
    var = np.asarray(bn["running_var"], np.float64)
    eps = float(bn["eps"])
    s = gamma / np.sqrt(var + eps)
    w = np.asarray(conv_w, np.float64) * s[:, None, None, None]
    b = beta - mean * s
    return w.astype(np.float32), b.astype(np.float32)


def conv_relu_steps(w, b, hw, pool=False):
    """One conv+BN+ReLU(+maxpool) layer, w shaped (C_out, C_in, 3, 3), NHWC in/out."""
    c_out, c_in = w.shape[0], w.shape[1]
    wmat = np.zeros((c_in * 9 + 9, c_out), np.float32)
    wmat[: c_in * 9] = w.transpose(1, 2, 3, 0).reshape(c_in * 9, c_out)
    wmat[c_in * 9 :] = (b / 9.0)[None, :]

    steps = [
        ("pad_hw", _ft(np.pad, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))),
        ("ones", _ft(np.pad, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)),
        ("win", _ft(np.lib.stride_tricks.sliding_window_view, window_shape=(3, 3), axis=(1, 2))),
        ("cols", reshape((-1, (c_in + 1) * 9))),
        ("dot", _ft(np.dot, b=wmat)),
        ("relu", _ft(np.clip, a_min=0.0, a_max=None)),
        ("out", reshape((-1, hw, hw, c_out))),
    ]
    if pool:
        h = hw // 2
        steps += [
            ("pool_r", reshape((-1, h, 2, h, 2, c_out))),
            ("pool_m", _ft(np.max, axis=(2, 4))),
        ]
    return steps


def bake(state_dict, widths, pool=True, resolutions=(32, 16, 8, 4), norm=None):
    """state_dict comes from a BakeNet; returns a list of numpy step descriptors."""
    def get(name):
        return state_dict[name if name in state_dict else name.replace(".", "_")]

    steps = [("as_float32", _ft(np.asarray, dtype=np.float32))]
    if norm is not None:
        mean, std = norm
        c = len(mean)
        wmat = np.zeros((c + 1, c), np.float32)
        wmat[:c] = np.diag((1.0 / (255.0 * np.asarray(std, np.float64))).astype(np.float32))
        wmat[c] = (-np.asarray(mean, np.float64) / np.asarray(std, np.float64)).astype(np.float32)
        res0 = resolutions[0]
        steps += [
            ("n_ones", _ft(np.pad, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)),
            ("n_win", _ft(np.lib.stride_tricks.sliding_window_view, window_shape=(1, 1), axis=(1, 2))),
            ("n_cols", reshape((-1, c + 1))),
            ("n_dot", _ft(np.dot, b=wmat)),
            ("n_out", reshape((-1, res0, res0, c))),
        ]
    for i, (c_out, res) in enumerate(zip(widths, resolutions)):
        c_in = 3 if i == 0 else widths[i - 1]
        conv_w = np.asarray(get(f"convs.{i}.0.weight"))
        bn = {
            "weight": np.asarray(get(f"convs.{i}.1.weight")),
            "bias": np.asarray(get(f"convs.{i}.1.bias")),
            "running_mean": np.asarray(get(f"convs.{i}.1.running_mean")),
            "running_var": np.asarray(get(f"convs.{i}.1.running_var")),
            "eps": 1e-5,
        }
        w, b = fold_bn(conv_w, bn)
        last = i == len(widths) - 1
        steps += conv_relu_steps(w, b, res, pool=pool and not last)
    steps.append(("gap", _ft(np.mean, axis=(1, 2))))
    return steps


def apply_steps(steps, X):
    """Run the descriptor list with plain NumPy (used by the GPU-side verifier)."""
    for _, (func, kw) in steps:
        X = func(X, **(kw or {}))
    return X


def to_pipeline(steps):
    """Convert the descriptors into a scikit-learn Pipeline (submission side)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline([
        (name, FunctionTransformer(func, kw_args=kw, validate=False))
        for name, (func, kw) in steps
    ])

import time

import numpy as np

from mlp import MLP, OPTIMIZERS


def k_rel_errors(theta_hat, theta):
    k_hat, k = np.exp(theta_hat), np.exp(theta)
    return np.linalg.norm(k_hat - k, axis=1) / np.linalg.norm(k, axis=1)


def fit(sizes, optimizer, lr, batch, seed, Ztr, Ttr, Zval, Tval, epochs=100):
    net = MLP(sizes, seed=seed)
    opt = OPTIMIZERS[optimizer](net.params, lr)
    rng = np.random.default_rng([seed, 1])
    n = len(Ztr)

    hist = {"train": [], "val": []}
    best = {"val": np.inf, "epoch": 0, "params": net.copy_params()}
    diverged = False
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            _, grads = net.loss_and_grads(Ztr[idx], Ttr[idx])
            opt.step(grads)
        with np.errstate(over="ignore", invalid="ignore"):
            ltr = net.loss(net.predict(Ztr), Ttr)
            lval = net.loss(net.predict(Zval), Tval)
        hist["train"].append(ltr)
        hist["val"].append(lval)
        if not (np.isfinite(ltr) and np.isfinite(lval)):
            diverged = True
            break
        if lval < best["val"]:
            best = {"val": lval, "epoch": epoch, "params": net.copy_params()}
    seconds = time.perf_counter() - t0

    net.set_params(best["params"])
    theta_val = net.predict(Zval)
    return {
        "net": net,
        "history": hist,
        "best_epoch": best["epoch"],
        "val_loss": best["val"],
        "val_rmse_theta": float(np.sqrt(np.mean((theta_val - Tval) ** 2))),
        "val_k_median": float(np.median(k_rel_errors(theta_val, Tval))),
        "diverged": diverged,
        "seconds": seconds,
    }

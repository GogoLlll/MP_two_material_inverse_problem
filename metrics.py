import numpy as np

import config as C
from solver import solve

K_SANE = (1e-8, 1e8)

def is_failure(theta_hat):
    with np.errstate(over="ignore", invalid="ignore"):
        k_hat = np.exp(theta_hat)
    bad = ~np.isfinite(theta_hat).all(axis=1) | ~np.isfinite(k_hat).all(axis=1)
    bad |= (k_hat < K_SANE[0]).any(axis=1) | (k_hat > K_SANE[1]).any(axis=1)
    return bad

def k_errors(theta_hat, theta):
    fail = is_failure(theta_hat)
    e = np.full(len(theta), np.inf)
    ok = ~fail
    k_hat, k = np.exp(theta_hat[ok]), np.exp(theta[ok])
    e[ok] = np.linalg.norm(k_hat - k, axis=1) / np.linalg.norm(k, axis=1)
    return e, fail

def true_profiles(theta, N=C.N_TEST):
    return np.array([solve(*np.exp(t), N) for t in theta])

def profile_errors(theta_hat, U_true, N=C.N_TEST):
    fail = is_failure(theta_hat)
    e = np.full(len(theta_hat), np.inf)
    for i in np.flatnonzero(~fail):
        with np.errstate(all="ignore"):
            u = solve(*np.exp(theta_hat[i]), N)
        if not np.isfinite(u).all():
            fail[i] = True
            continue
        rel = np.linalg.norm(u - U_true[i], axis=1) / np.linalg.norm(U_true[i], axis=1)
        e[i] = rel.mean()
    return e, fail

def summarize(theta_hat, theta, U_true):
    ek, fail_k = k_errors(theta_hat, theta)
    eu, fail_u = profile_errors(theta_hat, U_true)
    fail = fail_k | fail_u
    ok = ~fail
    rmse = float(np.sqrt(np.mean((theta_hat[ok] - theta[ok]) ** 2))) if ok.any() else np.inf
    return {
        "k_med": float(np.median(ek)), "k_p90": float(np.percentile(ek, 90)),
        "u_med": float(np.median(eu)), "u_p90": float(np.percentile(eu, 90)),
        "rmse_theta": rmse, "failures": int(fail.sum()), "n": len(theta),
    }

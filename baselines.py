import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

import config as C
from solver import measure

class NearestNeighbor:
    def __init__(self, Z_train, theta_train):
        self.tree = cKDTree(Z_train)
        self.theta = theta_train

    def predict(self, Z):
        _, idx = self.tree.query(Z, k=1)
        return self.theta[idx]

def nls_one(z_obs, m, d, N=C.N_TEST, tol=1e-12, cols=slice(0, 6)):
    def residual(th):
        return ((measure(*np.exp(th), N) - m) / d)[cols] - z_obs

    t0 = time.perf_counter()
    try:
        r = least_squares(residual, x0=np.zeros(2),
                          bounds=([C.LNK_OOD_LO] * 2, [C.LNK_OOD_HI] * 2),
                          method="trf", ftol=tol, xtol=tol, gtol=tol, max_nfev=200)
        theta_hat = r.x
        info = {"status": int(r.status), "nfev": int(r.nfev), "cost": float(r.cost),
                "at_bound": bool(np.any(r.active_mask != 0)), "error": ""}
    except Exception as exc:                       # явная запись срыва расчёта
        theta_hat = np.full(2, np.nan)
        info = {"status": -99, "nfev": 0, "cost": np.nan, "at_bound": False, "error": repr(exc)}
    info["seconds"] = time.perf_counter() - t0
    return theta_hat, info

def nls(Z_obs, m, d, N=C.N_TEST):
    out = [nls_one(z, m, d, N) for z in Z_obs]
    return np.array([o[0] for o in out]), [o[1] for o in out]

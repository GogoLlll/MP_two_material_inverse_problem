import numpy as np

import config as C
from solver import measure

RANK_TOL = 1e-10

def forward_z(theta, m, d, N=C.N_TEST):
    return (measure(*np.exp(theta), N) - m) / d

def jacobian(theta, m, d, N=C.N_TEST, h=1e-5):
    J = np.empty((6, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (forward_z(theta + e, m, d, N) - forward_z(theta - e, m, d, N)) / (2 * h)
    return J

def svd_stats(J):
    s = np.linalg.svd(J, compute_uv=False)
    smax, smin = s[0], s[-1]
    full_rank = smin / smax > RANK_TOL
    return {
        "smax": smax, "smin": smin, "full_rank": bool(full_rank),
        "kappa": smax / smin if full_rank else np.inf,
        "pinv_F": np.sqrt(smax**-2 + smin**-2) if full_rank else np.inf,
    }

def net_jacobian(net, z, h=1e-5):
    n_in = z.size
    M = np.empty((2, n_in))
    for i in range(n_in):
        e = np.zeros(n_in)
        e[i] = h
        M[:, i] = (net.predict((z + e)[None]) - net.predict((z - e)[None]))[0] / (2 * h)
    return M

def left_inverse_split(M, J):
    Q, _ = np.linalg.qr(J)
    P = Q @ Q.T
    MP, MPp = M @ P, M @ (np.eye(J.shape[0]) - P)
    return {
        "M_F": np.linalg.norm(M), "M_range_F": np.linalg.norm(MP), "M_perp_F": np.linalg.norm(MPp),
        "MJ_minus_I": np.linalg.norm(M @ J - np.eye(2), 2),
    }

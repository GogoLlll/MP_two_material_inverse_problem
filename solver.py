import numpy as np
from scipy.linalg import solve_banded

from config import A, X_MEAS


def f1(x):
    return np.ones_like(x)


def f2(x):
    return 1.0 + 4.0 * x**2


LOADS = (f1, f2)

def edge_coefficients(k1, k2, N, a=A):
    h = 1.0 / N
    left = np.arange(N) * h
    right = left + h
    kk = np.where(right <= a, k1, k2).astype(float)
    cut = (left < a) & (right > a)                 # единственное ребро, содержащее a
    l1 = a - left[cut]
    l2 = right[cut] - a
    kk[cut] = h / (l1 / k1 + l2 / k2)
    return kk

def solve(k1, k2, N, loads=LOADS, a=A):
    h = 1.0 / N
    x = np.arange(N + 1) * h
    kk = edge_coefficients(k1, k2, N, a) / h**2
    n = N - 1                                      # внутренние неизвестные u_1..u_{N-1}
    ab = np.zeros((3, n))
    ab[0, 1:] = -kk[1:n]                           # наддиагональ: A[r, r+1] = -k_{r+3/2}
    ab[1, :] = kk[0:n] + kk[1:n + 1]               # диагональ: (k_{i-1/2} + k_{i+1/2})/h^2
    ab[2, :-1] = -kk[1:n]                          # поддиагональ: A[r+1, r] = -k_{r+3/2}
    rhs = np.stack([f(x[1:N]) for f in loads], axis=1)
    u = np.zeros((len(loads), N + 1))
    u[:, 1:N] = solve_banded((1, 1), ab, rhs).T
    return u


def measurement_indices(N):
    idx = [round(xm * N) for xm in X_MEAS]
    for i, xm in zip(idx, X_MEAS):
        assert abs(i / N - xm) < 1e-14, f"x = {xm} не является узлом сетки N = {N}"
    return idx

def measure(k1, k2, N, a=A):
    u = solve(k1, k2, N, a=a)
    idx = measurement_indices(N)
    return np.concatenate([u[0, idx], u[1, idx]])

_G = (
    lambda x: x**2 / 2,                    # f1 = 1        -> F = x,             G = x^2/2
    lambda x: x**2 / 2 + x**4 / 3,         # f2 = 1 + 4x^2 -> F = x + 4x^3/3,    G = x^2/2 + x^4/3
)

def exact(x, k1, k2, load, a=A):
    G = _G[load]
    x = np.asarray(x, dtype=float)
    C = (G(a) / k1 + (G(1.0) - G(a)) / k2) / (a / k1 + (1.0 - a) / k2)
    left = (C * x - G(x)) / k1
    right = (C * a - G(a)) / k1 + (C * (x - a) - (G(x) - G(a))) / k2
    return np.where(x < a, left, right)

def exact_measure(k1, k2, a=A):
    xm = np.array(X_MEAS)
    return np.concatenate([exact(xm, k1, k2, 0, a), exact(xm, k1, k2, 1, a)])

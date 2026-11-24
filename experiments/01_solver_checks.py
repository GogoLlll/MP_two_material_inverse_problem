import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import A, S, LNK_LO, LNK_HI, SEED_SOLVER_CHECK, RESULTS_DIR, X_MEAS
from solver import solve, measure, exact, exact_measure, edge_coefficients

GRIDS = (100, 200, 400)
out = RESULTS_DIR / "01_solver"
out.mkdir(parents=True, exist_ok=True)
lines = []


def log(s=""):
    print(s)
    lines.append(s)


log("## Положение границы материалов\n")
log("| N | a·N | расстояние от a до ближайшего узла |")
for N in GRIDS:
    dist = abs(A * N - round(A * N)) / N
    assert dist > 1e-12
    cut_edge = int(np.floor(A * N))
    log(f"| {N} | {A * N:.2f} | {dist:.4g} (a внутри ребра [{cut_edge}/{N}, {cut_edge + 1}/{N}]) |")
log()

log("| k | N | макс. абс. ошибка | макс. отн. ошибка |")
for k in (0.5, 1.0, 2.0):
    for N in GRIDS:
        x = np.arange(N + 1) / N
        u = solve(k, k, N)[0]
        ue = x * (1 - x) / (2 * k)
        err = np.max(np.abs(u - ue))
        log(f"| {k} | {N} | {err:.2e} | {err / np.max(ue):.2e} |")
log()

rng = np.random.default_rng(SEED_SOLVER_CHECK)
theta = rng.uniform(LNK_LO, LNK_HI, size=(10, 2))
K = np.exp(theta)

V = {N: np.array([measure(k1, k2, N) for k1, k2 in K]) for N in GRIDS}
Vex = np.array([exact_measure(k1, k2) for k1, k2 in K])

log(f"## 1b. Сходимость по сеткам, 10 профилей (seed {SEED_SOLVER_CHECK}, ln k ~ U[ln 0.5, ln 2])\n")
log("Разности - максимум по 3 точкам измерения каждой нагрузки.\n")
log("| # | k1 | k2 | нагрузка | ‖v100−v200‖∞ | ‖v200−v400‖∞ | отношение | наблюдаемый порядок |")
ratios = {0: [], 1: []}
for i, (k1, k2) in enumerate(K):
    for load in (0, 1):
        sl = slice(3 * load, 3 * load + 3)
        d1 = np.max(np.abs(V[100][i, sl] - V[200][i, sl]))
        d2 = np.max(np.abs(V[200][i, sl] - V[400][i, sl]))
        r = d1 / d2 if d2 > 0 else np.inf
        ratios[load].append(r)
        order = np.log2(r) if np.isfinite(r) and r > 0 else float("nan")
        log(f"| {i + 1} | {k1:.4f} | {k2:.4f} | f{load + 1} | {d1:.3e} | {d2:.3e} | {r:.2f} | {order:.2f} |")
log()
for load in (0, 1):
    log(f"медиана наблюдаемого порядка, f{load + 1}: {np.log2(np.median(ratios[load])):.2f}")
log()

log("## 1c. (дополнительно) Ошибка относительно точного решения в точках измерения\n")
log("| # | нагрузка | ошибка N=100 | ошибка N=200 | ошибка N=400 | отн. ошибка N=200 | отн. ошибка N=400 |")
for i in range(10):
    for load in (0, 1):
        sl = slice(3 * load, 3 * load + 3)
        e = [np.max(np.abs(V[N][i, sl] - Vex[i, sl])) for N in GRIDS]
        scale = np.max(np.abs(Vex[i, sl]))
        log(f"| {i + 1} | f{load + 1} | {e[0]:.3e} | {e[1]:.3e} | {e[2]:.3e} | {e[1] / scale:.2e} | {e[2] / scale:.2e} |")
log()

est = np.max(np.abs(V[200] - V[400]), axis=1)
true = np.max(np.abs(V[400] - Vex), axis=1)
log("Разность сеток ‖v200−v400‖∞ против истинной ошибки при N=400 (максимум по всем 6 измерениям):\n")
log("| # | ‖v200−v400‖∞ | истинная ошибка N=400 | истинная ошибка N=200 |")
true200 = np.max(np.abs(V[200] - Vex), axis=1)
for i in range(10):
    log(f"| {i + 1} | {est[i]:.3e} | {true[i]:.3e} | {true200[i]:.3e} |")
log()

log("## Почему «порядок» для f1 не является порядком сходимости\n")
log("Ошибка измерений для f1 возникает только на ребре, содержащем a, и должна быть")
log("∝ h²·θ(1−θ), θ = frac(a·N) (относительное положение a внутри пересечённого ребра).\n")
log("| N | θ = frac(aN) | h²·θ(1−θ) |")
pred = {}
for N in GRIDS:
    th = A * N - np.floor(A * N)
    pred[N] = th * (1 - th) / N**2
    log(f"| {N} | {th:.2f} | {pred[N]:.3e} |")
log()
e1 = {N: np.max(np.abs(V[N][:, :3] - Vex[:, :3]), axis=1) for N in GRIDS}
log("| отношение | предсказание | наблюдение (медиана по 10 профилям) |")
log(f"| err(100)/err(200) | {pred[100] / pred[200]:.2f} | {np.median(e1[100] / e1[200]):.2f} |")
log(f"| err(200)/err(400) | {pred[200] / pred[400]:.2f} | {np.median(e1[200] / e1[400]):.2f} |")
dd = (pred[100] - pred[200]) / (pred[200] - pred[400])
log(f"| (e100−e200)/(e200−e400) | {dd:.2f} | (это столбец «отношение» таблицы 1b для f1) |")
log()

log("## Проверка полных профилей против точного решения (все узлы, N = 400)\n")
log("| # | нагрузка | макс. абс. ошибка | отн. ошибка L2 |")
for i, (k1, k2) in enumerate(K[:3]):
    x = np.arange(401) / 400
    u = solve(k1, k2, 400)
    for load in (0, 1):
        ue = exact(x, k1, k2, load)
        log(f"| {i + 1} | f{load + 1} | {np.max(np.abs(u[load] - ue)):.3e} | "
            f"{np.linalg.norm(u[load] - ue) / np.linalg.norm(ue):.3e} |")
log()

k1, k2 = 0.6, 1.8
x = np.arange(401) / 400
u = solve(k1, k2, 400)
xf = np.linspace(0, 1, 2001)
fig, ax = plt.subplots(figsize=(7, 4))
for load, c in ((0, "C0"), (1, "C1")):
    ax.plot(xf, exact(xf, k1, k2, load), color=c, lw=1, label=f"точное решение, f{load + 1}")
    ax.plot(x[::10], u[load, ::10], "o", ms=3, color=c, label=f"схема N=400, f{load + 1}")
    xm = np.array(X_MEAS)
    ax.plot(xm, exact(xm, k1, k2, load), "s", ms=8, mfc="none", color="k")
ax.axvline(A, color="gray", ls="--", label=f"граница a = {A:.4f}")
ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.set_title(f"k1 = {k1}, k2 = {k2}; квадраты - точки измерения")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "profiles.png", dpi=150)

(out / "solver_checks.md").write_text("\n".join(lines) + "\n")
print(f"\nсохранено: {out / 'solver_checks.md'}, {out / 'profiles.png'}")

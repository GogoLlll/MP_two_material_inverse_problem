import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data import load, load_norm, normalize
from mlp import MLP
from solver import measure, solve
from baselines import NearestNeighbor, nls_one
from gradcheck import central_difference
from sensitivity import jacobian, svd_stats, net_jacobian

out = C.RESULTS_DIR / "09_selfcheck"
out.mkdir(parents=True, exist_ok=True)
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


m, d = load_norm()
net = MLP.load(C.MODELS_DIR / "main_s11.npz")
ctrl = MLP.load(C.MODELS_DIR / "ctrl_s11.npz")

rng = np.random.default_rng(C.SEED_SELFCHECK)
theta = rng.uniform(C.LNK_LO, C.LNK_HI, size=2)
xi = rng.standard_normal(6)
k = np.exp(theta)
v_clean = measure(*k, C.N_TEST)
v = v_clean + C.NOISE_LEVEL * d * xi
z, z_clean = normalize(v, m, d), normalize(v_clean, m, d)

log("Самостоятельная проверка замороженной основной сетью (seed 11)\n")
log(f"генератор default_rng({C.SEED_SELFCHECK}): первые 2 числа → θ, следующие 6 → ξ (наш зафиксированный выбор).\n")
log(f"истинное θ = ({theta[0]:+.6f}, {theta[1]:+.6f}),  k = ({k[0]:.6f}, {k[1]:.6f})")
log(f"ξ = ({', '.join(f'{x:+.3f}' for x in xi)})\n")
labels = ["u1(.25)", "u1(.5)", "u1(.75)", "u2(.25)", "u2(.5)", "u2(.75)"]
log("| датчик | v без шума (N=400) | v с шумом | z с шумом |")
log("|---|---|---|---|")
for j in range(6):
    log(f"| {labels[j]} | {v_clean[j]:.6f} | {v[j]:.6f} | {z[j]:+.4f} |")
log()

U_true = solve(*k, C.N_TEST)


def report(name, th_hat):
    k_hat = np.exp(th_hat)
    ek = np.linalg.norm(k_hat - k) / np.linalg.norm(k)
    U = solve(*k_hat, C.N_TEST)
    eu = np.mean(np.linalg.norm(U - U_true, axis=1) / np.linalg.norm(U_true, axis=1))
    log(f"| {name} | {k_hat[0]:.5f} | {k_hat[1]:.5f} | {ek:.2e} | {eu:.2e} | {np.linalg.norm(th_hat - theta):.2e} |")
    return k_hat, U

v_tr, th_tr = load("train")
knn = NearestNeighbor(normalize(v_tr, m, d), th_tr)
th_nls, info = nls_one(z, m, d)

log("## Восстановление по зашумлённым измерениям\n")
log("| метод | k̂1 | k̂2 | e_k | e_u (профили, N=400) | ‖θ̂ − θ‖₂ |")
log("|---|---|---|---|---|---|")
log(f"| истина | {k[0]:.5f} | {k[1]:.5f} | - | - | - |")
k_net, U_net = report("**основная сеть s11**", net.predict(z[None])[0])
report("основная сеть s11, вход без шума", net.predict(z_clean[None])[0])
report("контрольная сеть s11", ctrl.predict(z[None, :3])[0])
report("1-NN", knn.predict(z[None])[0])
report(f"NLS (вызовов {info['nfev']})", th_nls)
log()

J = jacobian(theta, m, d)
st = svd_stats(J)
M = net_jacobian(net, z_clean)
log(f"Ожидаемая шумовая ошибка в этой точке: τ‖J⁺‖_F = {C.NOISE_LEVEL * st['pinv_F']:.2e} (лучшая возможная), "
    f"для этой сети τ‖M‖_F = {C.NOISE_LEVEL * np.linalg.norm(M):.2e}; κ = {st['kappa']:.2f}.")
log("Наблюдаемая ‖θ̂ − θ‖ - одна реализация шума такого масштаба.\n")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(C.N_TEST + 1) / C.N_TEST
for load_i, ax in enumerate(axes):
    ax.plot(x, U_true[load_i], lw=2, label=f"u(k), k = ({k[0]:.3f}, {k[1]:.3f})")
    ax.plot(x, U_net[load_i], "--", lw=1.5, label=f"u(k̂ сети), k̂ = ({k_net[0]:.3f}, {k_net[1]:.3f})")
    xm = np.array(C.X_MEAS)
    ax.plot(xm, v[3 * load_i:3 * load_i + 3], "ks", mfc="none", ms=8, label="зашумлённые измерения")
    ax.axvline(C.A, color="gray", ls=":", label=f"a = {C.A:.4f}")
    ax.set_title(f"нагрузка f{load_i + 1}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    inset = ax.inset_axes([0.36, 0.06, 0.28, 0.28])
    inset.plot(x, U_net[load_i] - U_true[load_i], lw=1)
    inset.axhline(0, color="gray", lw=0.5)
    inset.set_title("u(k̂) − u(k)", fontsize=7)
    inset.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(out / "selfcheck_profiles.png", dpi=150)

log("## Производная tanh\n")
log("tanh t = sinh t / cosh t, (sinh)' = cosh, (cosh)' = sinh:")
log("(tanh t)' = (cosh t · cosh t − sinh t · sinh t) / cosh² t = 1 / cosh² t = 1 − tanh² t.")
t = rng.uniform(-3, 3, 5)
h = 1e-5
log("\n| t | (tanh(t+h) − tanh(t−h))/(2h) | 1 − tanh² t | разность |")
for ti in t:
    num = (np.tanh(ti + h) - np.tanh(ti - h)) / (2 * h)
    ana = 1 - np.tanh(ti) ** 2
    log(f"| {ti:+.4f} | {num:.12f} | {ana:.12f} | {num - ana:+.1e} |")
log()

names = net.param_names()
pi = int(rng.integers(len(net.params)))
fi = int(rng.integers(net.params[pi].size))
idx = tuple(int(i) for i in np.unravel_index(fi, net.params[pi].shape))
log(f"## Проверка градиента случайно выбранного параметра замороженной сети: {names[pi]}{list(idx)}\n")
log("| на чём считалась потеря | g (обратный проход) | b (центральная разность, ε = 1e-5) | относительная разность |")
batches = {"пример самопроверки (зашумлённый z, истинное θ)": (z[None], theta[None]),
           "первые 32 примера train": (normalize(v_tr[:32], m, d), th_tr[:32])}
before = net.copy_params()
for name, (Zb, Tb) in batches.items():
    _, grads = net.loss_and_grads(Zb, Tb)
    g = grads[pi].ravel()[fi]
    b = central_difference(net, Zb, Tb, pi, fi)
    rel = abs(g - b) / max(1e-8, abs(g) + abs(b))
    log(f"| {name} | {g:+.12e} | {b:+.12e} | {rel:.1e} |")
assert all(np.array_equal(a, b_) for a, b_ in zip(before, net.params)), "веса должны оставаться замороженными"
log("\nПроверка не изменила замороженные веса.\n")

(out / "selfcheck.md").write_text("\n".join(lines) + "\n")
print(f"сохранено: {out}")

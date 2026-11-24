import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data import load, normalize
from mlp import MLP, OPTIMIZERS

out = C.RESULTS_DIR / "03_mlp"
out.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s)
    lines.append(s)

main, ctrl = MLP((6, 64, 64, 2), seed=11), MLP((3, 64, 64, 2), seed=11)
log("## 1. Число параметров\n")
for name, net in (("основная 6-64-64-2", main), ("контрольная 3-64-64-2", ctrl)):
    shapes = ", ".join(f"{n}{p.shape}" for n, p in zip(net.param_names(), net.params))
    log(f"- {name}: {net.n_params()} параметров ({shapes})")
assert main.n_params() == 4738 and ctrl.n_params() == 4546
log()

t = np.linspace(-4, 4, 81)
h = 1e-5
num = (np.tanh(t + h) - np.tanh(t - h)) / (2 * h)
ana = 1 - np.tanh(t) ** 2
log("## 2. Производная tanh\n")
log("tanh t = sinh t / cosh t,  (tanh t)' = (cosh² t − sinh² t)/cosh² t = 1/cosh² t = 1 − tanh² t.")
log(f"макс. |центральная разность − (1 − tanh²)| на [−4, 4]: {np.max(np.abs(num - ana)):.2e}\n")

v, th = load("train")
Z = normalize(v)
log("## 3. Инициализация (LeCun, W ~ N(0, 1/n_in)), прямой проход по всем входам train\n")
log("| слой | дисперсия входа активации H | среднее |tanh'(H)| = среднее(1 − A²) |")
A = Z
for l, (W, b) in enumerate(zip(main.W, main.b), 1):
    H = A @ W.T + b
    if l < len(main.W):
        A = np.tanh(H)
        log(f"| {l} | {H.var():.3f} | {np.mean(1 - A**2):.3f} |")
    else:
        log(f"| {l} (выход) | {H.var():.3f} | - |")
log("\nДисперсия входов скрытых слоёв меньше 1, средняя tanh' 0.7–0.8: нейроны не насыщены.\n")

Zs, Ts = Z[:32], th[:32]
settings = {"sgd": 0.1, "momentum": 0.03, "adam": 3e-3}
steps = 3000
log(f"## 4. Запоминание 32 примеров train (полный батч, {steps} шагов)\n")
log("| оптимизатор | шаг | начальная потеря | итоговая потеря | итоговая RMSE θ |")
fig, ax = plt.subplots(figsize=(6, 4))
for name, lr in settings.items():
    net = MLP((6, 64, 64, 2), seed=11)
    opt = OPTIMIZERS[name](net.params, lr)
    hist = []
    for _ in range(steps):
        loss, grads = net.loss_and_grads(Zs, Ts)
        hist.append(loss)
        opt.step(grads)
    final = net.loss(net.predict(Zs), Ts)
    rmse = np.sqrt(np.mean((net.predict(Zs) - Ts) ** 2))
    log(f"| {name} | {lr} | {hist[0]:.3e} | {final:.3e} | {rmse:.2e} |")
    ax.semilogy(hist, label=f"{name}, шаг={lr}")
ax.set_xlabel("шаг")
ax.set_ylabel("потеря на 32 примерах")
ax.set_title("запоминание маленькой выборки")
ax.legend()
fig.tight_layout()
fig.savefig(out / "memorize.png", dpi=150)
log("\nЭто только проверка, что прямой проход, обратный проход и обновления работают вместе; шаги здесь")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "net.npz"
    main.save(path)
    back = MLP.load(path)
    diff = max(np.max(np.abs(a - b)) for a, b in zip(main.params, back.params))
log(f"## 5. Сохранение и загрузка весов: макс. разность параметров {diff:.1e}\n")
assert diff == 0

(out / "mlp_checks.md").write_text("\n".join(lines) + "\n")
print(f"сохранено: {out}")

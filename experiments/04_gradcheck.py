import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config as C
from data import load, normalize
from mlp import MLP
from gradcheck import check, pick_parameters, central_difference

EPS = 1e-5
TOL = 1e-4
BATCH = 32

out = C.RESULTS_DIR / "04_gradcheck"
out.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s)
    lines.append(s)


v, th = load("train")
Z = normalize(v)[:BATCH]
T = th[:BATCH]
assert Z.dtype == np.float64

log("Проверка градиента (до обучения)\n")
log(f"мини-батч: первые {BATCH} примеров train; eps = {EPS}; float64; seed выбора параметров = {C.S}\n")

results = {}
for name, sizes, cols in (("основная 6-64-64-2", (6, 64, 64, 2), slice(0, 6)),
                          ("контрольная 3-64-64-2", (3, 64, 64, 2), slice(0, 3))):
    net = MLP(sizes, seed=11)
    picks = pick_parameters(net, np.random.default_rng(C.S))
    rows, rel = check(net, Z[:, cols], T, picks, EPS)
    results[name] = rel
    log(f"## {name}  ({net.n_params()} параметров)\n")
    log("| # | параметр | g (обратный проход) | b (центральная разность) | g − b |")
    for i, (pname, gi, bi) in enumerate(rows, 1):
        log(f"| {i} | {pname} | {gi:+.12e} | {bi:+.12e} | {gi - bi:+.2e} |")
    status = "ПРОЙДЕНО" if rel <= TOL else "НЕ ПРОЙДЕНО"
    log(f"\n‖g − b‖₂ / max(1e-8, ‖g‖₂ + ‖b‖₂) = **{rel:.3e}**  (допуск {TOL:.0e}) → **{status}**\n")

net = MLP((6, 64, 64, 2), seed=11)
_, grads = net.loss_and_grads(Z, T)
log("## Дополнительно: все параметры основной сети\n")
log("| массив | размер | отн. ошибка по всему массиву | макс. |g − b| |")
G_all, B_all = [], []
for pi, (pname, p) in enumerate(zip(net.param_names(), net.params)):
    g = grads[pi].ravel()
    b = np.array([central_difference(net, Z, T, pi, fi, EPS) for fi in range(p.size)])
    G_all.append(g)
    B_all.append(b)
    rel = np.linalg.norm(g - b) / max(1e-8, np.linalg.norm(g) + np.linalg.norm(b))
    log(f"| {pname} | {p.size} | {rel:.2e} | {np.max(np.abs(g - b)):.2e} |")
G_all, B_all = np.concatenate(G_all), np.concatenate(B_all)
rel_all = np.linalg.norm(G_all - B_all) / max(1e-8, np.linalg.norm(G_all) + np.linalg.norm(B_all))
log(f"\nвсе {G_all.size} параметров: относительная ошибка {rel_all:.3e}\n")

assert all(r <= TOL for r in results.values()) and rel_all <= TOL
(out / "gradcheck.md").write_text("\n".join(lines) + "\n")
print(f"сохранено: {out / 'gradcheck.md'}")

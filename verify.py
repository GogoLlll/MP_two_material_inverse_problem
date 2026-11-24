import sys

import numpy as np

import config as C
from solver import solve, measure, exact_measure

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'OK' if ok else 'ОШИБКА'}] {name}: {detail}")


x = np.arange(401) / 400
err = max(np.max(np.abs(solve(k, k, 400)[0] - x * (1 - x) / (2 * k))) for k in (0.5, 1.0, 2.0))
check("решатель, k1 = k2", err < 1e-12, f"макс. ошибка относительно x(1-x)/(2k) = {err:.1e}")

rel = max(np.max(np.abs(measure(k1, k2, 400) - exact_measure(k1, k2)) / np.abs(exact_measure(k1, k2)))
          for k1, k2 in ((0.6, 1.8), (1.9, 0.55), (1.0, 1.3)))
check("решатель против точного решения (N = 400)", rel < 1e-5, f"макс. относительная ошибка измерений = {rel:.1e}")

if not (C.DATA_DIR / "test_noisy.npz").exists():
    from data import generate_all
    print("       папка data/ не найдена, генерирую заново (детерминированно) ...")
    generate_all(verbose=False)
from data import load, load_norm, normalize

m, d = load_norm()
v_tr, th_tr = load("train")
sizes = {n: len(load(n)[0]) for n in ("train", "val", "test", "test_noisy", "ood")}
ok = sizes == {"train": 4000, "val": 1000, "test": 1000, "test_noisy": 1000, "ood": 300}
z_tr = normalize(v_tr, m, d)
ok &= np.allclose(z_tr.mean(0), 0, atol=1e-12) and np.allclose(z_tr.std(0), 1)
check("наборы данных", ok, f"размеры {sizes}, train нормирован к среднему 0 и std 1")

from mlp import MLP 
from gradcheck import check as gcheck, pick_parameters

net = MLP((6, 64, 64, 2), seed=11)
_, relg = gcheck(net, z_tr[:32], th_tr[:32], pick_parameters(net, np.random.default_rng(C.S)))
check("ручной backprop против центральных разностей", relg <= 1e-4,
      f"относительная ошибка {relg:.1e} (допуск 1e-4)")

from metrics import k_errors

expected = {"clean": 1.42e-3, "noisy": 6.44e-3}
sets = {"clean": "test", "noisy": "test_noisy"}
set_names = {"clean": "чистом", "noisy": "шумовом"}
for tag, name in sets.items():
    v, th = load(name)
    z = normalize(v, m, d)
    meds = [np.median(k_errors(MLP.load(C.MODELS_DIR / f"main_s{s}.npz").predict(z), th)[0]) for s in C.SEEDS_NET]
    val = float(np.mean(meds))
    check(f"замороженные основные сети на {set_names[tag]} test", abs(val - expected[tag]) / expected[tag] < 0.01,
          f"медиана e_k = {val:.3e} (в отчёте: {expected[tag]:.2e})")

from baselines import nls_one 
from sensitivity import jacobian, svd_stats

v_n, th_test = load("test_noisy")
z_n = normalize(v_n, m, d)
errs, preds = [], []
for i in range(5):
    t_hat, _ = nls_one(z_n[i], m, d)
    errs.append(np.linalg.norm(t_hat - th_test[i]))
    preds.append(C.NOISE_LEVEL * svd_stats(jacobian(th_test[i], m, d))["pinv_F"])
r_err, r_pred = np.sqrt(np.mean(np.square(errs))), np.sqrt(np.mean(np.square(preds)))
check("ошибка NLS против τ‖J⁺‖_F (5 точек)", 0.3 < r_err / r_pred < 3,
      f"RMS ошибки NLS {r_err:.2e}, прогноз {r_pred:.2e}")

print(f"\nпройдено проверок: {sum(results)}/{len(results)}")
sys.exit(0 if all(results) else 1)

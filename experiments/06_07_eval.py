import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data import load, load_norm, normalize
from mlp import MLP
from metrics import summarize, true_profiles, k_errors
from baselines import NearestNeighbor, nls

out = C.RESULTS_DIR / "06_07_eval"
out.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s, flush=True)
    lines.append(s)

m, d = load_norm()
v_tr, th_tr = load("train")
Z_tr = normalize(v_tr, m, d)
SETS = {"clean": "test", "noisy": "test_noisy", "ood": "ood"}
SET_NAMES = {"clean": "чистый test", "noisy": "шумовой test", "ood": "OOD"}
data = {}
for tag, name in SETS.items():
    v, th = load(name)
    data[tag] = (normalize(v, m, d), th)
U_true = {"clean": true_profiles(data["clean"][1]), "ood": true_profiles(data["ood"][1])}
U_true["noisy"] = U_true["clean"]                  # те же параметры, шум только в измерениях

nets = {(tag, s): MLP.load(C.MODELS_DIR / f"{tag}_s{s}.npz") for tag in ("main", "ctrl") for s in C.SEEDS_NET}
COLS = {"main": slice(0, 6), "ctrl": slice(0, 3)}
NET_NAMES = {"main": "сеть основная", "ctrl": "сеть контрольная"}
knn = NearestNeighbor(Z_tr, th_tr)

COLS_MET = ["k_med", "k_p90", "u_med", "u_p90", "rmse_theta"]
HDR = "| метод | медиана e_k | p90 e_k | медиана e_u | p90 e_u | RMSE θ | срывы |"
SEP = "|---|---|---|---|---|---|---|"

def fmt_single(name, s):
    return (f"| {name} | " + " | ".join(f"{s[c]:.2e}" for c in COLS_MET) + f" | {s['failures']}/{s['n']} |")

def fmt_seeds(name, ss):
    cells = []
    for c in COLS_MET:
        vals = np.array([s[c] for s in ss])
        cells.append(f"{vals.mean():.2e} ± {vals.std(ddof=1):.1e}")
    fails = "/".join(str(s["failures"]) for s in ss)
    return f"| {name} | " + " | ".join(cells) + f" | {fails} (из {ss[0]['n']}) |"

log("e_k = ‖k̂ − k‖₂/‖k‖₂; e_u = среднее по двум нагрузкам ‖u(k̂) − u(k)‖₂/‖u(k)‖₂ (полные профили, N = 400).")
log("Сети: среднее ± выборочное std каждой статистики по seed 11, 23, 37; срывы указаны для каждого seed.")
log("Срывы входят в статистику как +∞ (никогда не отбрасываются).\n")

preds = {}
log("## Полные наборы: сети и 1-NN\n")
for tag in SETS:
    Z, th = data[tag]
    log(f"### {SET_NAMES[tag]} ({len(th)} примеров)\n")
    log(HDR)
    log(SEP)
    for net_tag in ("main", "ctrl"):
        ss = []
        for s in C.SEEDS_NET:
            p = nets[(net_tag, s)].predict(Z[:, COLS[net_tag]])
            preds[(tag, net_tag, s)] = p
            ss.append(summarize(p, th, U_true[tag]))
        log(fmt_seeds(NET_NAMES[net_tag], ss))
    p = knn.predict(Z)
    preds[(tag, "knn")] = p
    log(fmt_single("1-NN", summarize(p, th, U_true[tag])))
    log()

log("## Первые 100 примеров каждого набора: основная сеть, 1-NN, NLS\n")
log("Время - реальное время на один пример. Сеть: один батч из 100 (и вызов на одном примере); "
    "1-NN: один пакетный запрос; NLS: своя оптимизация для каждого примера.\n")
nls_store = {}
for tag in SETS:
    Z, th = data[tag]
    Z100, th100, U100 = Z[:100], th[:100], U_true[tag][:100]
    log(f"### {SET_NAMES[tag]}, первые 100\n")
    log(HDR + " время на пример |")
    log(SEP + "---|")

    ss, times = [], []
    for s in C.SEEDS_NET:
        net = nets[("main", s)]
        t0 = time.perf_counter()
        for _ in range(100):
            p = net.predict(Z100)
        times.append((time.perf_counter() - t0) / 100 / 100)
        ss.append(summarize(p, th100, U100))
    t0 = time.perf_counter()
    for z in Z100:
        nets[("main", 11)].predict(z[None, :])
    t_single = (time.perf_counter() - t0) / 100
    log(fmt_seeds("сеть основная", ss) + f" {np.mean(times):.1e} с (один вызов {t_single:.1e} с) |")

    t0 = time.perf_counter()
    p = knn.predict(Z100)
    t_knn = (time.perf_counter() - t0) / 100
    log(fmt_single("1-NN", summarize(p, th100, U100)) + f" {t_knn:.1e} с |")

    p, info = nls(Z100, m, d)
    nls_store[tag] = (p, info)
    s_nls = summarize(p, th100, U100)
    t_nls = np.mean([i["seconds"] for i in info])
    log(fmt_single("NLS", s_nls) + f" {t_nls:.1e} с |")
    status = np.array([i["status"] for i in info])
    log(f"\nДиагностика NLS: среднее число вызовов {np.mean([i['nfev'] for i in info]):.1f}, "
        f"исключений {int((status == -99).sum())}, статус ≤ 0: {int((status <= 0).sum())}, "
        f"решение на границе: {int(sum(i['at_bound'] for i in info))}, "
        f"макс. итоговая невязка {np.nanmax([i['cost'] for i in info]):.1e}\n")

np.savez(out / "nls_first100.npz",
         **{f"{tag}_theta": nls_store[tag][0] for tag in SETS},
         **{f"{tag}_nfev": np.array([i["nfev"] for i in nls_store[tag][1]]) for tag in SETS},
         **{f"{tag}_at_bound": np.array([i["at_bound"] for i in nls_store[tag][1]]) for tag in SETS})

log("## Влияние шума (полные наборы для сетей и 1-NN)\n")
log("| метод | медиана e_k, чистый | медиана e_k, шумовой | отношение шумовой / чистый |")
for name, key in (("сеть основная", "main"), ("сеть контрольная", "ctrl")):
    c = np.mean([np.median(k_errors(preds[("clean", key, s)], data["clean"][1])[0]) for s in C.SEEDS_NET])
    n = np.mean([np.median(k_errors(preds[("noisy", key, s)], data["noisy"][1])[0]) for s in C.SEEDS_NET])
    log(f"| {name} | {c:.2e} | {n:.2e} | {n / c:.1f} |")
c = np.median(k_errors(preds[("clean", "knn")], data["clean"][1])[0])
n = np.median(k_errors(preds[("noisy", "knn")], data["noisy"][1])[0])
log(f"| 1-NN | {c:.2e} | {n:.2e} | {n / c:.1f} |")
log()

th_ood = data["ood"][1]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (name, p) in zip(axes, (("основная сеть, seed 11", preds[("ood", "main", 11)]),
                                ("1-NN", preds[("ood", "knn")]),
                                ("NLS (первые 100)", nls_store["ood"][0]))):
    n = len(p)
    for j, mk in ((0, "o"), (1, "^")):
        ax.scatter(np.exp(th_ood[:n, j]), np.exp(p[:, j]), s=10, marker=mk, alpha=0.6, label=f"k{j + 1}")
    ax.plot([0.25, 4], [0.25, 4], "k--", lw=0.8)
    ax.axvspan(0.5, 2, color="gray", alpha=0.12, label="диапазон train")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("истинное k")
    ax.set_ylabel("восстановленное k")
    ax.set_title(f"OOD: {name}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "ood_scatter.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.logspace(-6, 0.5, 60)
for name, key in (("основная s11, чистый", ("clean", "main", 11)), ("основная s11, шумовой", ("noisy", "main", 11)),
                  ("контрольная s11, шумовой", ("noisy", "ctrl", 11)), ("1-NN, чистый", ("clean", "knn"))):
    e, _ = k_errors(preds[key], data[key[0]][1])
    ax.hist(e[np.isfinite(e)], bins=bins, histtype="step", label=name)
ax.set_xscale("log")
ax.set_xlabel("относительная ошибка k")
ax.set_ylabel("число примеров (из 1000)")
ax.set_title("распределение e_k на тестовом наборе")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "error_hist.png", dpi=150)

(out / "eval.md").write_text("\n".join(lines) + "\n")
print(f"сохранено: {out}")

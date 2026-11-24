import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data import generate_all, load, load_norm, normalize
from solver import measure

out = C.RESULTS_DIR / "02_data"
out.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s)
    lines.append(s)

t0 = time.perf_counter()
generate_all(verbose=False)
log(f"Наборы данных\n\ns = {C.S}, a = {C.A:.4f}; время генерации {time.perf_counter() - t0:.1f} с\n")

names = ["train", "val", "test", "test_noisy", "ood", "test_N200"]
D = {n: load(n) for n in names}
m, d = load_norm()

log("## Размеры и диапазоны параметров\n")
log("| набор | n | seed | сетка N | диапазон ln k1 | диапазон ln k2 | диапазон k |")
info = {
    "train": (C.SEED_TRAIN, C.N_TRAIN), "val": (C.SEED_VAL, C.N_TRAIN), "test": (C.SEED_TEST, C.N_TEST),
    "test_noisy": (f"{C.SEED_TEST} + шум {C.SEED_NOISE}", C.N_TEST), "ood": (C.SEED_OOD, C.N_TEST),
    "test_N200": (C.SEED_TEST, C.N_TRAIN),
}
for n in names:
    v, th = D[n]
    k = np.exp(th)
    log(f"| {n} | {len(v)} | {info[n][0]} | {info[n][1]} | [{th[:, 0].min():.3f}, {th[:, 0].max():.3f}] | "
        f"[{th[:, 1].min():.3f}, {th[:, 1].max():.3f}] | [{k.min():.3f}, {k.max():.3f}] |")
log(f"\nln 0.5 = {C.LNK_LO:.4f}, ln 2 = {C.LNK_HI:.4f}, ln 0.25 = {C.LNK_OOD_LO:.4f}, ln 4 = {C.LNK_OOD_HI:.4f}\n")

expected = {"train": 4000, "val": 1000, "test": 1000, "test_noisy": 1000, "ood": 300, "test_N200": 1000}
for n, cnt in expected.items():
    assert len(D[n][0]) == cnt, n

th_ood = D["ood"][1]
outside = (th_ood < C.LNK_LO) | (th_ood > C.LNK_HI)
rejected = int(np.load(C.DATA_DIR / "ood.npz")["rejected"])
assert outside.any(axis=1).all()
log("## OOD\n")
log(f"принято 300 пар, отброшено {rejected} (оба k внутри [0.5, 2]); "
    f"доля принятых {300 / (300 + rejected):.3f} (теория 1 − (1/2)² = 0.75)")
log(f"пар, где вне диапазона только k1: {int((outside[:, 0] & ~outside[:, 1]).sum())}, "
    f"только k2: {int((~outside[:, 0] & outside[:, 1]).sum())}, оба: {int(outside.all(axis=1).sum())}\n")

rng = np.random.default_rng(0)
log("## Выборочная проверка: сохранённые v совпадают с новым вызовом решателя\n")
for n, N in (("train", C.N_TRAIN), ("test", C.N_TEST), ("ood", C.N_TEST)):
    v, th = D[n]
    idx = rng.choice(len(v), 5, replace=False)
    err = max(np.max(np.abs(v[i] - measure(*np.exp(th[i]), N))) for i in idx)
    log(f"- {n}: макс. |разность| по 5 случайным строкам = {err:.1e}")
log()

labels = ["u1(.25)", "u1(.5)", "u1(.75)", "u2(.25)", "u2(.5)", "u2(.75)"]
log("## Нормировка (только по train)\n")
log("| признак | m | d | d/m | среднее z_train | std z_train | среднее z_test | std z_test | мин. z_ood | макс. z_ood |")
Z = {n: normalize(D[n][0], m, d) for n in names}
for j, lab in enumerate(labels):
    log(f"| {lab} | {m[j]:.5f} | {d[j]:.5f} | {d[j] / m[j]:.2f} | {Z['train'][:, j].mean():+.1e} | "
        f"{Z['train'][:, j].std():.4f} | {Z['test'][:, j].mean():+.3f} | {Z['test'][:, j].std():.3f} | "
        f"{Z['ood'][:, j].min():+.2f} | {Z['ood'][:, j].max():+.2f} |")
log()

log("Корреляционная матрица z_train (признаки сильно зависимы: 6 чисел управляются 2 параметрами):\n")
corr = np.corrcoef(Z["train"].T)
log("| | " + " | ".join(labels) + " |")
log("|---" * 7 + "|")
for j, lab in enumerate(labels):
    log(f"| {lab} | " + " | ".join(f"{c:+.3f}" for c in corr[j]) + " |")
log()

dz = Z["test_noisy"] - Z["test"]
xi = np.load(C.DATA_DIR / "test_noisy.npz")["xi"]
log("## Шумовой test\n")
log(f"z_noisy − z_test: среднее {dz.mean():+.2e}, std {dz.std():.5f} (ожидается 0.01); "
    f"макс. |z_noisy − z_test − 0.01·xi| = {np.max(np.abs(dz - 0.01 * xi)):.1e}")
log("Контрольная сеть использует столбцы 0..2 тех же зашумлённых векторов.\n")

shift = Z["test_N200"] - Z["test"]
log("## Сдвиг от дискретизации: те же параметры test при N = 200 и N = 400 (в единицах z)\n")
log("| признак | средний сдвиг | макс. |сдвиг| | отношение к шуму 0.01 |")
for j, lab in enumerate(labels):
    log(f"| {lab} | {shift[:, j].mean():+.2e} | {np.abs(shift[:, j]).max():.2e} | "
        f"{np.abs(shift[:, j]).max() / 0.01:.3f} |")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.scatter(*D["train"][1].T, s=2, alpha=0.4, label="train (4000)")
ax.scatter(*D["test"][1].T, s=2, alpha=0.6, label="test (1000)")
ax.scatter(*th_ood.T, s=6, color="C3", label="OOD (300)")
for v_ in (C.LNK_LO, C.LNK_HI):
    ax.axvline(v_, color="gray", lw=0.8)
    ax.axhline(v_, color="gray", lw=0.8)
ax.set_xlabel("θ1 = ln k1")
ax.set_ylabel("θ2 = ln k2")
ax.set_title("параметры")
ax.legend(fontsize=8, markerscale=3)
ax.set_aspect("equal")

ax = axes[1]
for j, lab in enumerate(labels):
    ax.hist(D["train"][0][:, j], bins=50, histtype="step", label=lab)
ax.set_xlabel("измеренное u")
ax.set_title("измерения v в train")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "datasets.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, j in zip(axes, (1, 4)):
    sc = ax.scatter(*D["train"][1].T, c=D["train"][0][:, j], s=4, cmap="viridis")
    ax.set_xlabel("θ1 = ln k1")
    ax.set_ylabel("θ2 = ln k2")
    ax.set_title(f"{labels[j]} в пространстве параметров (train)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax)
fig.tight_layout()
fig.savefig(out / "measurement_maps.png", dpi=150)

(out / "data_checks.md").write_text("\n".join(lines) + "\n")
print(f"\nсохранено: {out}")

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
from baselines import nls_one
from sensitivity import jacobian, svd_stats, net_jacobian, left_inverse_split

TAU = C.NOISE_LEVEL
NPTS = 20
NMC = 100
SEED_MC = C.SEED_NOISE + 1

out = C.RESULTS_DIR / "08_sensitivity"
out.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s, flush=True)
    lines.append(s)

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

m, d = load_norm()
v_test, th_test = load("test")
v_noisy, _ = load("test_noisy")
v_t200, _ = load("test_N200")
Z, Zn, Z200 = normalize(v_test, m, d), normalize(v_noisy, m, d), normalize(v_t200, m, d)
th = th_test[:NPTS]
nets = {(t, s): MLP.load(C.MODELS_DIR / f"{t}_s{s}.npz") for t in ("main", "ctrl") for s in C.SEEDS_NET}
COLS = {6: slice(0, 6), 3: slice(0, 3)}
NET_OF = {6: "main", 3: "ctrl"}
NAMES = {"main": "основная", "ctrl": "контрольная"}

log("Локальная чувствительность к шуму\n")
log(f"Первые {NPTS} точек test, N = 400, J = ∂z/∂θ центральными разностями (шаг 1e-5), τ = {TAU}.\n")

J6 = np.array([jacobian(t, m, d) for t in th])
J = {6: J6, 3: J6[:, :3, :]}
S = {n: [svd_stats(J[n][i]) for i in range(NPTS)] for n in (6, 3)}

log("## 1. Сингулярные числа J\n")
log("| # | k1 | k2 | σmax (3) | σmin (3) | κ (3) | ‖J⁺‖_F (3) | σmax (6) | σmin (6) | κ (6) | ‖J⁺‖_F (6) |")
for i in range(NPTS):
    a, b = S[3][i], S[6][i]
    k1, k2 = np.exp(th[i])
    log(f"| {i + 1} | {k1:.3f} | {k2:.3f} | {a['smax']:.3f} | {a['smin']:.4f} | {a['kappa']:.1f} | {a['pinv_F']:.2f} | "
        f"{b['smax']:.3f} | {b['smin']:.4f} | {b['kappa']:.1f} | {b['pinv_F']:.2f} |")
med = {n: {key: np.median([s[key] for s in S[n]]) for key in ("smax", "smin", "kappa", "pinv_F")} for n in (6, 3)}
log(f"| медиана | | | {med[3]['smax']:.3f} | {med[3]['smin']:.4f} | {med[3]['kappa']:.1f} | {med[3]['pinv_F']:.2f} | "
    f"{med[6]['smax']:.3f} | {med[6]['smin']:.4f} | {med[6]['kappa']:.1f} | {med[6]['pinv_F']:.2f} |")
rank_lost = {n: sum(not s["full_rank"] for s in S[n]) for n in (6, 3)}
log(f"\nПотеря ранга (σmin/σmax < 1e-10): 3 датчика - {rank_lost[3]} точек, 6 датчиков - {rank_lost[6]} точек.")
smin_up = sum(S[6][i]["smin"] >= S[3][i]["smin"] - 1e-12 for i in range(NPTS))
smax_up = sum(S[6][i]["smax"] >= S[3][i]["smax"] - 1e-12 for i in range(NPTS))
k_up = sum(S[6][i]["kappa"] > S[3][i]["kappa"] for i in range(NPTS))
p_down = sum(S[6][i]["pinv_F"] <= S[3][i]["pinv_F"] for i in range(NPTS))
log(f"6 датчиков против 3: σmin не меньше в {smin_up}/{NPTS}, σmax не меньше в {smax_up}/{NPTS}, "
    f"κ больше в {k_up}/{NPTS}, ‖J⁺‖_F не больше в {p_down}/{NPTS} точках.\n")

_, _, Vt = np.linalg.svd(J6[0])
log(f"Точка 1, 6 датчиков: самое слабое направление в θ (правый сингулярный вектор σmin) = "
    f"({Vt[-1, 0]:+.3f}, {Vt[-1, 1]:+.3f}); самое сильное = ({Vt[0, 0]:+.3f}, {Vt[0, 1]:+.3f}).\n")

nls_saved = np.load(C.RESULTS_DIR / "06_07_eval" / "nls_first100.npz")["noisy_theta"][:NPTS]
rng = np.random.default_rng(SEED_MC)
XI = rng.standard_normal((NPTS, NMC, 6))             # общий для всех методов: одинаковый шум для всех

mc_nls = {6: np.zeros((NPTS, NMC)), 3: np.zeros((NPTS, NMC))}
nls_official3 = np.zeros(NPTS)
for i in range(NPTS):
    for n in (6, 3):
        for r in range(NMC):
            z_obs = (Z[i] + TAU * XI[i, r])[COLS[n]]
            t_hat, _ = nls_one(z_obs, m, d, cols=COLS[n])
            mc_nls[n][i, r] = np.linalg.norm(t_hat - th[i])
    t_hat, _ = nls_one(Zn[i, :3], m, d, cols=COLS[3])
    nls_official3[i] = np.linalg.norm(t_hat - th[i])
nls_official6 = np.linalg.norm(nls_saved - th, axis=1)

log("## 2. Прогноз шумовой ошибки τ‖J⁺‖_F против NLS\n")
log(f"Официальный шумовой test (seed 101): одна реализация шума на точку. Монте-Карло: {NMC} дополнительных "
    f"реализаций на точку (seed {SEED_MC}), RMS по реализациям.\n")
log("| # | прогноз (3) | NLS офиц. (3) | NLS МК RMS (3) | прогноз (6) | NLS офиц. (6) | NLS МК RMS (6) |")
for i in range(NPTS):
    log(f"| {i + 1} | {TAU * S[3][i]['pinv_F']:.2e} | {nls_official3[i]:.2e} | {rms(mc_nls[3][i]):.2e} | "
        f"{TAU * S[6][i]['pinv_F']:.2e} | {nls_official6[i]:.2e} | {rms(mc_nls[6][i]):.2e} |")
pred = {n: rms([TAU * s["pinv_F"] for s in S[n]]) for n in (6, 3)}
log(f"| **RMS по 20** | **{pred[3]:.2e}** | **{rms(nls_official3):.2e}** | **{rms(mc_nls[3]):.2e}** | "
    f"**{pred[6]:.2e}** | **{rms(nls_official6):.2e}** | **{rms(mc_nls[6]):.2e}** |")
log("\nВсе числа - ‖θ̂ − θ‖₂. Одна реализация на точку колеблется на десятки процентов; "
    "формула предсказывает именно RMS по Монте-Карло.\n")

log("## 3. Усиление шума сетями\n")
log("M = ∂θ̂/∂z на чистом входе. Если сеть точно обращает прямое отображение на многообразии данных, то "
    "M·J = I и ‖M‖²_F = ‖J⁺‖²_F + ‖M P⊥‖²_F (P⊥ - проектор на направления, ортогональные образу J, "
    "которые не встречались в обучающих данных без шума).\n")
log("| сеть | seed | медиана ‖MJ − I‖₂ | RMS τ‖J⁺‖_F | RMS τ‖M P_range‖_F | RMS τ‖M P⊥‖_F | RMS τ‖M‖_F (прогноз) | "
    "МК RMS ‖δθ‖ (100 реализаций) | офиц. шумовой RMS ‖θ̂ − θ‖ |")
net_tab = {}
for n in (6, 3):
    tag = NET_OF[n]
    for s in C.SEEDS_NET:
        net = nets[(tag, s)]
        rows = [left_inverse_split(net_jacobian(net, Z[i, COLS[n]]), J[n][i]) for i in range(NPTS)]
        clean_pred = net.predict(Z[:NPTS, COLS[n]])
        mc = np.array([[np.linalg.norm(net.predict((Z[i] + TAU * XI[i, r])[None, COLS[n]])[0] - clean_pred[i])
                        for r in range(NMC)] for i in range(NPTS)])
        official = np.linalg.norm(net.predict(Zn[:NPTS, COLS[n]]) - th, axis=1)
        r_ = {k: rms([TAU * row[k] for row in rows]) for k in ("M_F", "M_range_F", "M_perp_F")}
        net_tab[(tag, s)] = {**r_, "mc": rms(mc), "official": rms(official),
                             "MJ": np.median([row["MJ_minus_I"] for row in rows])}
        log(f"| {NAMES[tag]} | {s} | {net_tab[(tag, s)]['MJ']:.2e} | {pred[n]:.2e} | {r_['M_range_F']:.2e} | "
            f"{r_['M_perp_F']:.2e} | {r_['M_F']:.2e} | {rms(mc):.2e} | {rms(official):.2e} |")
log("\nМК RMS ‖δθ‖ измеряет изменение ответа сети, вызванное только шумом (θ̂(z+δz) − θ̂(z)); "
    "официальный столбец включает ещё и ошибку сети на чистых данных.\n")

log("## 4. Дискретизация: сетка обучения N = 200 против сетки теста N = 400\n")
dz = Z200[:NPTS] - Z[:NPTS]
lin = np.array([np.linalg.norm(np.linalg.pinv(J6[i]) @ dz[i]) for i in range(NPTS)])
log(f"‖δz‖₂ = ‖z(N=200) − z(N=400)‖₂: медиана {np.median(np.linalg.norm(dz, axis=1)):.2e} "
    f"(шум: τ·√6 = {TAU * np.sqrt(6):.2e})")
log(f"Линейный эффект на θ, ‖J⁺δz‖₂ (точное обращение, применённое к данным N=200): RMS по 20 точкам {rms(lin):.2e}\n")
log("| сеть | seed | RMS ‖θ̂(z_N200) − θ̂(z_N400)‖, 1000 точек test | RMS ошибки на чистых ‖θ̂ − θ‖, 1000 точек test |")
for n in (6, 3):
    tag = NET_OF[n]
    for s in C.SEEDS_NET:
        net = nets[(tag, s)]
        a = net.predict(Z200[:, COLS[n]])
        b = net.predict(Z[:, COLS[n]])
        log(f"| {NAMES[tag]} | {s} | {rms(np.linalg.norm(a - b, axis=1)):.2e} | {rms(np.linalg.norm(b - th_test, axis=1)):.2e} |")
log()

log("## 5. Разложение ошибки основной сети (RMS ‖θ̂ − θ‖₂ по первым 20 точкам test, среднее по seed)\n")
clean20 = np.mean([rms(np.linalg.norm(nets[("main", s)].predict(Z[:NPTS]) - th, axis=1)) for s in C.SEEDS_NET])
noise_net = np.mean([net_tab[("main", s)]["mc"] for s in C.SEEDS_NET])
perp = np.mean([net_tab[("main", s)]["M_perp_F"] for s in C.SEEDS_NET])
log("| составляющая | величина |")
log(f"| ошибка сети на чистых данных (аппроксимация + сдвиг сетки) | {clean20:.2e} |")
log(f"| из неё: сдвиг сетки через точное обращение, ‖J⁺δz‖ | {rms(lin):.2e} |")
log(f"| шумовая ошибка сети, Монте-Карло | {noise_net:.2e} |")
log(f"| шумовой предел задачи, τ‖J⁺‖_F (= лучшая возможная линейная оценка) | {pred[6]:.2e} |")
log(f"| лишнее шумовое усиление сети, τ‖M P⊥‖_F | {perp:.2e} |")
log(f"| NLS на шумных данных, Монте-Карло | {rms(mc_nls[6]):.2e} |")
log(f"| сеть на шумных данных, ожидаемый итог ≈ sqrt(чистая² + шумовая²) | {np.hypot(clean20, noise_net):.2e} |")
log()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
x = np.arange(1, NPTS + 1)
ax.semilogy(x, [TAU * s["pinv_F"] for s in S[3]], "o-", label="τ‖J⁺‖_F, 3 датчика")
ax.semilogy(x, [TAU * s["pinv_F"] for s in S[6]], "s-", label="τ‖J⁺‖_F, 6 датчиков")
ax.semilogy(x, [rms(r) for r in mc_nls[3]], "o", mfc="none", ms=9, color="C0", label="NLS МК RMS, 3 датчика")
ax.semilogy(x, [rms(r) for r in mc_nls[6]], "s", mfc="none", ms=9, color="C1", label="NLS МК RMS, 6 датчиков")
ax.set_xlabel("точка test")
ax.set_ylabel("‖δθ‖₂ от шума τ = 0.01")
ax.set_title("прогноз против нелинейного МНК")
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(x, [s["kappa"] for s in S[3]], "o-", label="κ, 3 датчика")
ax.plot(x, [s["kappa"] for s in S[6]], "s-", label="κ, 6 датчиков")
ax.set_xlabel("точка test")
ax.set_ylabel("κ = σmax/σmin")
ax.set_title("число обусловленности")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "sensitivity.png", dpi=150)

(out / "sensitivity.md").write_text("\n".join(lines) + "\n")
print(f"сохранено: {out}")

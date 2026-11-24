import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data import load, normalize
from train import fit

LRS = {"sgd": (0.01, 0.03, 0.1, 0.3), "momentum": (0.01, 0.03, 0.1, 0.3), "adam": (3e-4, 1e-3, 3e-3, 1e-2)}
BATCHES = (32, 128)
EPOCHS = 100
NETS = {"main": ((6, 64, 64, 2), slice(0, 6)), "ctrl": ((3, 64, 64, 2), slice(0, 3))}
NAMES = {"main": "основная", "ctrl": "контрольная"}

out = C.RESULTS_DIR / "05_training"
out.mkdir(parents=True, exist_ok=True)
C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
lines = []

def log(s=""):
    print(s, flush=True)
    lines.append(s)

vtr, Ttr = load("train")
vval, Tval = load("val")
Ztr_all, Zval_all = normalize(vtr), normalize(vval)

log("Обучение и выбор модели по val\n")
log(f"train {len(Ttr)}, val {len(Tval)}; эпох ≤ {EPOCHS}; ранняя остановка = веса эпохи с минимальной val-потерей.")
log("Потеря L = 1/(2B)·Σ‖θ̂ − θ‖²; RMSE θ = sqrt(среднее по обеим компонентам); "
    "медиана k = медиана ‖k̂ − k‖₂/‖k‖₂ на val.\n")

selection = {}
for tag, (sizes, cols) in NETS.items():
    Ztr, Zval = Ztr_all[:, cols], Zval_all[:, cols]
    log(f"## Сетка, {NAMES[tag]} сеть {'-'.join(map(str, sizes))}, seed 11\n")
    log("| оптимизатор | шаг | батч | лучшая эпоха | val-потеря | val RMSE θ | val медиана k | расходимость | время, с |")
    rows = []
    for opt_name, lrs in LRS.items():
        for lr in lrs:
            for batch in BATCHES:
                r = fit(sizes, opt_name, lr, batch, 11, Ztr, Ttr, Zval, Tval, EPOCHS)
                rows.append(((opt_name, lr, batch), r))
                val = "-" if r["diverged"] and r["best_epoch"] == 0 else f"{r['val_loss']:.3e}"
                log(f"| {opt_name} | {lr:g} | {batch} | {r['best_epoch']} | {val} | {r['val_rmse_theta']:.2e} | "
                    f"{r['val_k_median']:.2e} | {'да' if r['diverged'] else ''} | {r['seconds']:.1f} |")
    valid = [(cfg, r) for cfg, r in rows if r["best_epoch"] > 0]
    (best_opt, best_lr, best_batch), best_r = min(valid, key=lambda cr: cr[1]["val_loss"])
    selection[tag] = {"optimizer": best_opt, "lr": best_lr, "batch": best_batch}
    log(f"\n**Выбрано для сети «{NAMES[tag]}»:** {best_opt}, шаг = {best_lr:g}, батч = {best_batch} "
        f"(val-потеря {best_r['val_loss']:.3e}, лучшая эпоха {best_r['best_epoch']}).\n")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for opt_name in LRS:
        cand = [(cfg, r) for cfg, r in valid if cfg[0] == opt_name]
        (o, lr, b), r = min(cand, key=lambda cr: cr[1]["val_loss"])
        ep = np.arange(1, len(r["history"]["val"]) + 1)
        line, = ax.semilogy(ep, r["history"]["val"], label=f"{o}, шаг={lr:g}, B={b}: val")
        ax.semilogy(ep, r["history"]["train"], ls="--", color=line.get_color(), lw=0.8)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("потеря (сплошная: val, штрих: train)")
    ax.set_title(f"{NAMES[tag]} сеть: лучшая конфигурация каждого оптимизатора (seed 11)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"optimizers_{tag}.png", dpi=150)
    plt.close(fig)

log("## Финальные сети (выбранная конфигурация, seed 11, 23, 37)\n")
log("| сеть | seed | лучшая эпоха | val-потеря | val RMSE θ | val медиана k | время, с |")
summary = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (tag, (sizes, cols)) in zip(axes, NETS.items()):
    cfg = selection[tag]
    vals = []
    for seed in C.SEEDS_NET:
        r = fit(sizes, cfg["optimizer"], cfg["lr"], cfg["batch"], seed,
                Ztr_all[:, cols], Ttr, Zval_all[:, cols], Tval, EPOCHS)
        assert not r["diverged"]
        r["net"].save(C.MODELS_DIR / f"{tag}_s{seed}.npz", best_epoch=r["best_epoch"],
                      optimizer=cfg["optimizer"], lr=cfg["lr"], batch=cfg["batch"])
        vals.append((r["val_loss"], r["val_rmse_theta"], r["val_k_median"]))
        log(f"| {NAMES[tag]} | {seed} | {r['best_epoch']} | {r['val_loss']:.3e} | {r['val_rmse_theta']:.2e} | "
            f"{r['val_k_median']:.2e} | {r['seconds']:.1f} |")
        ep = np.arange(1, len(r["history"]["val"]) + 1)
        line, = ax.semilogy(ep, r["history"]["val"], label=f"seed {seed}: val")
        ax.semilogy(ep, r["history"]["train"], ls="--", color=line.get_color(), lw=0.8)
        ax.axvline(r["best_epoch"], color=line.get_color(), ls=":", lw=0.8)
    vals = np.array(vals)
    summary[tag] = {"mean": vals.mean(0).tolist(), "std": vals.std(0, ddof=1).tolist()}
    ax.set_title(f"{NAMES[tag]}: {cfg['optimizer']}, шаг={cfg['lr']:g}, B={cfg['batch']}")
    ax.set_xlabel("эпоха")
    ax.legend(fontsize=8)
axes[0].set_ylabel("потеря (сплошная: val, штрих: train; точки: выбранная эпоха)")
fig.tight_layout()
fig.savefig(out / "final_curves.png", dpi=150)

log("\n| сеть | val-потеря | val RMSE θ | val медиана k |")
for tag, s in summary.items():
    log(f"| {NAMES[tag]} | {s['mean'][0]:.3e} ± {s['std'][0]:.1e} | {s['mean'][1]:.2e} ± {s['std'][1]:.1e} | "
        f"{s['mean'][2]:.2e} ± {s['std'][2]:.1e} |")
log("\n(среднее ± выборочное std по 3 seed). Веса сохранены в models/ и с этого момента заморожены;")
log("main_s11 - модель для самостоятельной проверки.\n")

(out / "selection.json").write_text(json.dumps({"selection": selection, "val_summary": summary}, indent=2))
(out / "training.md").write_text("\n".join(lines) + "\n")
print(f"сохранено: {out}")

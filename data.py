import numpy as np

import config as C
from solver import measure

def measurements(theta, N):
    k = np.exp(theta)
    return np.array([measure(k1, k2, N) for k1, k2 in k])

def sample_uniform(seed, n, lo=C.LNK_LO, hi=C.LNK_HI):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(n, 2))

def sample_ood(seed=C.SEED_OOD, n=C.N_OOD_SAMPLES):
    rng = np.random.default_rng(seed)
    kept, rejected = [], 0
    while len(kept) < n:
        th = rng.uniform(C.LNK_OOD_LO, C.LNK_OOD_HI, size=2)
        inside = (th >= C.LNK_LO) & (th <= C.LNK_HI)
        if inside.all():
            rejected += 1
        else:
            kept.append(th)
    return np.array(kept), rejected

def generate_all(verbose=True):
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    specs = {
        "train": (C.SEED_TRAIN, C.N_TRAIN_SAMPLES, C.N_TRAIN),
        "val": (C.SEED_VAL, C.N_VAL_SAMPLES, C.N_TRAIN),
        "test": (C.SEED_TEST, C.N_TEST_SAMPLES, C.N_TEST),
    }
    sets = {}
    for name, (seed, n, N) in specs.items():
        theta = sample_uniform(seed, n)
        v = measurements(theta, N)
        sets[name] = (v, theta)
        np.savez(C.DATA_DIR / f"{name}.npz", v=v, theta=theta, seed=seed, N=N)
        say(f"{name:10s} n = {n:5d}  seed = {seed}  N = {N}")

    m = sets["train"][0].mean(axis=0)
    d = sets["train"][0].std(axis=0)
    np.savez(C.DATA_DIR / "norm.npz", m=m, d=d)

    v_test, th_test = sets["test"]
    xi = np.random.default_rng(C.SEED_NOISE).standard_normal(v_test.shape)
    v_noisy = v_test + C.NOISE_LEVEL * d * xi
    np.savez(C.DATA_DIR / "test_noisy.npz", v=v_noisy, theta=th_test, xi=xi, seed=C.SEED_NOISE)
    say(f"{'test_noisy':10s} n = {len(v_noisy):5d}  seed шума = {C.SEED_NOISE}")

    th_ood, rejected = sample_ood()
    v_ood = measurements(th_ood, C.N_TEST)
    np.savez(C.DATA_DIR / "ood.npz", v=v_ood, theta=th_ood, rejected=rejected, seed=C.SEED_OOD)
    say(f"{'ood':10s} n = {len(v_ood):5d}  seed = {C.SEED_OOD}  отброшено пар = {rejected}")

    v_t200 = measurements(th_test, C.N_TRAIN)
    np.savez(C.DATA_DIR / "test_N200.npz", v=v_t200, theta=th_test, N=C.N_TRAIN)
    say(f"{'test_N200':10s} n = {len(v_t200):5d}  (только для диагностики)")

def load(name):
    f = np.load(C.DATA_DIR / f"{name}.npz")
    return f["v"], f["theta"]

def load_norm():
    f = np.load(C.DATA_DIR / "norm.npz")
    return f["m"], f["d"]

def normalize(v, m=None, d=None):
    if m is None:
        m, d = load_norm()
    return (v - m) / d

if __name__ == "__main__":
    generate_all()

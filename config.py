from pathlib import Path
import math

S = 2718                                  # выбрано один раз, 1 <= S <= 9999
A = 0.3523 + 0.01 * (S % 31)

N_TRAIN = 200
N_TEST = 400
X_MEAS = (0.25, 0.5, 0.75)

LNK_LO, LNK_HI = math.log(0.5), math.log(2.0)
LNK_OOD_LO, LNK_OOD_HI = math.log(0.25), math.log(4.0)  # выборка OOD и ограничения NLS

SEED_TRAIN = S
SEED_VAL = S + 10_000
SEED_TEST = S + 20_000
SEED_OOD = S + 30_000
SEED_SELFCHECK = S + 40_000
SEED_SOLVER_CHECK = S + 50_000
SEED_NOISE = 101
SEEDS_NET = (11, 23, 37)

N_TRAIN_SAMPLES, N_VAL_SAMPLES, N_TEST_SAMPLES, N_OOD_SAMPLES = 4000, 1000, 1000, 300
NOISE_LEVEL = 0.01                        # v_noisy = v + NOISE_LEVEL * d * xi

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

for stage in 01_solver_checks 02_data 03_mlp_checks 04_gradcheck 05_train 06_07_eval 08_sensitivity 09_selfcheck; do
    echo "=== $stage"
    start=$(date +%s)
    "$PY" "experiments/$stage.py" > /dev/null
    echo "    готово за $(( $(date +%s) - start )) с"
done
echo "Все этапы завершены, таблицы и рисунки лежат в results/."

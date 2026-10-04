#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(dirname "$ROOT")/.venv/bin/python"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$ROOT/logs/environment_analysis"
"$PY" "$ROOT/code/environment_analysis/run_jobs.py" > "$ROOT/logs/environment_analysis/jobs.log" 2>&1
"$PY" "$ROOT/code/environment_analysis/stratification.py" > "$ROOT/logs/environment_analysis/stratification.log" 2>&1
"$PY" "$ROOT/code/environment_analysis/process_and_monitor.py" > "$ROOT/logs/environment_analysis/process_monitor.log" 2>&1
bash "$ROOT/code/rebuild_narrative.sh"

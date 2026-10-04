#!/usr/bin/env bash
# Rebuild the revised narrative from existing results; does not train models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(dirname "$ROOT")/.venv/bin/python"
TECTONIC="${HOME}/.local/bin/tectonic"
if [[ ! -x "$PY" ]]; then PY=python3; fi
if [[ ! -x "$TECTONIC" ]]; then TECTONIC=tectonic; fi
mkdir -p "$ROOT/logs/narrative_rewrite"
"$PY" "$ROOT/code/rewrite_assets.py" "$ROOT" > "$ROOT/logs/narrative_rewrite/assets.log"
cd "$(dirname "$ROOT")"
"$PY" "$ROOT/code/fig_arch.py" > "$ROOT/logs/narrative_rewrite/figure_labels.log" 2>&1
"$PY" "$ROOT/code/fig_bench.py" 10 17 18 >> "$ROOT/logs/narrative_rewrite/figure_labels.log" 2>&1
"$PY" "$ROOT/code/fig_interp.py" 11 12 14 15 19 >> "$ROOT/logs/narrative_rewrite/figure_labels.log" 2>&1
"$PY" "$ROOT/code/environment_analysis/make_displays.py" > "$ROOT/logs/environment_analysis/displays.log" 2>&1
cd "$ROOT/paper"
"$TECTONIC" -X compile main.tex --outdir . --keep-logs --keep-intermediates > "$ROOT/logs/narrative_rewrite/english.log" 2>&1
"$TECTONIC" -X compile main_zh.tex --outdir . --keep-logs --keep-intermediates > "$ROOT/logs/narrative_rewrite/chinese.log" 2>&1
pdfinfo main.pdf | head -18
pdfinfo main_zh.pdf | head -18

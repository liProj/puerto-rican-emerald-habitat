#!/usr/bin/env bash
# Regenerate every derived artefact from whatever results exist: merge the CV result files,
# rebuild numbers.tex and the LaTeX tables, redraw all figures, compile the paper, write the
# Word report. Safe to re-run at any time.
set -u
cd /home/snakehand/Documents/mdpi_papers
PY=./.venv/bin/python

echo "== merging result files =="
$PY Birds/code/merge_results.py || exit 1
echo "== figures =="
$PY Birds/code/fig_data.py   1 2 3 16
$PY Birds/code/fig_arch.py
$PY Birds/code/fig_bench.py  5 6 7 8 9 10 17 18
$PY Birds/code/fig_interp.py 11 12 13 14 15 19 20
echo "== numbers and tables =="
$PY Birds/code/make_numbers.py || exit 1
$PY Birds/code/make_tables.py  || exit 1
echo "== paper (English) =="
( cd Birds/paper && tectonic -X compile main.tex --outdir . >/dev/null 2>&1 \
  && echo "   main.pdf: $(pdfinfo main.pdf | awk '/^Pages/{print $2}') pages" \
  || echo "   LaTeX FAILED" )
echo "== paper (Chinese) =="
( cd Birds/paper && tectonic -X compile main_zh.tex --outdir . >/dev/null 2>&1 \
  && echo "   main_zh.pdf: $(pdfinfo main_zh.pdf | awk '/^Pages/{print $2}') pages" \
  || echo "   Chinese LaTeX FAILED" )
echo "== word report =="
$PY Birds/code/make_report.py  && $PY Birds/code/make_report2.py
echo "== done =="
ls -la Birds/paper/main.pdf Birds/paper/main_zh.pdf Birds/report.docx 2>/dev/null

# Post-Hurricane Habitat Use of the Puerto Rican Emerald along Elevation and Vegetation Greenness Gradients and Implications for Monitoring

Research code, manuscript sources, figures, and retained analysis outputs for the Puerto Rican Emerald habitat-use study.

## Current manuscript

- [Latest English manuscript](paper/main.pdf) · [LaTeX source](paper/main.tex)
- [Current main figure (Figure 3)](paper/主图/1.pdf)
- Snapshot: 4 October 2026. The English manuscript includes the revised main figure and caption; the Simple Summary and author-declaration placeholder section have been removed.
- Chinese LaTeX files are retained as an earlier working translation and have not been synchronized with all English edits. No older Chinese PDF is included.

## Contents

| Directory | Contents |
| --- | --- |
| `code/` | Python and R models, validation, diagnostics, figure and table scripts |
| `code/environment_analysis/` | Environmental stratification, corrected occupancy, spatial predictions, monitoring-budget analyses |
| `paper/` | Manuscript sources, bibliography, tables, current English PDF and main figure |
| `figures/` | Retained PDF and PNG figures |
| `results/` | Retained benchmark, diagnostic, occupancy and environmental-analysis outputs |
| `results/author_channel/` | Original-analysis script, supplied NDVI data, retained tables and validation records |

Historical backups, local account/remote-access scripts, temporary renderings, Python caches, LaTeX build files, and large original fitted-model bundles are omitted.

## Setup and directory layout

Many archived scripts resolve paths relative to a directory named `Birds`. Clone with that local name and run analysis commands from its parent:

```bash
git clone https://github.com/liProj/puerto-rican-emerald-habitat.git Birds
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Birds/requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Shell wrappers require Bash. Python 3.12 was used by the retained source snapshot. The dependency file is an import-based list, not a recovered lockfile of the original training environment. Install a PyTorch build suited to your CPU/GPU when rerunning neural models.

Install R and the packages used by the scripts:

```r
install.packages(c("unmarked", "readr", "dplyr", "broom", "ggplot2", "scales", "tidyr", "sf"))
```

## Data inputs

The supplied NDVI analysis table is retained at:

```text
Birds/results/author_channel/data/Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv
```

The shared Python loader expects the original supplement layout. Place a copy of that table at:

```text
Birds/supp/birds-07-00055-s001/Riccordia_maugaeus_reproducibility_package/
  02 Riccordia_main_analysis/data/Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv
```

For EVI2 analyses and full regeneration of study-area/terrain figures, restore the remaining original supplement files, including the EVI2 CSV and SRTM GeoTIFF. They are not present in this repository. The retained terrain-feature cache is in `results/terrain_features.csv.gz`.

The local study records identify the original analysis as Ortiz-Andrade and Rojas-Perez, *Elevation and Vegetation Greenness Structure Post-Hurricane Resilience and Local Persistence of the Puerto Rican Emerald (Riccordia maugaeus)*, Birds 2026, 7, 55, DOI `10.3390/birds7030055`, with supplement `birds-07-00055-s001.zip`. See [data verification](data_check.md) and the [original-analysis notes](results/author_channel/README.txt) for provenance. These records are retained from the supplied project; a full supplement is still required for end-to-end reproduction.

## Check retained results

From the parent of `Birds`:

```bash
python Birds/code/environment_analysis/validate_outputs.py
python Birds/code/test_stats.py
```

The first command checks archived sample totals, prediction completeness, monitoring budgets and occupancy coefficient agreement. Its fold-separation check reads the saved validation record; it does not retrain models or independently reconstruct the complete train/test partitions.

## Main analysis entry points

After restoring the required input layout, run these commands from the parent of `Birds`:

```bash
python Birds/code/environment_analysis/rf_oof.py
Rscript Birds/code/environment_analysis/occupancy_predictions.R
python Birds/code/environment_analysis/stratification.py
python Birds/code/environment_analysis/process_and_monitor.py
python Birds/code/environment_analysis/validate_outputs.py
```

These commands rerun analyses and may replace retained outputs. The complete prediction benchmark is controlled by `code/run_main.py`, with spatial, random, temporal and ablation blocks; inspect `--help` before starting an expensive run. Historical orchestration scripts (`run_jobs.py`, `run_all.sh`, and `finalize_legacy.sh`) retain machine-specific paths from the original compute environment. Use the direct entry points above or adapt those paths to your environment.

## Compile the paper

The manuscript can be compiled from the retained figures and tables without retraining:

```bash
cd Birds/paper
tectonic -X compile main.tex --outdir . --keep-logs --keep-intermediates
```

The current main figure is `paper/主图/1.pdf`. `code/fig_arch.py` generates the older HERON architecture diagram, not this current overview figure. Older figure/table-generation scripts can regenerate historical captions; compile the current checked-in manuscript directly to preserve its edited text. Compiling the earlier Chinese translation additionally requires its specified Noto CJK fonts.

## Interpretation and validation scope

The current manuscript distinguishes checklist reporting, latent occupancy persistence, and predictive ranking for monitoring. It documents training-source overlap in the archived HERON configurations. Those neural results are retained as diagnostics and should not be treated as unbiased spatial-transfer estimates; use the current manuscript's separated-training conventional analyses for that interpretation.

This repository preserves the supplied research implementation and results. Upload preparation does not constitute a new full training run. See [repository validation](VALIDATION.md) for the checks performed for this snapshot.

## Attribution and licensing

The `results/author_channel/` material is attributed to the original analysis described above. No new license or redistribution rights are granted for third-party data, code, or illustrations by this upload. No open-source license has been selected for this repository.

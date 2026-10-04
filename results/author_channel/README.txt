RICCORDIA MAUGAEUS - PRIMARY CORRECTED-NDVI ANALYSIS
Revised manuscript reproducibility package

FILES
-----
Riccordia_analysis.R
    Main primary-analysis script.

data/Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv
    Analysis-ready checklist dataset with corrected NDVI.

HOW TO RUN
----------
1. Extract the complete reproducibility package.
2. Keep the data folder beside Riccordia_analysis.R.
3. Open Riccordia_analysis.R in RStudio.
4. Restart R and press Source.

When the extracted project path is short, outputs are created in:
outputs/

When the Windows project path is long, the script automatically uses:
~/Riccordia_out

This fallback prevents Windows path-length export errors.

MANUSCRIPT FIGURES GENERATED
----------------------------
Figure2_annual_observed_encounter_rate
Figure3_reporting_probability
Figure4_elevational_centroid
Figure5_local_extinction_elevation
Figure6_local_extinction_map

Each figure is exported in PNG and TIFF format.

Manuscript Figure 1 is reproduced separately in:
../01 Riccordia_preprocessing_environmental_covariates/03_study_area_topography_map/

The revised Figure 3, Figure 4, and Figure 5 code uses the enlarged axis text,
cleaner scales, and publication-style export settings used in the revised
manuscript. These are graphical changes only and do not alter fitted models,
predictions, centroids, or dynamic occupancy estimates.

A successful run ends with:
The corrected-NDVI analysis was reproduced successfully.

DATA SHA-256
------------
ccafe18d668eb586c360aefced919aae72366dbb61199b42a7ce0942a6a6c91f

SCOPE
-----
This script starts with the supplied analysis-ready integrated CSV. It
reproduces the primary statistical analyses, fitted models, model-selection
results, Supplementary Tables S1-S3, predictions, centroids, and manuscript
Figures 2-6.

Raw eBird downloading/filtering and the remote-sensing workflow used to derive
and attach corrected NDVI values are documented in Folder 01 and are outside
this primary R script.

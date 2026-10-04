# ============================================================
# Riccordia maugaeus
# Reproducibility analysis using corrected NDVI
# Reviewer release 1.2
# ============================================================

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

analysis_start_time <- Sys.time()

required_packages <- c(
  "readr",
  "dplyr",
  "broom",
  "ggplot2",
  "scales",
  "tidyr",
  "unmarked",
  "sf"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Install the following required packages before running this script: ",
      paste(
        missing_packages,
        collapse = ", "
      )
    )
  )
}

invisible(
  lapply(
    required_packages,
    library,
    character.only = TRUE
  )
)

get_script_directory <- function() {

  frame_files <- vapply(
    sys.frames(),
    function(frame) {
      if (is.null(frame$ofile)) {
        NA_character_
      } else {
        as.character(frame$ofile)
      }
    },
    character(1)
  )

  frame_files <- frame_files[
    !is.na(frame_files)
  ]

  if (length(frame_files) == 0) {
    return(
      normalizePath(
        getwd(),
        winslash = "/",
        mustWork = TRUE
      )
    )
  }

  dirname(
    normalizePath(
      frame_files[
        length(frame_files)
      ],
      winslash = "/",
      mustWork = TRUE
    )
  )
}

project_dir <- get_script_directory()

input_file <- file.path(
  project_dir,
  "data",
  "Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv"
)

preferred_output_dir <- file.path(
  project_dir,
  "outputs"
)

preferred_output_path_length <- nchar(
  preferred_output_dir
)

if (
  .Platform$OS.type == "windows" &&
    preferred_output_path_length > 150
) {
  output_dir <- file.path(
    path.expand("~"),
    "Riccordia_out"
  )

  cat(
    "\nThe project path is long. Outputs will be written to:\n",
    output_dir,
    "\n"
  )
} else {
  output_dir <- preferred_output_dir
}

for (
  directory_name in
  c(
    output_dir,
    file.path(
      output_dir,
      "tables"
    ),
    file.path(
      output_dir,
      "models"
    ),
    file.path(
      output_dir,
      "validation"
    ),
    file.path(
      output_dir,
      "figures"
    )
  )
) {
  dir.create(
    directory_name,
    recursive = TRUE,
    showWarnings = FALSE
  )
}


figure_dir <- file.path(
  output_dir,
  "figures"
)

dir.create(
  figure_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

if (!dir.exists(figure_dir)) {
  stop(
    paste0(
      "The figure output folder could not be created: ",
      figure_dir
    )
  )
}

figure_dir <- normalizePath(
  figure_dir,
  winslash = "/",
  mustWork = TRUE
)

save_reproducibility_figure <- function(
    plot_object,
    file_stem,
    width,
    height,
    dpi
) {

  png_path <- file.path(
    figure_dir,
    paste0(
      file_stem,
      ".png"
    )
  )

  tiff_path <- file.path(
    figure_dir,
    paste0(
      file_stem,
      ".tiff"
    )
  )

  ggplot2::ggsave(
    filename = basename(
      png_path
    ),
    path = dirname(
      png_path
    ),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = dpi,
    bg = "white"
  )

  if (!file.exists(png_path)) {
    grDevices::png(
      filename = png_path,
      width = width,
      height = height,
      units = "in",
      res = dpi
    )
    print(
      plot_object
    )
    grDevices::dev.off()
  }

  ggplot2::ggsave(
    filename = basename(
      tiff_path
    ),
    path = dirname(
      tiff_path
    ),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = dpi,
    compression = "lzw",
    bg = "white"
  )

  if (!file.exists(tiff_path)) {
    grDevices::tiff(
      filename = tiff_path,
      width = width,
      height = height,
      units = "in",
      res = dpi,
      compression = "lzw"
    )
    print(
      plot_object
    )
    grDevices::dev.off()
  }

  Sys.sleep(
    0.5
  )

  if (
    !file.exists(
      png_path
    ) ||
      !file.exists(
        tiff_path
      )
  ) {
    stop(
      paste0(
        "The figure could not be exported.\n",
        "PNG path: ",
        png_path,
        "\nTIFF path: ",
        tiff_path,
        "\nOutput folder writable: ",
        file.access(
          figure_dir,
          mode = 2
        ) == 0
      )
    )
  }

  invisible(
    list(
      png = png_path,
      tiff = tiff_path
    )
  )
}

if (!file.exists(input_file)) {
  stop(
    paste0(
      "Input file not found: ",
      input_file,
      "\nKeep the CSV inside the package's data folder."
    )
  )
}

cat(
  "\nStarting the corrected-NDVI reproducibility analysis.\n",
  "Project directory: ",
  project_dir,
  "\nInput file: ",
  input_file,
  "\nOutput directory: ",
  output_dir,
  "\n",
  sep = ""
)

# ============================================================
# 1. Read and standardize the corrected NDVI dataset
# ============================================================

dat0 <- readr::read_csv(
  input_file,
  show_col_types = FALSE
)

required_columns <- c(
  "SAMPLINGEVENTIDENTIFIER",
  "Date",
  "Period",
  "Presence",
  "Elevation_m",
  "NDVI_corrected",
  "Longitude",
  "Latitude",
  "Duration_min",
  "Distance_km"
)

stopifnot(
  all(required_columns %in% names(dat0))
)

dat <- dat0 |>
  dplyr::mutate(
    Date = as.Date(Date),
    Period = dplyr::case_when(
      Period %in% c(
        "Pre-Maria",
        "Pre-María"
      ) ~ "Pre-Maria",
      Period %in% c(
        "Inter-Hurricanes",
        "Inter-Huracanes"
      ) ~ "Inter-Hurricanes",
      Period == "Post-Fiona" ~ "Post-Fiona",
      TRUE ~ as.character(Period)
    ),
    Period = factor(
      Period,
      levels = c(
        "Pre-Maria",
        "Inter-Hurricanes",
        "Post-Fiona"
      )
    ),
    NDVI = NDVI_corrected
  ) |>
  dplyr::filter(
    Date >= as.Date("2013-01-01"),
    Date <= as.Date("2025-12-31")
  )

# ============================================================
# 2. Structural validation
# ============================================================

stopifnot(
  nrow(dat) == 89682,
  sum(dat$Presence == 1, na.rm = TRUE) == 5174,
  sum(dat$Presence == 0, na.rm = TRUE) == 84508,
  sum(is.na(dat$Period)) == 0,
  sum(is.na(dat$NDVI)) == 20706,
  sum(!is.na(dat$NDVI)) == 68976,
  all(
    dat$NDVI[
      !is.na(dat$NDVI)
    ] >= -1 &
      dat$NDVI[
        !is.na(dat$NDVI)
      ] <= 1
  )
)

dataset_summary <- dat |>
  dplyr::summarise(
    total_checklists = dplyr::n(),
    reported = sum(
      Presence == 1,
      na.rm = TRUE
    ),
    not_reported = sum(
      Presence == 0,
      na.rm = TRUE
    ),
    missing_corrected_NDVI = sum(
      is.na(NDVI)
    ),
    complete_corrected_NDVI = sum(
      !is.na(NDVI)
    ),
    minimum_corrected_NDVI = min(
      NDVI,
      na.rm = TRUE
    ),
    maximum_corrected_NDVI = max(
      NDVI,
      na.rm = TRUE
    )
  )

period_summary <- dat |>
  dplyr::group_by(
    Period
  ) |>
  dplyr::summarise(
    total_checklists = dplyr::n(),
    reported = sum(
      Presence == 1,
      na.rm = TRUE
    ),
    reporting_frequency_percent = 100 * mean(
      Presence == 1,
      na.rm = TRUE
    ),
    missing_corrected_NDVI = sum(
      is.na(NDVI)
    ),
    complete_corrected_NDVI = sum(
      !is.na(NDVI)
    ),
    corrected_NDVI_completeness_percent =
      100 * mean(
        !is.na(NDVI)
      ),
    .groups = "drop"
  )

print(
  dataset_summary,
  row.names = FALSE
)

print(
  period_summary,
  row.names = FALSE
)

readr::write_csv(
  dataset_summary,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_dataset_summary.csv"
  )
)

readr::write_csv(
  period_summary,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_period_summary.csv"
  )
)

# ============================================================
# 2.1 Supplementary Tables S2 and S3
# ============================================================

# Supplementary Table S2: descriptive NDVI summary by checklist-level
# reporting status. These values correspond to the formatted Table S2 supplied
# with the manuscript.

supplementary_table_s2 <- dat |>
  dplyr::filter(!is.na(NDVI)) |>
  dplyr::mutate(
    reporting_status = dplyr::case_when(
      Presence == 0 ~ "Not reported (0)",
      Presence == 1 ~ "Reported (1)",
      TRUE ~ NA_character_
    )
  ) |>
  dplyr::filter(!is.na(reporting_status)) |>
  dplyr::group_by(reporting_status) |>
  dplyr::summarise(
    n = dplyr::n(),
    mean_NDVI = mean(NDVI),
    SD = stats::sd(NDVI),
    median_NDVI = stats::median(NDVI),
    .groups = "drop"
  )

supplementary_table_s2_total <- dat |>
  dplyr::filter(!is.na(NDVI)) |>
  dplyr::summarise(
    reporting_status = "All complete-NDVI checklists",
    n = dplyr::n(),
    mean_NDVI = mean(NDVI),
    SD = stats::sd(NDVI),
    median_NDVI = stats::median(NDVI)
  )

supplementary_table_s2 <- dplyr::bind_rows(
  supplementary_table_s2 |>
    dplyr::mutate(
      reporting_status = factor(
        reporting_status,
        levels = c("Not reported (0)", "Reported (1)")
      )
    ) |>
    dplyr::arrange(reporting_status) |>
    dplyr::mutate(reporting_status = as.character(reporting_status)),
  supplementary_table_s2_total
)

stopifnot(
  supplementary_table_s2$n[1] == 65606,
  supplementary_table_s2$n[2] == 3370,
  supplementary_table_s2$n[3] == 68976
)

readr::write_csv(
  supplementary_table_s2,
  file.path(
    output_dir,
    "tables",
    "Supplementary_Table_S2_NDVI_by_reporting_status.csv"
  )
)

# Supplementary Table S3: NDVI availability by hurricane-related period.

supplementary_table_s3 <- period_summary |>
  dplyr::transmute(
    Period = as.character(Period),
    total_checklists = total_checklists,
    complete_NDVI = complete_corrected_NDVI,
    missing_NDVI = missing_corrected_NDVI,
    completeness_percent = corrected_NDVI_completeness_percent
  )

supplementary_table_s3_total <- dat |>
  dplyr::summarise(
    Period = "Total",
    total_checklists = dplyr::n(),
    complete_NDVI = sum(!is.na(NDVI)),
    missing_NDVI = sum(is.na(NDVI)),
    completeness_percent = 100 * mean(!is.na(NDVI))
  )

supplementary_table_s3 <- dplyr::bind_rows(
  supplementary_table_s3,
  supplementary_table_s3_total
)

stopifnot(
  supplementary_table_s3$total_checklists[1] == 14728,
  supplementary_table_s3$total_checklists[2] == 37701,
  supplementary_table_s3$total_checklists[3] == 37253,
  supplementary_table_s3$total_checklists[4] == 89682,
  supplementary_table_s3$complete_NDVI[4] == 68976,
  supplementary_table_s3$missing_NDVI[4] == 20706
)

readr::write_csv(
  supplementary_table_s3,
  file.path(
    output_dir,
    "tables",
    "Supplementary_Table_S3_NDVI_availability_by_period.csv"
  )
)

# ============================================================
# 3. Manuscript Figure 2: Annual observed encounter rate
# ============================================================

stopifnot(
  exists("dat"),
  exists("output_dir"),
  all(
    c(
      "Date",
      "Presence"
    ) %in% names(dat)
  ),
  nrow(dat) == 89682
)

# Manuscript Figure 2 uses the complete checklist dataset because it is a
# descriptive annual encounter-rate figure and does not require NDVI.

annual_encounter_rates_corrected_analysis <-
  dat |>
  dplyr::mutate(
    Year = as.integer(
      format(
        Date,
        "%Y"
      )
    )
  ) |>
  dplyr::group_by(
    Year
  ) |>
  dplyr::summarise(
    total_checklists =
      dplyr::n(),
    reported =
      sum(
        Presence == 1,
        na.rm = TRUE
      ),
    not_reported =
      sum(
        Presence == 0,
        na.rm = TRUE
      ),
    encounter_rate =
      reported /
      total_checklists,
    .groups = "drop"
  ) |>
  dplyr::rowwise() |>
  dplyr::mutate(
    lower_95 =
      stats::binom.test(
        x = reported,
        n = total_checklists,
        conf.level = 0.95
      )$conf.int[1],
    upper_95 =
      stats::binom.test(
        x = reported,
        n = total_checklists,
        conf.level = 0.95
      )$conf.int[2]
  ) |>
  dplyr::ungroup()

stopifnot(
  nrow(
    annual_encounter_rates_corrected_analysis
  ) == 13,
  min(
    annual_encounter_rates_corrected_analysis$
      Year
  ) == 2013,
  max(
    annual_encounter_rates_corrected_analysis$
      Year
  ) == 2025,
  sum(
    annual_encounter_rates_corrected_analysis$
      total_checklists
  ) == 89682,
  sum(
    annual_encounter_rates_corrected_analysis$
      reported
  ) == 5174,
  sum(
    annual_encounter_rates_corrected_analysis$
      not_reported
  ) == 84508,
  all(
    annual_encounter_rates_corrected_analysis$
      lower_95 <=
      annual_encounter_rates_corrected_analysis$
        encounter_rate
  ),
  all(
    annual_encounter_rates_corrected_analysis$
      upper_95 >=
      annual_encounter_rates_corrected_analysis$
        encounter_rate
  )
)

minimum_corrected_analysis_encounter_year <-
  annual_encounter_rates_corrected_analysis |>
  dplyr::slice_min(
    encounter_rate,
    n = 1,
    with_ties = FALSE
  ) |>
  dplyr::pull(
    Year
  )

stopifnot(
  minimum_corrected_analysis_encounter_year ==
    2019
)

cat(
  "\nAnnual observed encounter rates used in manuscript Figure 2:\n"
)

print(
  annual_encounter_rates_corrected_analysis,
  row.names = FALSE
)

readr::write_csv(
  annual_encounter_rates_corrected_analysis,
  file.path(
    output_dir,
    "tables",
    "annual_observed_encounter_rates_2013_2025.csv"
  )
)

dir.create(
  file.path(
    output_dir,
    "figures"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

hurricane_positions_corrected_analysis <-
  c(
    Maria =
      2017 + 262 / 365,
    Fiona =
      2022 + 260 / 365
  )

figure1_corrected_analysis <-
  ggplot2::ggplot(
    annual_encounter_rates_corrected_analysis,
    ggplot2::aes(
      x = Year,
      y = encounter_rate
    )
  ) +
  ggplot2::geom_vline(
    xintercept =
      hurricane_positions_corrected_analysis[
        "Maria"
      ],
    linetype = "dashed",
    color = "#B22222",
    linewidth = 0.8
  ) +
  ggplot2::geom_vline(
    xintercept =
      hurricane_positions_corrected_analysis[
        "Fiona"
      ],
    linetype = "dashed",
    color = "#20B2AA",
    linewidth = 0.8
  ) +
  ggplot2::annotate(
    "text",
    x = 2017.5,
    y =
      max(
        annual_encounter_rates_corrected_analysis$
          encounter_rate
      ) * 0.95,
    label =
      "Hurricane María (2017)",
    color = "#B22222",
    angle = 90,
    vjust = -0.5,
    fontface = "bold",
    size = 3.5
  ) +
  ggplot2::annotate(
    "text",
    x = 2022.5,
    y =
      max(
        annual_encounter_rates_corrected_analysis$
          encounter_rate
      ) * 0.95,
    label =
      "Hurricane Fiona (2022)",
    color = "#20B2AA",
    angle = 90,
    vjust = -0.5,
    fontface = "bold",
    size = 3.5
  ) +
  ggplot2::geom_errorbar(
    ggplot2::aes(
      ymin = lower_95,
      ymax = upper_95
    ),
    width = 0.15,
    linewidth = 0.7,
    color = "#4A5568"
  ) +
  ggplot2::geom_line(
    linewidth = 1,
    color = "#1A365D"
  ) +
  ggplot2::geom_point(
    ggplot2::aes(
      color = Year
    ),
    size = 4
  ) +
  ggplot2::scale_color_viridis_c(
    option = "mako",
    end = 0.8,
    guide = "none"
  ) +
  ggplot2::scale_x_continuous(
    breaks = seq(
      2013,
      2025,
      by = 2
    ),
    limits = c(
      2012.7,
      2025.5
    ),
    expand =
      ggplot2::expansion(
        mult = c(
          0,
          0
        )
      )
  ) +
  ggplot2::scale_y_continuous(
    limits = c(
      0.03,
      max(
        annual_encounter_rates_corrected_analysis$
          upper_95
      ) * 1.03
    ),
    breaks = seq(
      0.04,
      0.08,
      by = 0.02
    ),
    labels =
      scales::label_number(
        accuracy = 0.01
      ),
    expand =
      ggplot2::expansion(
        mult = c(
          0,
          0.02
        )
      )
  ) +
  ggplot2::labs(
    x = "Year",
    y =
      "Observed Encounter Rate (Presence Proportion)"
  ) +
  ggplot2::theme_classic(
    base_size = 14
  ) +
  ggplot2::theme(
    axis.title =
      ggplot2::element_text(
        face = "bold",
        color = "black"
      ),
    axis.text =
      ggplot2::element_text(
        color = "black"
      ),
    axis.line =
      ggplot2::element_line(
        linewidth = 0.6,
        color = "black"
      ),
    plot.margin =
      ggplot2::margin(
        t = 8,
        r = 12,
        b = 8,
        l = 8
      )
  )

print(
  figure1_corrected_analysis
)

figure1_paths <- save_reproducibility_figure(
  plot_object =
    figure1_corrected_analysis,
  file_stem =
    "Figure2_annual_observed_encounter_rate",
  width = 7.5,
  height = 4.8,
  dpi = 600
)

figure1_corrected_png <-
  figure1_paths$png

figure1_corrected_tiff <-
  figure1_paths$tiff

cat(
  "\nManuscript Figure 2 was generated and exported successfully.\n"
)

# ============================================================
# 4. Complete-case dataset for equivalent GLMs
# ============================================================

dat_ndvi_corrected <- dat |>
  dplyr::filter(
    !is.na(NDVI)
  ) |>
  dplyr::mutate(
    Elevation_100m = Elevation_m / 100
  )

stopifnot(
  nrow(dat_ndvi_corrected) == 68976,
  all(
    stats::complete.cases(
      dat_ndvi_corrected[
        c(
          "Presence",
          "Period",
          "Elevation_100m",
          "NDVI",
          "Duration_min",
          "Distance_km"
        )
      ]
    )
  ),
  levels(dat_ndvi_corrected$Period)[1] ==
    "Pre-Maria"
)

# ============================================================
# 5. Fit the same five GLM candidates used previously
# ============================================================

m0 <- stats::glm(
  Presence ~ 1,
  data = dat_ndvi_corrected,
  family = stats::binomial()
)

m1 <- stats::glm(
  Presence ~ Period,
  data = dat_ndvi_corrected,
  family = stats::binomial()
)

m2 <- stats::glm(
  Presence ~ Period +
    Duration_min +
    Distance_km,
  data = dat_ndvi_corrected,
  family = stats::binomial()
)

m3 <- stats::glm(
  Presence ~ Period +
    Elevation_100m +
    NDVI +
    Duration_min +
    Distance_km,
  data = dat_ndvi_corrected,
  family = stats::binomial()
)

m4 <- stats::glm(
  Presence ~ Period * Elevation_100m +
    NDVI +
    Duration_min +
    Distance_km,
  data = dat_ndvi_corrected,
  family = stats::binomial()
)

glm_models_corrected_ndvi <- list(
  m0_intercept_only = m0,
  m1_period = m1,
  m2_period_effort = m2,
  m3_environment_effort_additive = m3,
  m4_period_elevation_interaction = m4
)

glm_observation_counts <- data.frame(
  model = names(
    glm_models_corrected_ndvi
  ),
  observations = vapply(
    glm_models_corrected_ndvi,
    stats::nobs,
    numeric(1)
  ),
  stringsAsFactors = FALSE
)

stopifnot(
  all(
    glm_observation_counts$observations ==
      68976
  )
)

# ============================================================
# 6. Model selection
# ============================================================

glm_model_selection_corrected_ndvi <- data.frame(
  model = names(
    glm_models_corrected_ndvi
  ),
  AIC = vapply(
    glm_models_corrected_ndvi,
    stats::AIC,
    numeric(1)
  ),
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(
    AIC
  ) |>
  dplyr::mutate(
    delta_AIC = AIC - min(AIC),
    akaike_weight = exp(
      -0.5 * delta_AIC
    ) / sum(
      exp(
        -0.5 * delta_AIC
      )
    ),
    rank = dplyr::row_number()
  ) |>
  dplyr::select(
    rank,
    model,
    AIC,
    delta_AIC,
    akaike_weight
  )

print(
  glm_observation_counts,
  row.names = FALSE
)

print(
  glm_model_selection_corrected_ndvi,
  row.names = FALSE
)

# ============================================================
# 7. Export coefficients for all candidate models
# ============================================================

glm_coefficients_corrected_ndvi <- dplyr::bind_rows(
  lapply(
    names(glm_models_corrected_ndvi),
    function(model_name) {
      broom::tidy(
        glm_models_corrected_ndvi[[model_name]],
        conf.int = TRUE
      ) |>
        dplyr::mutate(
          model = model_name,
          .before = 1
        )
    }
  )
)

readr::write_csv(
  glm_observation_counts,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_glm_observation_counts.csv"
  )
)

readr::write_csv(
  glm_model_selection_corrected_ndvi,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_glm_model_selection.csv"
  )
)

readr::write_csv(
  glm_coefficients_corrected_ndvi,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_glm_coefficients.csv"
  )
)

base::saveRDS(
  glm_models_corrected_ndvi,
  file.path(
    output_dir,
    "models",
    "corrected_ndvi_glm_candidate_models.rds"
  )
)

cat("\nEquivalent GLMs with corrected NDVI were fitted successfully.\n")

# ============================================================
# 8. Detailed results for the best GLM with corrected NDVI
# ============================================================

stopifnot(
  exists("m0"),
  exists("m4"),
  exists("glm_model_selection_corrected_ndvi"),
  exists("output_dir"),
  inherits(m0, "glm"),
  inherits(m4, "glm"),
  stats::nobs(m0) == 68976,
  stats::nobs(m4) == 68976,
  glm_model_selection_corrected_ndvi$model[1] ==
    "m4_period_elevation_interaction"
)

glm_best_corrected_ndvi <- m4

cat(
  "\nSummary of the best GLM with corrected NDVI:\n"
)

print(
  summary(glm_best_corrected_ndvi)
)

# ============================================================
# 8.1 Coefficients, Wald confidence intervals, and odds ratios
# ============================================================

glm_best_corrected_coefficients <- broom::tidy(
  glm_best_corrected_ndvi
) |>
  dplyr::mutate(
    conf_low_logit = estimate - 1.96 * std.error,
    conf_high_logit = estimate + 1.96 * std.error,
    odds_ratio = exp(estimate),
    odds_ratio_conf_low_95 = exp(conf_low_logit),
    odds_ratio_conf_high_95 = exp(conf_high_logit),
    percent_change_in_odds = 100 * (
      odds_ratio - 1
    )
  ) |>
  dplyr::select(
    term,
    estimate,
    std.error,
    statistic,
    p.value,
    conf_low_logit,
    conf_high_logit,
    odds_ratio,
    odds_ratio_conf_low_95,
    odds_ratio_conf_high_95,
    percent_change_in_odds
  )

expected_terms_corrected <- c(
  "(Intercept)",
  "PeriodInter-Hurricanes",
  "PeriodPost-Fiona",
  "Elevation_100m",
  "NDVI",
  "Duration_min",
  "Distance_km",
  "PeriodInter-Hurricanes:Elevation_100m",
  "PeriodPost-Fiona:Elevation_100m"
)

stopifnot(
  all(
    expected_terms_corrected %in%
      glm_best_corrected_coefficients$term
  )
)

cat(
  "\nCoefficients and odds ratios:\n"
)

print(
  glm_best_corrected_coefficients,
  n = Inf
)

# ============================================================
# 8.2 Total elevation effect within each period
# ============================================================

glm_corrected_coef <- stats::coef(
  glm_best_corrected_ndvi
)

glm_corrected_vcov <- stats::vcov(
  glm_best_corrected_ndvi
)

find_interaction_term <- function(period_coefficient) {

  coefficient_names <- names(
    glm_corrected_coef
  )

  candidates <- coefficient_names[
    grepl(
      "Elevation_100m",
      coefficient_names,
      fixed = TRUE
    ) &
      grepl(
        period_coefficient,
        coefficient_names,
        fixed = TRUE
      )
  ]

  stopifnot(
    length(candidates) == 1
  )

  candidates
}

inter_hurricanes_elevation_interaction <-
  find_interaction_term(
    "PeriodInter-Hurricanes"
  )

post_fiona_elevation_interaction <-
  find_interaction_term(
    "PeriodPost-Fiona"
  )

calculate_corrected_elevation_effect <- function(
    period_name,
    interaction_term = NULL
) {

  base_term <- "Elevation_100m"

  if (is.null(interaction_term)) {

    estimate <- glm_corrected_coef[
      base_term
    ]

    variance <- glm_corrected_vcov[
      base_term,
      base_term
    ]

  } else {

    estimate <-
      glm_corrected_coef[
        base_term
      ] +
      glm_corrected_coef[
        interaction_term
      ]

    variance <-
      glm_corrected_vcov[
        base_term,
        base_term
      ] +
      glm_corrected_vcov[
        interaction_term,
        interaction_term
      ] +
      2 * glm_corrected_vcov[
        base_term,
        interaction_term
      ]
  }

  standard_error <- sqrt(
    variance
  )

  z_value <- estimate /
    standard_error

  p_value <- 2 * stats::pnorm(
    abs(z_value),
    lower.tail = FALSE
  )

  conf_low_logit <- estimate -
    1.96 * standard_error

  conf_high_logit <- estimate +
    1.96 * standard_error

  odds_ratio <- exp(
    estimate
  )

  data.frame(
    Period = period_name,
    estimate_log_odds_per_100m =
      unname(estimate),
    standard_error =
      unname(standard_error),
    z_value =
      unname(z_value),
    p_value =
      unname(p_value),
    conf_low_logit =
      unname(conf_low_logit),
    conf_high_logit =
      unname(conf_high_logit),
    odds_ratio_per_100m =
      unname(odds_ratio),
    odds_ratio_conf_low_95 =
      unname(
        exp(conf_low_logit)
      ),
    odds_ratio_conf_high_95 =
      unname(
        exp(conf_high_logit)
      ),
    percent_change_in_odds_per_100m =
      unname(
        100 * (
          odds_ratio - 1
        )
      ),
    stringsAsFactors = FALSE
  )
}

glm_corrected_elevation_effects_by_period <-
  dplyr::bind_rows(
    calculate_corrected_elevation_effect(
      period_name = "Pre-Maria"
    ),
    calculate_corrected_elevation_effect(
      period_name = "Inter-Hurricanes",
      interaction_term =
        inter_hurricanes_elevation_interaction
    ),
    calculate_corrected_elevation_effect(
      period_name = "Post-Fiona",
      interaction_term =
        post_fiona_elevation_interaction
    )
  )

cat(
  "\nTotal elevation effect by period:\n"
)

print(
  glm_corrected_elevation_effects_by_period,
  row.names = FALSE
)

# ============================================================
# 8.3 Effect of a 0.1-unit increase in corrected NDVI
# ============================================================

ndvi_term <- "NDVI"

ndvi_estimate_01 <-
  0.1 * glm_corrected_coef[
    ndvi_term
  ]

ndvi_standard_error_01 <-
  0.1 * sqrt(
    glm_corrected_vcov[
      ndvi_term,
      ndvi_term
    ]
  )

ndvi_z_value <- ndvi_estimate_01 /
  ndvi_standard_error_01

ndvi_p_value <- 2 * stats::pnorm(
  abs(ndvi_z_value),
  lower.tail = FALSE
)

ndvi_conf_low_logit_01 <-
  ndvi_estimate_01 -
  1.96 * ndvi_standard_error_01

ndvi_conf_high_logit_01 <-
  ndvi_estimate_01 +
  1.96 * ndvi_standard_error_01

ndvi_odds_ratio_01 <- exp(
  ndvi_estimate_01
)

glm_corrected_ndvi_effect_01 <- data.frame(
  NDVI_increment = 0.1,
  estimate_log_odds =
    unname(ndvi_estimate_01),
  standard_error =
    unname(ndvi_standard_error_01),
  z_value =
    unname(ndvi_z_value),
  p_value =
    unname(ndvi_p_value),
  odds_ratio =
    unname(ndvi_odds_ratio_01),
  odds_ratio_conf_low_95 =
    unname(
      exp(
        ndvi_conf_low_logit_01
      )
    ),
  odds_ratio_conf_high_95 =
    unname(
      exp(
        ndvi_conf_high_logit_01
      )
    ),
  percent_change_in_odds =
    unname(
      100 * (
        ndvi_odds_ratio_01 - 1
      )
    ),
  stringsAsFactors = FALSE
)

cat(
  "\nEffect of a 0.1-unit increase in corrected NDVI:\n"
)

print(
  glm_corrected_ndvi_effect_01,
  row.names = FALSE
)

# ============================================================
# 8.4 McFadden pseudo-R-squared
# ============================================================

stopifnot(
  stats::nobs(
    glm_best_corrected_ndvi
  ) ==
    stats::nobs(
      m0
    )
)

glm_corrected_mcfadden_r2 <-
  1 - as.numeric(
    stats::logLik(
      glm_best_corrected_ndvi
    ) /
      stats::logLik(
        m0
      )
  )

glm_corrected_mcfadden_summary <- data.frame(
  metric = "McFadden pseudo-R2",
  value = glm_corrected_mcfadden_r2,
  value_rounded =
    round(
      glm_corrected_mcfadden_r2,
      3
    ),
  percent_rounded =
    round(
      100 * glm_corrected_mcfadden_r2,
      1
    ),
  stringsAsFactors = FALSE
)

cat(
  "\nMcFadden pseudo-R-squared:\n"
)

print(
  glm_corrected_mcfadden_summary,
  row.names = FALSE
)

# ============================================================
# 8.5 Export the corrected-GLM results
# ============================================================

readr::write_csv(
  glm_best_corrected_coefficients,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_m4_coefficients_odds_ratios.csv"
  )
)

readr::write_csv(
  glm_corrected_elevation_effects_by_period,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_m4_elevation_effects_by_period.csv"
  )
)

readr::write_csv(
  glm_corrected_ndvi_effect_01,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_m4_effect_of_0_1_NDVI.csv"
  )
)

readr::write_csv(
  glm_corrected_mcfadden_summary,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_m4_mcfadden_r2.csv"
  )
)


base::saveRDS(
  glm_best_corrected_ndvi,
  file.path(
    output_dir,
    "models",
    "corrected_ndvi_best_glm_m4.rds"
  )
)

stopifnot(
  is.finite(
    glm_corrected_mcfadden_r2
  ),
  glm_corrected_mcfadden_r2 > 0,
  glm_corrected_mcfadden_r2 < 1,
  nrow(
    glm_corrected_elevation_effects_by_period
  ) == 3,
  nrow(
    glm_corrected_ndvi_effect_01
  ) == 1
)

cat(
  "\nCorrected-NDVI m4 coefficients and derived effects were exported successfully.\n"
)

# ============================================================
# 9. Corrected-NDVI predictions across elevation and Figure 2
# ============================================================

stopifnot(
  exists("glm_best_corrected_ndvi"),
  exists("dat_ndvi_corrected"),
  exists("output_dir"),
  inherits(
    glm_best_corrected_ndvi,
    "glm"
  ),
  stats::nobs(
    glm_best_corrected_ndvi
  ) == 68976
)

# ============================================================
# 9.1 Mean covariate values used for prediction
# ============================================================

corrected_prediction_covariate_means <-
  dat_ndvi_corrected |>
  dplyr::summarise(
    mean_corrected_NDVI = mean(
      NDVI,
      na.rm = TRUE
    ),
    mean_Duration_min = mean(
      Duration_min,
      na.rm = TRUE
    ),
    mean_Distance_km = mean(
      Distance_km,
      na.rm = TRUE
    )
  )

cat(
  "\nMean covariate values used for corrected-NDVI predictions:\n"
)

print(
  corrected_prediction_covariate_means,
  row.names = FALSE
)

# ============================================================
# 9.2 Prediction grid
# ============================================================

corrected_prediction_grid <- expand.grid(
  Elevation_m = seq(
    0,
    1300,
    by = 10
  ),
  Period = levels(
    dat_ndvi_corrected$Period
  ),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
) |>
  dplyr::mutate(
    Period = factor(
      Period,
      levels = levels(
        dat_ndvi_corrected$Period
      )
    ),
    Elevation_100m = Elevation_m / 100,
    NDVI =
      corrected_prediction_covariate_means$
        mean_corrected_NDVI,
    Duration_min =
      corrected_prediction_covariate_means$
        mean_Duration_min,
    Distance_km =
      corrected_prediction_covariate_means$
        mean_Distance_km
  )

stopifnot(
  nrow(
    corrected_prediction_grid
  ) == 393,
  min(
    corrected_prediction_grid$Elevation_m
  ) == 0,
  max(
    corrected_prediction_grid$Elevation_m
  ) == 1300,
  length(
    unique(
      corrected_prediction_grid$Period
    )
  ) == 3
)

# ============================================================
# 9.3 Predicted probabilities and 95% confidence intervals
# ============================================================

corrected_prediction_link <- stats::predict(
  glm_best_corrected_ndvi,
  newdata = corrected_prediction_grid,
  type = "link",
  se.fit = TRUE
)

glm_predictions_corrected_ndvi <-
  corrected_prediction_grid |>
  dplyr::mutate(
    predicted_logit =
      corrected_prediction_link$fit,
    standard_error =
      corrected_prediction_link$se.fit,
    lower_logit =
      predicted_logit -
      1.96 * standard_error,
    upper_logit =
      predicted_logit +
      1.96 * standard_error,
    predicted_probability =
      stats::plogis(
        predicted_logit
      ),
    lower_95 =
      stats::plogis(
        lower_logit
      ),
    upper_95 =
      stats::plogis(
        upper_logit
      )
  )

stopifnot(
  all(
    is.finite(
      glm_predictions_corrected_ndvi$
        predicted_probability
    )
  ),
  all(
    glm_predictions_corrected_ndvi$
      predicted_probability >= 0
  ),
  all(
    glm_predictions_corrected_ndvi$
      predicted_probability <= 1
  ),
  all(
    glm_predictions_corrected_ndvi$
      lower_95 <=
      glm_predictions_corrected_ndvi$
        predicted_probability
  ),
  all(
    glm_predictions_corrected_ndvi$
      upper_95 >=
      glm_predictions_corrected_ndvi$
        predicted_probability
  )
)

# ============================================================
# 9.4 Reference predictions at 100 and 900 m
# ============================================================

corrected_prediction_check <-
  glm_predictions_corrected_ndvi |>
  dplyr::filter(
    Elevation_m %in% c(
      100,
      900
    )
  ) |>
  dplyr::mutate(
    predicted_percent =
      100 * predicted_probability,
    lower_95_percent =
      100 * lower_95,
    upper_95_percent =
      100 * upper_95
  ) |>
  dplyr::select(
    Period,
    Elevation_m,
    predicted_probability,
    lower_95,
    upper_95,
    predicted_percent,
    lower_95_percent,
    upper_95_percent
  )

cat(
  "\nCorrected-NDVI predictions at 100 and 900 m:\n"
)

print(
  corrected_prediction_check,
  row.names = FALSE
)

# ============================================================
# 9.5 Export prediction tables
# ============================================================

readr::write_csv(
  corrected_prediction_covariate_means,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_prediction_covariate_means.csv"
  )
)

readr::write_csv(
  glm_predictions_corrected_ndvi,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_glm_predictions_elevation_period.csv"
  )
)

readr::write_csv(
  corrected_prediction_check,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_predictions_100m_900m.csv"
  )
)

# ============================================================
# 9.6 Manuscript Figure 3 with corrected NDVI
# ============================================================

dir.create(
  file.path(
    output_dir,
    "figures"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

corrected_period_colors <- c(
  "Pre-Maria" = "#0072B2",
  "Inter-Hurricanes" = "#D55E00",
  "Post-Fiona" = "#009E73"
)

corrected_period_linetypes <- c(
  "Pre-Maria" = "solid",
  "Inter-Hurricanes" = "dashed",
  "Post-Fiona" = "dotdash"
)

figure2_corrected_ndvi <-
  ggplot2::ggplot(
    data = glm_predictions_corrected_ndvi,
    mapping = ggplot2::aes(
      x = Elevation_m,
      y = predicted_probability,
      group = Period,
      color = Period,
      fill = Period,
      linetype = Period
    )
  ) +
  ggplot2::geom_ribbon(
    mapping = ggplot2::aes(
      ymin = lower_95,
      ymax = upper_95
    ),
    alpha = 0.12,
    color = NA
  ) +
  ggplot2::geom_line(
    linewidth = 1.2
  ) +
  ggplot2::scale_color_manual(
    values = corrected_period_colors
  ) +
  ggplot2::scale_fill_manual(
    values = corrected_period_colors
  ) +
  ggplot2::scale_linetype_manual(
    values = corrected_period_linetypes
  ) +
  ggplot2::scale_x_continuous(
    limits = c(0, 1300),
    breaks = c(0, 250, 500, 750, 1000, 1250),
    labels = scales::label_number(
      accuracy = 1,
      big.mark = ","
    ),
    expand = ggplot2::expansion(
      mult = c(0, 0.01)
    )
  ) +
  ggplot2::scale_y_continuous(
    limits = c(
      0,
      min(
        1,
        max(
          glm_predictions_corrected_ndvi$upper_95,
          na.rm = TRUE
        ) * 1.02
      )
    ),
    breaks = seq(0, 1, by = 0.10),
    labels = scales::label_percent(
      accuracy = 1
    ),
    expand = ggplot2::expansion(
      mult = c(0, 0.02)
    )
  ) +
  ggplot2::labs(
    x = "Elevation (m a.s.l.)",
    y = "Predicted reporting probability",
    color = NULL,
    fill = NULL,
    linetype = NULL
  ) +
  ggplot2::guides(
    fill = "none",
    color = ggplot2::guide_legend(
      nrow = 1,
      byrow = TRUE,
      override.aes = list(
        linewidth = 1.6
      )
    )
  ) +
  ggplot2::theme_classic(
    base_size = 14
  ) +
  ggplot2::theme(
    legend.position = "top",
    legend.text = ggplot2::element_text(
      size = 13,
      color = "black"
    ),
    axis.title.x = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black",
      margin = ggplot2::margin(t = 10)
    ),
    axis.title.y = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black",
      margin = ggplot2::margin(r = 10)
    ),
    axis.text.x = ggplot2::element_text(
      size = 13.5,
      color = "black"
    ),
    axis.text.y = ggplot2::element_text(
      size = 13.5,
      color = "black"
    ),
    axis.line = ggplot2::element_line(
      linewidth = 0.7,
      color = "black"
    ),
    axis.ticks = ggplot2::element_line(
      linewidth = 0.6,
      color = "black"
    ),
    axis.ticks.length = grid::unit(
      0.18,
      "cm"
    ),
    plot.margin = ggplot2::margin(
      t = 8,
      r = 16,
      b = 12,
      l = 12
    )
  )

print(
  figure2_corrected_ndvi
)

figure2_paths <- save_reproducibility_figure(
  plot_object =
    figure2_corrected_ndvi,
  file_stem =
    "Figure3_reporting_probability",
  width = 8,
  height = 5.5,
  dpi = 600
)

figure2_corrected_png <-
  figure2_paths$png

figure2_corrected_tiff <-
  figure2_paths$tiff

cat(
  "\nCorrected-NDVI predictions and manuscript Figure 3 were exported successfully.\n"
)

# ============================================================
# 10. Elevational centroid analysis with corrected NDVI
# ============================================================

stopifnot(
  exists("glm_predictions_corrected_ndvi"),
  exists("output_dir"),
  all(
    c(
      "Period",
      "Elevation_m",
      "predicted_probability"
    ) %in%
      names(
        glm_predictions_corrected_ndvi
      )
  )
)

# ============================================================
# 10.1 Verify the elevation grid
# ============================================================

corrected_centroid_grid_check <-
  glm_predictions_corrected_ndvi |>
  dplyr::summarise(
    minimum_elevation_m = min(
      Elevation_m,
      na.rm = TRUE
    ),
    maximum_elevation_m = max(
      Elevation_m,
      na.rm = TRUE
    ),
    elevation_increment_m = min(
      diff(
        sort(
          unique(
            Elevation_m
          )
        )
      )
    ),
    number_of_elevation_values =
      dplyr::n_distinct(
        Elevation_m
      ),
    number_of_periods =
      dplyr::n_distinct(
        Period
      )
  )

stopifnot(
  corrected_centroid_grid_check$
    minimum_elevation_m == 0,
  corrected_centroid_grid_check$
    maximum_elevation_m == 1300,
  corrected_centroid_grid_check$
    elevation_increment_m == 10,
  corrected_centroid_grid_check$
    number_of_elevation_values == 131,
  corrected_centroid_grid_check$
    number_of_periods == 3
)

cat(
  "\nPrediction grid used for the corrected-NDVI centroid:\n"
)

print(
  corrected_centroid_grid_check,
  row.names = FALSE
)

# ============================================================
# 10.2 Probability-weighted elevational centroids
# ============================================================

corrected_elevational_centroids <-
  glm_predictions_corrected_ndvi |>
  dplyr::group_by(
    Period
  ) |>
  dplyr::summarise(
    predicted_centroid_m =
      stats::weighted.mean(
        x = Elevation_m,
        w = predicted_probability,
        na.rm = TRUE
      ),
    .groups = "drop"
  ) |>
  dplyr::mutate(
    Period = factor(
      as.character(
        Period
      ),
      levels = c(
        "Pre-Maria",
        "Inter-Hurricanes",
        "Post-Fiona"
      )
    )
  ) |>
  dplyr::arrange(
    Period
  )

corrected_pre_maria_centroid <-
  corrected_elevational_centroids |>
  dplyr::filter(
    Period == "Pre-Maria"
  ) |>
  dplyr::pull(
    predicted_centroid_m
  )

stopifnot(
  length(
    corrected_pre_maria_centroid
  ) == 1,
  is.finite(
    corrected_pre_maria_centroid
  )
)

corrected_elevational_centroids <-
  corrected_elevational_centroids |>
  dplyr::mutate(
    change_from_pre_maria_m =
      predicted_centroid_m -
      corrected_pre_maria_centroid,
    predicted_centroid_m_rounded =
      round(
        predicted_centroid_m,
        1
      ),
    change_from_pre_maria_m_rounded =
      round(
        change_from_pre_maria_m,
        1
      )
  )

stopifnot(
  nrow(
    corrected_elevational_centroids
  ) == 3,
  all(
    is.finite(
      corrected_elevational_centroids$
        predicted_centroid_m
    )
  ),
  all(
    corrected_elevational_centroids$
      predicted_centroid_m >= 0
  ),
  all(
    corrected_elevational_centroids$
      predicted_centroid_m <= 1300
  )
)

cat(
  "\nCorrected-NDVI elevational centroids:\n"
)

print(
  corrected_elevational_centroids,
  row.names = FALSE
)

# ============================================================
# 10.3 Export centroid tables
# ============================================================

readr::write_csv(
  corrected_centroid_grid_check,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_centroid_prediction_grid_check.csv"
  )
)

readr::write_csv(
  corrected_elevational_centroids,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_predicted_elevational_centroids.csv"
  )
)



# ============================================================
# 10.4 Manuscript Figure 4 with corrected NDVI
# ============================================================

dir.create(
  file.path(
    output_dir,
    "figures"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

corrected_centroid_plot_data <-
  corrected_elevational_centroids |>
  dplyr::mutate(
    baseline_centroid_m =
      corrected_pre_maria_centroid,
    centroid_plot_label =
      dplyr::case_when(
        as.character(
          Period
        ) == "Pre-Maria" ~
          sprintf(
            "%.1f m (baseline)",
            predicted_centroid_m
          ),
        TRUE ~
          sprintf(
            "%.1f m (%+.1f m)",
            predicted_centroid_m,
            change_from_pre_maria_m
          )
      )
  )

corrected_centroid_colors <- c(
  "Pre-Maria" = "#0072B2",
  "Inter-Hurricanes" = "#D55E00",
  "Post-Fiona" = "#009E73"
)

corrected_centroid_x_min <- floor(
  (
    min(
      corrected_centroid_plot_data$predicted_centroid_m,
      na.rm = TRUE
    ) - 15
  ) / 10
) * 10

corrected_centroid_x_max <- ceiling(
  (
    max(
      corrected_centroid_plot_data$predicted_centroid_m,
      na.rm = TRUE
    ) + 70
  ) / 10
) * 10

corrected_centroid_x_breaks <- pretty(
  c(
    corrected_centroid_x_min,
    corrected_centroid_x_max
  ),
  n = 5
)

corrected_centroid_x_breaks <- corrected_centroid_x_breaks[
  corrected_centroid_x_breaks >= corrected_centroid_x_min &
    corrected_centroid_x_breaks <= corrected_centroid_x_max
]

figure3_corrected_ndvi <-
  ggplot2::ggplot(
    data = corrected_centroid_plot_data,
    mapping = ggplot2::aes(
      x = predicted_centroid_m,
      y = Period
    )
  ) +
  ggplot2::geom_vline(
    xintercept = corrected_pre_maria_centroid,
    linetype = "dotted",
    linewidth = 0.8,
    color = "grey45"
  ) +
  ggplot2::geom_segment(
    mapping = ggplot2::aes(
      x = baseline_centroid_m,
      xend = predicted_centroid_m,
      y = Period,
      yend = Period
    ),
    linewidth = 1.0,
    color = "grey65"
  ) +
  ggplot2::geom_point(
    mapping = ggplot2::aes(
      color = Period
    ),
    size = 5
  ) +
  ggplot2::geom_text(
    mapping = ggplot2::aes(
      label = centroid_plot_label
    ),
    nudge_x = 4,
    hjust = 0,
    size = 4.5,
    color = "black"
  ) +
  ggplot2::scale_color_manual(
    values = corrected_centroid_colors
  ) +
  ggplot2::scale_x_continuous(
    limits = c(
      corrected_centroid_x_min,
      corrected_centroid_x_max
    ),
    breaks = corrected_centroid_x_breaks,
    labels = scales::label_number(
      accuracy = 1
    ),
    expand = ggplot2::expansion(
      mult = c(0, 0.01)
    )
  ) +
  ggplot2::labs(
    x = "Predicted elevational centroid (m a.s.l.)",
    y = NULL,
    color = NULL
  ) +
  ggplot2::guides(
    color = "none"
  ) +
  ggplot2::theme_classic(
    base_size = 14
  ) +
  ggplot2::theme(
    axis.title.x = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black",
      margin = ggplot2::margin(t = 10)
    ),
    axis.text.x = ggplot2::element_text(
      size = 13.5,
      color = "black"
    ),
    axis.text.y = ggplot2::element_text(
      size = 13.5,
      color = "black"
    ),
    axis.line = ggplot2::element_line(
      linewidth = 0.7,
      color = "black"
    ),
    axis.ticks = ggplot2::element_line(
      linewidth = 0.6,
      color = "black"
    ),
    axis.ticks.length = grid::unit(
      0.18,
      "cm"
    ),
    plot.margin = ggplot2::margin(
      t = 12,
      r = 25,
      b = 12,
      l = 10
    )
  )

print(
  figure3_corrected_ndvi
)

figure3_paths <- save_reproducibility_figure(
  plot_object =
    figure3_corrected_ndvi,
  file_stem =
    "Figure4_elevational_centroid",
  width = 8,
  height = 4.5,
  dpi = 600
)

figure3_corrected_png <-
  figure3_paths$png

figure3_corrected_tiff <-
  figure3_paths$tiff

cat(
  "\nCorrected-NDVI elevational centroid analysis and manuscript Figure 4 were exported successfully.\n"
)

# ============================================================
# 11. Prepare corrected-NDVI data for dynamic occupancy
# ============================================================

if (!requireNamespace("unmarked", quietly = TRUE)) {
  stop(
    "Package 'unmarked' is required for the dynamic occupancy analysis."
  )
}

stopifnot(
  exists("dat"),
  exists("output_dir"),
  is.data.frame(dat),
  all(
    c(
      "SAMPLINGEVENTIDENTIFIER",
      "Date",
      "Period",
      "Presence",
      "Longitude",
      "Latitude",
      "Elevation_m",
      "NDVI"
    ) %in% names(dat)
  )
)

dynamic_corrected_period_levels <- c(
  "Pre-Maria",
  "Inter-Hurricanes",
  "Post-Fiona"
)

# Preserve the input-row order. It resolves ties among checklists
# from the same site, period, and year, as in the validated analysis.

dynamic_corrected_data <- dat |>
  dplyr::mutate(
    Source_Row = dplyr::row_number(),
    Year = as.integer(
      format(
        Date,
        "%Y"
      )
    ),
    Period = dplyr::case_when(
      as.character(Period) %in% c(
        "Pre-Maria",
        "Pre-María"
      ) ~ "Pre-Maria",
      as.character(Period) %in% c(
        "Inter-Hurricanes",
        "Inter-Huracanes"
      ) ~ "Inter-Hurricanes",
      as.character(Period) == "Post-Fiona" ~
        "Post-Fiona",
      TRUE ~ NA_character_
    ),
    Period = factor(
      Period,
      levels =
        dynamic_corrected_period_levels
    ),
    Grid_Longitude = round(
      Longitude,
      digits = 2
    ),
    Grid_Latitude = round(
      Latitude,
      digits = 2
    ),
    Site_ID = paste(
      Grid_Longitude,
      Grid_Latitude,
      sep = "_"
    )
  ) |>
  dplyr::filter(
    !is.na(Longitude),
    !is.na(Latitude),
    !is.na(Presence),
    !is.na(Period)
  )

stopifnot(
  nrow(dynamic_corrected_data) == 89682,
  sum(
    is.na(
      dynamic_corrected_data$Site_ID
    )
  ) == 0
)

# ============================================================
# 11.1 Sites represented in all three primary periods
# ============================================================

dynamic_corrected_site_period_counts <-
  dynamic_corrected_data |>
  dplyr::count(
    Site_ID,
    Period,
    name = "number_of_checklists",
    .drop = FALSE
  ) |>
  dplyr::filter(
    number_of_checklists > 0
  )

dynamic_corrected_eligible_sites <-
  dynamic_corrected_site_period_counts |>
  dplyr::group_by(
    Site_ID
  ) |>
  dplyr::summarise(
    number_of_periods =
      dplyr::n_distinct(
        Period
      ),
    minimum_checklists_per_period =
      min(
        number_of_checklists
      ),
    maximum_checklists_per_period =
      max(
        number_of_checklists
      ),
    total_checklists =
      sum(
        number_of_checklists
      ),
    .groups = "drop"
  ) |>
  dplyr::filter(
    number_of_periods == 3
  ) |>
  dplyr::arrange(
    Site_ID
  )

stopifnot(
  nrow(
    dynamic_corrected_eligible_sites
  ) == 1048,
  all(
    dynamic_corrected_eligible_sites$
      minimum_checklists_per_period >= 1
  )
)

dynamic_corrected_retained_data <-
  dynamic_corrected_data |>
  dplyr::semi_join(
    dynamic_corrected_eligible_sites,
    by = "Site_ID"
  )

# ============================================================
# 11.2 Corrected-NDVI site covariates
# ============================================================

dynamic_corrected_site_covariates_all <-
  dynamic_corrected_retained_data |>
  dplyr::group_by(
    Site_ID
  ) |>
  dplyr::summarise(
    Grid_Longitude =
      dplyr::first(
        Grid_Longitude
      ),
    Grid_Latitude =
      dplyr::first(
        Grid_Latitude
      ),
    mean_elevation_m = if (
      all(
        is.na(
          Elevation_m
        )
      )
    ) {
      NA_real_
    } else {
      mean(
        Elevation_m,
        na.rm = TRUE
      )
    },
    mean_NDVI_corrected = if (
      all(
        is.na(
          NDVI
        )
      )
    ) {
      NA_real_
    } else {
      mean(
        NDVI,
        na.rm = TRUE
      )
    },
    number_of_checklists =
      dplyr::n(),
    number_with_corrected_NDVI =
      sum(
        !is.na(
          NDVI
        )
      ),
    proportion_with_corrected_NDVI =
      mean(
        !is.na(
          NDVI
        )
      ),
    .groups = "drop"
  ) |>
  dplyr::arrange(
    Site_ID
  )

dynamic_corrected_sites_without_ndvi <-
  dynamic_corrected_site_covariates_all |>
  dplyr::filter(
    !is.finite(
      mean_NDVI_corrected
    )
  )

cat(
  "\nSites represented in all periods but lacking corrected NDVI:\n"
)

print(
  dynamic_corrected_sites_without_ndvi,
  row.names = FALSE
)

stopifnot(
  nrow(
    dynamic_corrected_sites_without_ndvi
  ) == 4,
  all(
    dynamic_corrected_sites_without_ndvi$
      number_with_corrected_NDVI == 0
  )
)

dynamic_corrected_site_covariates <-
  dynamic_corrected_site_covariates_all |>
  dplyr::filter(
    is.finite(
      mean_elevation_m
    ),
    is.finite(
      mean_NDVI_corrected
    )
  ) |>
  dplyr::arrange(
    Site_ID
  ) |>
  dplyr::mutate(
    Elevation_z = as.numeric(
      scale(
        mean_elevation_m
      )
    ),
    NDVI_z = as.numeric(
      scale(
        mean_NDVI_corrected
      )
    )
  )

stopifnot(
  nrow(
    dynamic_corrected_site_covariates
  ) == 1044,
  all(
    stats::complete.cases(
      dynamic_corrected_site_covariates[
        c(
          "Elevation_z",
          "NDVI_z"
        )
      ]
    )
  )
)

dynamic_corrected_model_data <-
  dynamic_corrected_retained_data |>
  dplyr::semi_join(
    dynamic_corrected_site_covariates |>
      dplyr::select(
        Site_ID
      ),
    by = "Site_ID"
  )

stopifnot(
  dplyr::n_distinct(
    dynamic_corrected_model_data$Site_ID
  ) == 1044
)

# ============================================================
# 11.3 Select up to three checklists per site and period
# ============================================================

dynamic_corrected_selected_checklists <-
  dynamic_corrected_model_data |>
  dplyr::arrange(
    Site_ID,
    Period,
    Year,
    Source_Row
  ) |>
  dplyr::group_by(
    Site_ID,
    Period
  ) |>
  dplyr::mutate(
    Secondary_Occasion =
      dplyr::row_number()
  ) |>
  dplyr::filter(
    Secondary_Occasion <= 3
  ) |>
  dplyr::ungroup()

stopifnot(
  max(
    dynamic_corrected_selected_checklists$
      Secondary_Occasion,
    na.rm = TRUE
  ) <= 3,
  all(
    dynamic_corrected_selected_checklists$
      Presence %in% c(
        0,
        1
      )
  ),
  nrow(
    dynamic_corrected_selected_checklists
  ) == 7908
)

# ============================================================
# 11.4 Construct the 1,044 x 9 detection-history matrix
# ============================================================

dynamic_corrected_detection_wide <-
  dynamic_corrected_selected_checklists |>
  dplyr::mutate(
    Period_Code = dplyr::recode(
      as.character(
        Period
      ),
      "Pre-Maria" = "PreMaria",
      "Inter-Hurricanes" =
        "InterHurricanes",
      "Post-Fiona" = "PostFiona"
    ),
    Detection_Column = paste0(
      Period_Code,
      "_",
      Secondary_Occasion
    )
  ) |>
  dplyr::select(
    Site_ID,
    Detection_Column,
    Presence
  ) |>
  tidyr::pivot_wider(
    names_from =
      Detection_Column,
    values_from =
      Presence
  ) |>
  dplyr::arrange(
    Site_ID
  )

dynamic_corrected_detection_column_order <-
  c(
    "PreMaria_1",
    "PreMaria_2",
    "PreMaria_3",
    "InterHurricanes_1",
    "InterHurricanes_2",
    "InterHurricanes_3",
    "PostFiona_1",
    "PostFiona_2",
    "PostFiona_3"
  )

for (
  column_name in
  dynamic_corrected_detection_column_order
) {
  if (
    !column_name %in%
      names(
        dynamic_corrected_detection_wide
      )
  ) {
    dynamic_corrected_detection_wide[[column_name]] <- NA_integer_
  }
}

dynamic_corrected_detection_wide <-
  dynamic_corrected_detection_wide |>
  dplyr::select(
    Site_ID,
    dplyr::all_of(
      dynamic_corrected_detection_column_order
    )
  ) |>
  dplyr::arrange(
    Site_ID
  )

dynamic_corrected_detection_matrix <-
  as.matrix(
    dynamic_corrected_detection_wide[
      dynamic_corrected_detection_column_order
    ]
  )

storage.mode(
  dynamic_corrected_detection_matrix
) <- "numeric"

rownames(
  dynamic_corrected_detection_matrix
) <- dynamic_corrected_detection_wide$
  Site_ID

dynamic_corrected_site_covariates <-
  dynamic_corrected_site_covariates |>
  dplyr::arrange(
    match(
      Site_ID,
      dynamic_corrected_detection_wide$
        Site_ID
    )
  )

stopifnot(
  nrow(
    dynamic_corrected_detection_matrix
  ) == 1044,
  ncol(
    dynamic_corrected_detection_matrix
  ) == 9,
  sum(
    !is.na(
      dynamic_corrected_detection_matrix
    )
  ) == 7908,
  sum(
    is.na(
      dynamic_corrected_detection_matrix
    )
  ) == 1488,
  all(
    is.na(
      dynamic_corrected_detection_matrix
    ) |
      dynamic_corrected_detection_matrix %in%
        c(
          0,
          1
        )
  ),
  identical(
    dynamic_corrected_site_covariates$
      Site_ID,
    dynamic_corrected_detection_wide$
      Site_ID
  )
)

# ============================================================
# 11.5 Create the corrected-NDVI unmarked frame
# ============================================================

dynamic_corrected_site_covariates_umf <-
  data.frame(
    Elevation =
      dynamic_corrected_site_covariates$
        Elevation_z,
    NDVI =
      dynamic_corrected_site_covariates$
        NDVI_z,
    stringsAsFactors = FALSE
  )

rownames(
  dynamic_corrected_site_covariates_umf
) <- dynamic_corrected_site_covariates$
  Site_ID

dynamic_corrected_period_by_observation <-
  c(
    rep(
      "Pre-Maria",
      3
    ),
    rep(
      "Inter-Hurricanes",
      3
    ),
    rep(
      "Post-Fiona",
      3
    )
  )

dynamic_corrected_detection_reference_levels <-
  c(
    "Inter-Hurricanes",
    "Post-Fiona",
    "Pre-Maria"
  )

# Observation covariates must be stacked by observation column:
# repeat each column's period label across all sites.

dynamic_corrected_observation_covariates_umf <-
  data.frame(
    Period = factor(
      rep(
        dynamic_corrected_period_by_observation,
        each =
          nrow(
            dynamic_corrected_detection_matrix
          )
      ),
      levels =
        dynamic_corrected_detection_reference_levels
    )
  )

dynamic_corrected_umf <-
  unmarked::unmarkedMultFrame(
    y =
      dynamic_corrected_detection_matrix,
    siteCovs =
      dynamic_corrected_site_covariates_umf,
    obsCovs =
      dynamic_corrected_observation_covariates_umf,
    numPrimary = 3
  )

dynamic_corrected_y_check <-
  unmarked::getY(
    dynamic_corrected_umf
  )

dynamic_corrected_site_covariate_check <-
  unmarked::siteCovs(
    dynamic_corrected_umf
  )

dynamic_corrected_observation_covariate_check <-
  unmarked::obsCovs(
    dynamic_corrected_umf
  )

stopifnot(
  inherits(
    dynamic_corrected_umf,
    "unmarkedMultFrame"
  ),
  nrow(
    dynamic_corrected_y_check
  ) == 1044,
  ncol(
    dynamic_corrected_y_check
  ) == 9,
  nrow(
    dynamic_corrected_site_covariate_check
  ) == 1044,
  nrow(
    dynamic_corrected_observation_covariate_check
  ) == 9396,
  sum(
    !is.na(
      dynamic_corrected_y_check
    )
  ) == 7908,
  sum(
    is.na(
      dynamic_corrected_y_check
    )
  ) == 1488,
  levels(
    dynamic_corrected_observation_covariate_check$
      Period
  )[1] == "Inter-Hurricanes"
)

# ============================================================
# 11.6 Summaries and exports
# ============================================================

dynamic_corrected_preparation_summary <-
  data.frame(
    metric = c(
      "Sites represented in all three periods",
      "Sites lacking corrected NDVI",
      "Final sites with complete covariates",
      "Primary periods",
      "Maximum secondary occasions per period",
      "Detection-history columns",
      "Observed survey occasions",
      "Missing survey occasions",
      "Observation-covariate rows",
      "Detection reference period"
    ),
    value = c(
      nrow(
        dynamic_corrected_eligible_sites
      ),
      nrow(
        dynamic_corrected_sites_without_ndvi
      ),
      nrow(
        dynamic_corrected_detection_matrix
      ),
      3,
      3,
      ncol(
        dynamic_corrected_detection_matrix
      ),
      sum(
        !is.na(
          dynamic_corrected_detection_matrix
        )
      ),
      sum(
        is.na(
          dynamic_corrected_detection_matrix
        )
      ),
      nrow(
        dynamic_corrected_observation_covariates_umf
      ),
      levels(
        dynamic_corrected_observation_covariates_umf$
          Period
      )[1]
    ),
    stringsAsFactors = FALSE
  )


cat(
  "\nCorrected-NDVI dynamic occupancy preparation summary:\n"
)

print(
  dynamic_corrected_preparation_summary,
  row.names = FALSE
)


readr::write_csv(
  dynamic_corrected_eligible_sites,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_eligible_sites.csv"
  )
)

readr::write_csv(
  dynamic_corrected_sites_without_ndvi,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_dynamic_sites_without_ndvi.csv"
  )
)

readr::write_csv(
  dynamic_corrected_site_covariates,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_site_covariates.csv"
  )
)

readr::write_csv(
  dynamic_corrected_selected_checklists,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_selected_checklists.csv"
  )
)

readr::write_csv(
  dynamic_corrected_detection_wide,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_detection_history.csv"
  )
)

readr::write_csv(
  dynamic_corrected_preparation_summary,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_dynamic_preparation_summary.csv"
  )
)


base::saveRDS(
  dynamic_corrected_umf,
  file.path(
    output_dir,
    "models",
    "corrected_ndvi_dynamic_unmarked_frame.rds"
  )
)

cat(
  "\nCorrected-NDVI dynamic occupancy data and unmarked frame were prepared successfully.\n"
)

# ============================================================
# 12. Fit the corrected-NDVI dynamic occupancy model
# ============================================================

library(unmarked)

stopifnot(
  exists("dynamic_corrected_umf"),
  exists("dynamic_corrected_detection_matrix"),
  exists("dynamic_corrected_site_covariates"),
  exists("output_dir"),
  inherits(
    dynamic_corrected_umf,
    "unmarkedMultFrame"
  ),
  nrow(
    dynamic_corrected_detection_matrix
  ) == 1044,
  ncol(
    dynamic_corrected_detection_matrix
  ) == 9
)

cat(
  "\nFitting the corrected-NDVI dynamic occupancy model.\n",
  "This step may take several minutes. Do not press Stop.\n"
)

dynamic_corrected_model_best <- unmarked::colext(
  psiformula = ~ Elevation + NDVI,
  gammaformula = ~ NDVI,
  epsilonformula = ~ Elevation + NDVI,
  pformula = ~ Period,
  data = dynamic_corrected_umf,
  method = "BFGS",
  se = TRUE,
  control = list(
    maxit = 10000,
    trace = 0
  )
)

stopifnot(
  inherits(
    dynamic_corrected_model_best,
    "unmarkedFitColExt"
  ),
  is.finite(
    dynamic_corrected_model_best@AIC
  )
)

stopifnot(
  exists("dynamic_corrected_model_best"),
  inherits(
    dynamic_corrected_model_best,
    "unmarkedFitColExt"
  ),
  exists("dynamic_corrected_detection_matrix"),
  exists("output_dir"),
  is.finite(
    dynamic_corrected_model_best@AIC
  )
)

dynamic_corrected_summary_object <-
  summary(
    dynamic_corrected_model_best
  )

stopifnot(
  all(
    c(
      "psi",
      "col",
      "ext",
      "det"
    ) %in%
      names(
        dynamic_corrected_summary_object
      )
  )
)

extract_dynamic_component <- function(
    summary_object,
    component_name,
    component_label
) {

  component_table <-
    as.data.frame(
      summary_object[[component_name]]
    )

  stopifnot(
    nrow(component_table) > 0,
    ncol(component_table) >= 4
  )

  data.frame(
    component = component_label,
    term = rownames(
      component_table
    ),
    estimate = as.numeric(
      component_table[[1]]
    ),
    standard_error = as.numeric(
      component_table[[2]]
    ),
    z_value = as.numeric(
      component_table[[3]]
    ),
    p_value = as.numeric(
      component_table[[4]]
    ),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

dynamic_corrected_coefficients <-
  dplyr::bind_rows(
    extract_dynamic_component(
      dynamic_corrected_summary_object,
      "psi",
      "Initial occupancy"
    ),
    extract_dynamic_component(
      dynamic_corrected_summary_object,
      "col",
      "Colonization"
    ),
    extract_dynamic_component(
      dynamic_corrected_summary_object,
      "ext",
      "Local extinction"
    ),
    extract_dynamic_component(
      dynamic_corrected_summary_object,
      "det",
      "Detection"
    )
  ) |>
  dplyr::mutate(
    odds_ratio = exp(
      estimate
    ),
    conf_low_logit =
      estimate -
      1.96 * standard_error,
    conf_high_logit =
      estimate +
      1.96 * standard_error,
    odds_ratio_conf_low_95 =
      exp(
        conf_low_logit
      ),
    odds_ratio_conf_high_95 =
      exp(
        conf_high_logit
      )
  )

cat(
  "\nCorrected-NDVI dynamic occupancy coefficients:\n"
)

print(
  dynamic_corrected_coefficients,
  row.names = FALSE
)

dynamic_corrected_model_summary <-
  data.frame(
    metric = c(
      "AIC",
      "Number of sites",
      "Observed survey occasions",
      "Missing survey occasions"
    ),
    value = c(
      dynamic_corrected_model_best@AIC,
      nrow(
        dynamic_corrected_detection_matrix
      ),
      sum(
        !is.na(
          dynamic_corrected_detection_matrix
        )
      ),
      sum(
        is.na(
          dynamic_corrected_detection_matrix
        )
      )
    ),
    stringsAsFactors = FALSE
  )


cat(
  "\nCorrected-NDVI dynamic model summary:\n"
)

print(
  dynamic_corrected_model_summary,
  row.names = FALSE
)


readr::write_csv(
  dynamic_corrected_coefficients,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_model_coefficients.csv"
  )
)

readr::write_csv(
  dynamic_corrected_model_summary,
  file.path(
    output_dir,
    "validation",
    "corrected_ndvi_dynamic_model_summary.csv"
  )
)


base::saveRDS(
  dynamic_corrected_model_best,
  file.path(
    output_dir,
    "models",
    "corrected_ndvi_dynamic_model_best.rds"
  )
)

capture.output(
  summary(
    dynamic_corrected_model_best
  ),
  file = file.path(
    output_dir,
    "validation",
    "corrected_ndvi_dynamic_model_summary.txt"
  )
)

stopifnot(
  file.exists(
    file.path(
      output_dir,
      "models",
      "corrected_ndvi_dynamic_model_best.rds"
    )
  ),
  nrow(
    dynamic_corrected_coefficients
  ) == 11
)

cat(
  "\nCorrected-NDVI dynamic occupancy model was fitted and exported successfully.\n"
)

# ============================================================
# 13. Corrected-NDVI dynamic occupancy candidate-model selection
# ============================================================

library(unmarked)

stopifnot(
  exists("dynamic_corrected_model_best"),
  inherits(
    dynamic_corrected_model_best,
    "unmarkedFitColExt"
  ),
  exists("dynamic_corrected_detection_matrix"),
  exists("dynamic_corrected_site_covariates"),
  exists("dynamic_corrected_observation_covariates_umf"),
  exists("dynamic_corrected_umf"),
  exists("output_dir"),
  nrow(
    dynamic_corrected_detection_matrix
  ) == 1044
)

# Add the same categorical elevation variable used in the
# original elevation-only candidate model.

dynamic_corrected_candidate_site_covariates <-
  data.frame(
    Elevation =
      dynamic_corrected_site_covariates$
        Elevation_z,
    NDVI =
      dynamic_corrected_site_covariates$
        NDVI_z,
    ElevationZone = factor(
      ifelse(
        dynamic_corrected_site_covariates$
          mean_elevation_m < 250,
        "Lowlands (<250 m)",
        "Highlands (>=250 m)"
      ),
      levels = c(
        "Highlands (>=250 m)",
        "Lowlands (<250 m)"
      )
    ),
    stringsAsFactors = FALSE
  )

rownames(
  dynamic_corrected_candidate_site_covariates
) <- dynamic_corrected_site_covariates$
  Site_ID

stopifnot(
  nrow(
    dynamic_corrected_candidate_site_covariates
  ) == 1044,
  all(
    stats::complete.cases(
      dynamic_corrected_candidate_site_covariates
    )
  ),
  identical(
    rownames(
      dynamic_corrected_candidate_site_covariates
    ),
    rownames(
      dynamic_corrected_detection_matrix
    )
  )
)

dynamic_corrected_umf_candidates <-
  unmarked::unmarkedMultFrame(
    y =
      dynamic_corrected_detection_matrix,
    siteCovs =
      dynamic_corrected_candidate_site_covariates,
    obsCovs =
      dynamic_corrected_observation_covariates_umf,
    numPrimary = 3
  )

stopifnot(
  identical(
    unmarked::getY(
      dynamic_corrected_umf_candidates
    ),
    unmarked::getY(
      dynamic_corrected_umf
    )
  )
)

cat(
  "\nFitting corrected-NDVI dynamic candidate models.\n",
  "This step may take several minutes. Do not press Stop.\n"
)

# Candidate 2: environmental model without elevation
# in the local-extinction component.

dynamic_corrected_model_environmental <-
  unmarked::colext(
    psiformula = ~ Elevation + NDVI,
    gammaformula = ~ NDVI,
    epsilonformula = ~ NDVI,
    pformula = ~ Period,
    data =
      dynamic_corrected_umf_candidates,
    method = "BFGS",
    se = TRUE,
    control = list(
      maxit = 10000,
      trace = 0
    )
  )

# Candidate 3: categorical elevation-only model.

dynamic_corrected_model_elevation_only <-
  unmarked::colext(
    psiformula = ~ ElevationZone,
    gammaformula = ~ 1,
    epsilonformula = ~ 1,
    pformula = ~ Period,
    data =
      dynamic_corrected_umf_candidates,
    method = "BFGS",
    se = TRUE,
    control = list(
      maxit = 10000,
      trace = 0
    )
  )

# Candidate 4: fully constant null model.

dynamic_corrected_model_null <-
  unmarked::colext(
    psiformula = ~ 1,
    gammaformula = ~ 1,
    epsilonformula = ~ 1,
    pformula = ~ 1,
    data =
      dynamic_corrected_umf_candidates,
    method = "BFGS",
    se = TRUE,
    control = list(
      maxit = 10000,
      trace = 0
    )
  )

dynamic_corrected_candidate_models <-
  list(
    Final =
      dynamic_corrected_model_best,
    Environmental =
      dynamic_corrected_model_environmental,
    Elevation_only =
      dynamic_corrected_model_elevation_only,
    Null =
      dynamic_corrected_model_null
  )

stopifnot(
  all(
    vapply(
      dynamic_corrected_candidate_models,
      inherits,
      logical(1),
      what = "unmarkedFitColExt"
    )
  ),
  all(
    is.finite(
      vapply(
        dynamic_corrected_candidate_models,
        function(model) {
          model@AIC
        },
        numeric(1)
      )
    )
  )
)

dynamic_corrected_candidate_definitions <-
  data.frame(
    model = c(
      "Final",
      "Environmental",
      "Elevation_only",
      "Null"
    ),
    initial_occupancy = c(
      "Elevation + NDVI",
      "Elevation + NDVI",
      "ElevationZone",
      "Intercept only"
    ),
    colonization = c(
      "NDVI",
      "NDVI",
      "Intercept only",
      "Intercept only"
    ),
    local_extinction = c(
      "Elevation + NDVI",
      "NDVI",
      "Intercept only",
      "Intercept only"
    ),
    detection = c(
      "Period",
      "Period",
      "Period",
      "Intercept only"
    ),
    K = c(
      11L,
      10L,
      7L,
      4L
    ),
    stringsAsFactors = FALSE
  )

dynamic_corrected_model_selection <-
  data.frame(
    model = names(
      dynamic_corrected_candidate_models
    ),
    K =
      dynamic_corrected_candidate_definitions$K,
    AIC = vapply(
      dynamic_corrected_candidate_models,
      function(model) {
        model@AIC
      },
      numeric(1)
    ),
    stringsAsFactors = FALSE
  ) |>
  dplyr::arrange(
    AIC
  ) |>
  dplyr::mutate(
    delta_AIC =
      AIC -
      min(
        AIC
      ),
    Akaike_weight =
      exp(
        -0.5 * delta_AIC
      ) /
      sum(
        exp(
          -0.5 * delta_AIC
        )
      ),
    rank =
      dplyr::row_number()
  ) |>
  dplyr::select(
    rank,
    model,
    K,
    AIC,
    delta_AIC,
    Akaike_weight
  )

cat(
  "\nCorrected-NDVI dynamic candidate-model definitions:\n"
)

print(
  dynamic_corrected_candidate_definitions,
  row.names = FALSE
)

cat(
  "\nCorrected-NDVI dynamic model-selection table:\n"
)

print(
  dynamic_corrected_model_selection,
  row.names = FALSE
)


stopifnot(
  nrow(
    dynamic_corrected_model_selection
  ) == 4,
  all(
    dynamic_corrected_model_selection$
      delta_AIC >= 0
  ),
  abs(
    sum(
      dynamic_corrected_model_selection$
        Akaike_weight
    ) - 1
  ) < 1e-8,
  dynamic_corrected_model_selection$
    delta_AIC[1] == 0
)

base::saveRDS(
  dynamic_corrected_candidate_models,
  file.path(
    output_dir,
    "models",
    "corrected_ndvi_dynamic_candidate_models.rds"
  )
)

readr::write_csv(
  dynamic_corrected_candidate_definitions,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_candidate_model_definitions.csv"
  )
)

readr::write_csv(
  dynamic_corrected_model_selection,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_model_selection.csv"
  )
)

# Supplementary Table S1: combine candidate definitions and model-selection
# statistics into the same structure used in the formatted supplement.

supplementary_table_s1 <- dynamic_corrected_model_selection |>
  dplyr::mutate(
    model_description = dplyr::case_when(
      model == "Final" ~
        "psi ~ Elevation + NDVI; gamma ~ NDVI; epsilon ~ Elevation + NDVI; p ~ Period",
      model == "Environmental" ~
        "psi ~ Elevation + NDVI; gamma ~ NDVI; epsilon ~ NDVI; p ~ Period",
      model == "Elevation_only" ~
        "psi ~ Elevation zone; gamma ~ 1; epsilon ~ 1; p ~ Period",
      model == "Null" ~
        "psi ~ 1; gamma ~ 1; epsilon ~ 1; p ~ 1",
      TRUE ~ model
    )
  ) |>
  dplyr::select(
    model_description,
    K,
    AIC,
    delta_AIC,
    Akaike_weight
  )

readr::write_csv(
  supplementary_table_s1,
  file.path(
    output_dir,
    "tables",
    "Supplementary_Table_S1_dynamic_occupancy_model_selection.csv"
  )
)


capture.output(
  dynamic_corrected_model_selection,
  file = file.path(
    output_dir,
    "validation",
    "corrected_ndvi_dynamic_model_selection.txt"
  )
)

stopifnot(
  file.exists(
    file.path(
      output_dir,
      "models",
      "corrected_ndvi_dynamic_candidate_models.rds"
    )
  ),
  file.exists(
    file.path(
      output_dir,
      "tables",
      "corrected_ndvi_dynamic_model_selection.csv"
    )
  )
)

cat(
  "\nCorrected-NDVI dynamic candidate-model selection was completed successfully.\n"
)

# ============================================================
# 14. Dynamic occupancy figures with corrected NDVI
# ============================================================

library(unmarked)

if (!requireNamespace("sf", quietly = TRUE)) {
  stop(
    "Package 'sf' is required to generate the dynamic occupancy map."
  )
}

stopifnot(
  exists("dynamic_corrected_model_best"),
  inherits(
    dynamic_corrected_model_best,
    "unmarkedFitColExt"
  ),
  exists("dynamic_corrected_site_covariates"),
  exists("output_dir"),
  nrow(
    dynamic_corrected_site_covariates
  ) == 1044
)

dir.create(
  file.path(
    output_dir,
    "figures"
  ),
  recursive = TRUE,
  showWarnings = FALSE
)

# ============================================================
# 14.1 Manuscript Figure 5: Predicted local extinction across elevation
# ============================================================

dynamic_corrected_elevation_center <-
  mean(
    dynamic_corrected_site_covariates$
      mean_elevation_m,
    na.rm = TRUE
  )

dynamic_corrected_elevation_scale <-
  stats::sd(
    dynamic_corrected_site_covariates$
      mean_elevation_m,
    na.rm = TRUE
  )

stopifnot(
  is.finite(
    dynamic_corrected_elevation_center
  ),
  is.finite(
    dynamic_corrected_elevation_scale
  ),
  dynamic_corrected_elevation_scale > 0
)

dynamic_corrected_extinction_elevation_grid <-
  data.frame(
    Elevation_m = seq(
      min(
        dynamic_corrected_site_covariates$
          mean_elevation_m,
        na.rm = TRUE
      ),
      max(
        dynamic_corrected_site_covariates$
          mean_elevation_m,
        na.rm = TRUE
      ),
      length.out = 200
    )
  ) |>
  dplyr::mutate(
    Elevation = (
      Elevation_m -
        dynamic_corrected_elevation_center
    ) /
      dynamic_corrected_elevation_scale,
    NDVI = 0
  )

dynamic_corrected_extinction_elevation_prediction <-
  predict(
    dynamic_corrected_model_best,
    type = "ext",
    newdata =
      dynamic_corrected_extinction_elevation_grid,
    appendData = TRUE
  ) |>
  dplyr::mutate(
    predicted_percent =
      100 * Predicted,
    lower_percent =
      100 * lower,
    upper_percent =
      100 * upper
  )

stopifnot(
  nrow(
    dynamic_corrected_extinction_elevation_prediction
  ) == 200,
  all(
    is.finite(
      dynamic_corrected_extinction_elevation_prediction$
        Predicted
    )
  ),
  all(
    dynamic_corrected_extinction_elevation_prediction$
      Predicted >= 0
  ),
  all(
    dynamic_corrected_extinction_elevation_prediction$
      Predicted <= 1
  )
)

dynamic_corrected_extinction_reference_predictions <-
  dynamic_corrected_extinction_elevation_prediction |>
  dplyr::mutate(
    distance_from_100m =
      abs(
        Elevation_m - 100
      ),
    distance_from_500m =
      abs(
        Elevation_m - 500
      ),
    distance_from_900m =
      abs(
        Elevation_m - 900
      )
  ) |>
  dplyr::summarise(
    elevation_100m_percent =
      predicted_percent[
        which.min(
          distance_from_100m
        )
      ],
    elevation_500m_percent =
      predicted_percent[
        which.min(
          distance_from_500m
        )
      ],
    elevation_900m_percent =
      predicted_percent[
        which.min(
          distance_from_900m
        )
      ]
  )

cat(
  "\nCorrected-NDVI predicted local extinction at reference elevations:\n"
)

print(
  dynamic_corrected_extinction_reference_predictions,
  row.names = FALSE
)

figure4_corrected_dynamic_extinction <-
  ggplot2::ggplot(
    dynamic_corrected_extinction_elevation_prediction,
    ggplot2::aes(
      x = Elevation_m,
      y = predicted_percent
    )
  ) +
  ggplot2::geom_ribbon(
    ggplot2::aes(
      ymin = lower_percent,
      ymax = upper_percent
    ),
    fill = "#B8D8C0",
    alpha = 0.45
  ) +
  ggplot2::geom_line(
    color = "#006D5B",
    linewidth = 1.3
  ) +
  ggplot2::scale_x_continuous(
    breaks = scales::breaks_pretty(
      n = 6
    ),
    labels = scales::label_number(
      accuracy = 1,
      big.mark = ","
    ),
    expand = ggplot2::expansion(
      mult = c(0, 0.01)
    )
  ) +
  ggplot2::scale_y_continuous(
    limits = c(
      0,
      min(
        100,
        max(
          dynamic_corrected_extinction_elevation_prediction$upper_percent,
          na.rm = TRUE
        ) * 1.03
      )
    ),
    breaks = scales::breaks_pretty(
      n = 6
    ),
    labels = scales::label_number(
      accuracy = 1,
      suffix = "%"
    ),
    expand = ggplot2::expansion(
      mult = c(0, 0.02)
    )
  ) +
  ggplot2::labs(
    x = "Elevation (m a.s.l.)",
    y = "Predicted local extinction probability"
  ) +
  ggplot2::theme_classic(
    base_size = 14
  ) +
  ggplot2::theme(
    axis.title.x = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black",
      margin = ggplot2::margin(t = 10)
    ),
    axis.title.y = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black",
      margin = ggplot2::margin(r = 10)
    ),
    axis.text.x = ggplot2::element_text(
      size = 13.5,
      color = "black"
    ),
    axis.text.y = ggplot2::element_text(
      size = 13.5,
      color = "black"
    ),
    axis.line = ggplot2::element_line(
      linewidth = 0.7,
      color = "black"
    ),
    axis.ticks = ggplot2::element_line(
      linewidth = 0.6,
      color = "black"
    ),
    axis.ticks.length = grid::unit(
      0.18,
      "cm"
    ),
    plot.margin = ggplot2::margin(
      t = 10,
      r = 16,
      b = 12,
      l = 12
    )
  )

print(
  figure4_corrected_dynamic_extinction
)

figure4_paths <- save_reproducibility_figure(
  plot_object =
    figure4_corrected_dynamic_extinction,
  file_stem =
    "Figure5_local_extinction_elevation",
  width = 8,
  height = 5.5,
  dpi = 600
)

figure4_corrected_dynamic_png <-
  figure4_paths$png

figure4_corrected_dynamic_tiff <-
  figure4_paths$tiff

readr::write_csv(
  dynamic_corrected_extinction_elevation_prediction,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_local_extinction_by_elevation.csv"
  )
)

readr::write_csv(
  dynamic_corrected_extinction_reference_predictions,
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_local_extinction_reference_elevations.csv"
  )
)

# ============================================================
# 14.2 Manuscript Figure 6: Predicted local extinction by retained grid cell
# ============================================================

dynamic_corrected_site_prediction_data <-
  data.frame(
    Elevation =
      dynamic_corrected_site_covariates$
        Elevation_z,
    NDVI =
      dynamic_corrected_site_covariates$
        NDVI_z
  )

dynamic_corrected_extinction_site_prediction <-
  predict(
    dynamic_corrected_model_best,
    type = "ext",
    newdata =
      dynamic_corrected_site_prediction_data,
    appendData = FALSE
  )

stopifnot(
  nrow(
    dynamic_corrected_extinction_site_prediction
  ) ==
    nrow(
      dynamic_corrected_site_covariates
    )
)

dynamic_corrected_extinction_map_data <-
  dplyr::bind_cols(
    dynamic_corrected_site_covariates |>
      dplyr::select(
        Site_ID,
        Grid_Longitude,
        Grid_Latitude,
        mean_elevation_m,
        mean_NDVI_corrected,
        Elevation_z,
        NDVI_z
      ),
    dynamic_corrected_extinction_site_prediction |>
      dplyr::select(
        Predicted,
        SE,
        lower,
        upper
      )
  ) |>
  dplyr::mutate(
    predicted_percent =
      100 * Predicted,
    lower_percent =
      100 * lower,
    upper_percent =
      100 * upper
  )

stopifnot(
  nrow(
    dynamic_corrected_extinction_map_data
  ) == 1044,
  all(
    is.finite(
      dynamic_corrected_extinction_map_data$
        predicted_percent
    )
  )
)

make_corrected_dynamic_grid_cell <- function(
    longitude,
    latitude,
    size = 0.01
) {

  x_min <- longitude - size / 2
  x_max <- longitude + size / 2
  y_min <- latitude - size / 2
  y_max <- latitude + size / 2

  sf::st_polygon(
    list(
      matrix(
        c(
          x_min, y_min,
          x_max, y_min,
          x_max, y_max,
          x_min, y_max,
          x_min, y_min
        ),
        ncol = 2,
        byrow = TRUE
      )
    )
  )
}

dynamic_corrected_grid_geometries <-
  Map(
    make_corrected_dynamic_grid_cell,
    dynamic_corrected_extinction_map_data$
      Grid_Longitude,
    dynamic_corrected_extinction_map_data$
      Grid_Latitude
  )

dynamic_corrected_map_cells <-
  sf::st_sf(
    dynamic_corrected_extinction_map_data,
    geometry =
      sf::st_sfc(
        dynamic_corrected_grid_geometries,
        crs = 4326
      )
  )

figure5_corrected_dynamic_map <-
  ggplot2::ggplot() +
  ggplot2::geom_sf(
    data =
      dynamic_corrected_map_cells,
    ggplot2::aes(
      fill = predicted_percent
    ),
    color = NA
  ) +
  ggplot2::scale_fill_gradientn(
    name =
      "Predicted local\nextinction (%)",
    colours = c(
      "#006837",
      "#31A354",
      "#ADDD8E",
      "#FFF7BC"
    ),
    limits = range(
      dynamic_corrected_extinction_map_data$
        predicted_percent,
      na.rm = TRUE
    )
  ) +
  ggplot2::coord_sf(
    xlim = c(
      -67.35,
      -65.15
    ),
    ylim = c(
      17.85,
      18.60
    ),
    expand = FALSE
  ) +
  ggplot2::labs(
    x = "Longitude",
    y = "Latitude"
  ) +
  ggplot2::theme_classic(
    base_size = 12
  ) +
  ggplot2::theme(
    legend.position = "right",
    legend.title =
      ggplot2::element_text(
        size = 10
      ),
    legend.text =
      ggplot2::element_text(
        size = 9
      ),
    axis.title =
      ggplot2::element_text(
        size = 11
      ),
    axis.text =
      ggplot2::element_text(
        size = 9
      ),
    plot.margin =
      ggplot2::margin(
        t = 8,
        r = 8,
        b = 8,
        l = 8
      )
  )

print(
  figure5_corrected_dynamic_map
)

figure5_paths <- save_reproducibility_figure(
  plot_object =
    figure5_corrected_dynamic_map,
  file_stem =
    "Figure6_local_extinction_map",
  width = 8,
  height = 5,
  dpi = 600
)

figure5_corrected_dynamic_png <-
  figure5_paths$png

figure5_corrected_dynamic_tiff <-
  figure5_paths$tiff

readr::write_csv(
  sf::st_drop_geometry(
    dynamic_corrected_map_cells
  ),
  file.path(
    output_dir,
    "tables",
    "corrected_ndvi_dynamic_local_extinction_by_site.csv"
  )
)

stopifnot(
  file.exists(
    figure4_corrected_dynamic_png
  ),
  file.exists(
    figure4_corrected_dynamic_tiff
  ),
  file.exists(
    figure5_corrected_dynamic_png
  ),
  file.exists(
    figure5_corrected_dynamic_tiff
  )
)

cat(
  "\nCorrected-NDVI dynamic occupancy manuscript Figures 5 and 6 were exported successfully.\n"
)

# ============================================================
# 15. Reproducibility record and output manifest
# ============================================================

analysis_end_time <- Sys.time()

analysis_run_summary <- data.frame(
  metric = c(
    "Analysis start",
    "Analysis end",
    "Elapsed minutes",
    "Input file",
    "Input rows after 2013-2025 filtering",
    "Complete corrected-NDVI GLM rows",
    "Dynamic occupancy sites",
    "Best GLM",
    "Best GLM AIC",
    "Best dynamic model AIC"
  ),
  value = c(
    format(
      analysis_start_time,
      "%Y-%m-%d %H:%M:%S %Z"
    ),
    format(
      analysis_end_time,
      "%Y-%m-%d %H:%M:%S %Z"
    ),
    round(
      as.numeric(
        difftime(
          analysis_end_time,
          analysis_start_time,
          units = "mins"
        )
      ),
      2
    ),
    normalizePath(
      input_file,
      winslash = "/",
      mustWork = TRUE
    ),
    nrow(
      dat
    ),
    nrow(
      dat_ndvi_corrected
    ),
    nrow(
      dynamic_corrected_detection_matrix
    ),
    dynamic_corrected_model_selection$
      model[1],
    glm_model_selection_corrected_ndvi$
      AIC[1],
    dynamic_corrected_model_selection$
      AIC[1]
  ),
  stringsAsFactors = FALSE
)

readr::write_csv(
  analysis_run_summary,
  file.path(
    output_dir,
    "validation",
    "analysis_run_summary.csv"
  )
)

capture.output(
  utils::sessionInfo(),
  file = file.path(
    output_dir,
    "validation",
    "sessionInfo.txt"
  )
)

output_files <- list.files(
  output_dir,
  recursive = TRUE,
  full.names = TRUE,
  all.files = FALSE
)

output_manifest <- data.frame(
  relative_path = substring(
    normalizePath(
      output_files,
      winslash = "/",
      mustWork = TRUE
    ),
    nchar(
      normalizePath(
        output_dir,
        winslash = "/",
        mustWork = TRUE
      )
    ) + 2
  ),
  size_bytes = file.info(
    output_files
  )$size,
  stringsAsFactors = FALSE
) |>
  dplyr::arrange(
    relative_path
  )

readr::write_csv(
  output_manifest,
  file.path(
    output_dir,
    "validation",
    "output_manifest.csv"
  )
)

expected_figure_files <- c(
  "Figure2_annual_observed_encounter_rate.png",
  "Figure2_annual_observed_encounter_rate.tiff",
  "Figure3_reporting_probability.png",
  "Figure3_reporting_probability.tiff",
  "Figure4_elevational_centroid.png",
  "Figure4_elevational_centroid.tiff",
  "Figure5_local_extinction_elevation.png",
  "Figure5_local_extinction_elevation.tiff",
  "Figure6_local_extinction_map.png",
  "Figure6_local_extinction_map.tiff"
)

stopifnot(
  all(
    file.exists(
      file.path(
        output_dir,
        "figures",
        expected_figure_files
      )
    )
  ),
  file.exists(
    file.path(
      output_dir,
      "validation",
      "sessionInfo.txt"
    )
  ),
  file.exists(
    file.path(
      output_dir,
      "validation",
      "analysis_run_summary.csv"
    )
  )
)

cat(
  "\n============================================================\n",
  "The corrected-NDVI analysis was reproduced successfully.\n",
  "All tables, models, validation files, and manuscript Figures 2-6 were exported to:\n",
  output_dir,
  "\n============================================================\n",
  sep = ""
)

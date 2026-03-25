# Report helpers for the IQR outliers pipeline.

`%||%` <- function(x, y) if (!is.null(x)) x else y

# Pull in bootstrap + shared non-report helpers (same folder).
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
.this_dir <- if (exists("PIPELINE_PATH", inherits = TRUE)) {
  file.path(get("PIPELINE_PATH", inherits = TRUE), "utils")
} else if (!is.na(.this_file)) {
  dirname(.this_file)
} else {
  getwd()
}
source(file.path(.this_dir, "snt_dhis2_outliers_imputation_iqr.r"))

printdim <- function(df, name = deparse(substitute(df))) {
  if (is.null(df)) {
    message(sprintf("%s: NULL", name))
    return(invisible(NULL))
  }
  d <- dim(df)
  message(sprintf("%s: %s x %s", name, d[1], d[2]))
  invisible(d)
}

plot_outliers <- function(ind_name, df, outlier_col = "OUTLIER_DETECTED") {
  if (!ind_name %in% names(df)) return(NULL)
  if (!outlier_col %in% names(df)) return(NULL)

  d <- df %>%
    dplyr::mutate(
      YEAR = as.integer(.data$YEAR %||% substr(.data$PERIOD, 1, 4)),
      MONTH = as.integer(.data$MONTH %||% substr(.data$PERIOD, 5, 6)),
      DATE = as.Date(sprintf("%04d-%02d-01", YEAR, MONTH))
    ) %>%
    dplyr::group_by(.data$DATE) %>%
    dplyr::summarise(
      value = sum(.data[[ind_name]], na.rm = TRUE),
      has_outlier = any(.data[[outlier_col]] %in% TRUE, na.rm = TRUE),
      .groups = "drop"
    )

  ggplot2::ggplot(d, ggplot2::aes(x = .data$DATE, y = .data$value)) +
    ggplot2::geom_line(linewidth = 0.8, color = "grey40") +
    ggplot2::geom_point(ggplot2::aes(color = .data$has_outlier), size = 2, alpha = 0.9) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "#D55E00", `FALSE` = "#0072B2")) +
    ggplot2::labs(
      title = sprintf("Outliers - %s (%s)", ind_name, outlier_col),
      x = "Mois",
      y = "Valeur agregee",
      color = "Outlier present"
    ) +
    ggplot2::theme_minimal(base_size = 14)
}

plot_outliers_by_district_facet_year <- function(ind_name, df, outlier_col = "OUTLIER_DETECTED") {
  if (!ind_name %in% names(df)) return(NULL)
  if (!outlier_col %in% names(df)) return(NULL)
  if (!("ADM2_NAME" %in% names(df) && "ADM2_ID" %in% names(df))) return(NULL)

  d <- df %>%
    dplyr::mutate(
      YEAR = as.integer(.data$YEAR %||% substr(.data$PERIOD, 1, 4)),
      MONTH = as.integer(.data$MONTH %||% substr(.data$PERIOD, 5, 6)),
      DATE = as.Date(sprintf("%04d-%02d-01", YEAR, MONTH))
    ) %>%
    dplyr::group_by(.data$ADM2_ID, .data$ADM2_NAME, .data$YEAR, .data$MONTH, .data$DATE) %>%
    dplyr::summarise(
      value = sum(.data[[ind_name]], na.rm = TRUE),
      has_outlier = any(.data[[outlier_col]] %in% TRUE, na.rm = TRUE),
      .groups = "drop"
    )

  if (nrow(d) == 0) return(NULL)

  ggplot2::ggplot(
    d,
    ggplot2::aes(x = .data$DATE, y = .data$value, group = .data$ADM2_ID)
  ) +
    ggplot2::geom_line(alpha = 0.35, linewidth = 0.4, color = "grey40") +
    ggplot2::geom_point(ggplot2::aes(color = .data$has_outlier), alpha = 0.75, size = 1) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "#D55E00", `FALSE` = "grey70")) +
    ggplot2::facet_wrap(~YEAR, scales = "free_x") +
    ggplot2::labs(
      title = sprintf("Outliers par district - %s (%s)", ind_name, outlier_col),
      x = "Mois",
      y = "Valeur (ADM2 agrege)",
      color = "Outlier"
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      legend.position = "bottom",
      strip.text = ggplot2::element_text(face = "bold")
    )
}

plot_coherence_heatmap <- function(
  df,
  selected_year,
  agg_level = "ADM1_NAME",
  filename = NULL,
  do_plot = TRUE
) {
  if (!all(c("YEAR", "check_label", "pct_coherent") %in% names(df))) return(NULL)
  if (!agg_level %in% names(df)) return(NULL)

  d <- df %>%
    dplyr::mutate(YEAR = as.integer(.data$YEAR)) %>%
    dplyr::filter(.data$YEAR == as.integer(selected_year)) %>%
    dplyr::mutate(
      agg = as.character(.data[[agg_level]]),
      check_label = as.character(.data$check_label)
    )

  if (nrow(d) == 0) return(NULL)

  p <- ggplot2::ggplot(d, ggplot2::aes(
    x = .data$check_label,
    y = .data$agg,
    fill = .data$pct_coherent
  )) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_viridis_c(
      name = "% coherent",
      option = "viridis",
      limits = c(0, 100)
    ) +
    ggplot2::labs(
      title = sprintf("Coherence (%s) - %s", agg_level, selected_year),
      x = NULL,
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 30, hjust = 1),
      plot.title = ggplot2::element_text(face = "bold")
    )

  if (!is.null(filename)) {
    ggplot2::ggsave(filename = filename, plot = p, width = 14, height = 8, dpi = 150)
  }

  if (do_plot) print(p)
  invisible(p)
}

plot_coherence_map <- function(map_data, col_name, indicator_label = NULL) {
  if (!inherits(map_data, "sf")) return(NULL)
  if (!col_name %in% names(map_data)) return(NULL)

  ggplot2::ggplot(map_data) +
    ggplot2::geom_sf(ggplot2::aes(fill = .data[[col_name]]), color = NA) +
    ggplot2::scale_fill_viridis_c(
      option = "viridis",
      name = indicator_label %||% col_name,
      limits = c(0, 100),
      na.value = "grey90"
    ) +
    ggplot2::labs(title = indicator_label %||% col_name) +
    ggplot2::theme_void(base_size = 12) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", hjust = 0.5),
      legend.position = "right"
    )
}

get_coherence_definitions <- function() {
  checks <- list(
    allout_susp = c("ALLOUT", "SUSP"),
    allout_test = c("ALLOUT", "TEST"),
    susp_test = c("SUSP", "TEST"),
    test_conf = c("TEST", "CONF"),
    conf_treat = c("CONF", "MALTREAT"),
    adm_dth = c("MALADM", "MALDTH")
  )

  check_labels <- c(
    pct_coherent_allout_susp = "Ambulatoire >= Suspects",
    pct_coherent_allout_test = "Ambulatoire >= Testes",
    pct_coherent_susp_test = "Suspects >= Testes",
    pct_coherent_test_conf = "Testes >= Confirmes",
    pct_coherent_conf_treat = "Confirmes >= Traites",
    pct_coherent_adm_dth = "Admissions Palu >= Deces Palu"
  )

  list(checks = checks, check_labels = check_labels)
}

compute_national_coherency_metrics <- function(df, checks, check_labels) {
  df_checks <- df %>%
    dplyr::mutate(
      !!!lapply(names(checks), function(check_name) {
        cols <- checks[[check_name]]
        if (all(cols %in% names(df))) {
          rlang::expr(!!rlang::sym(cols[1]) >= !!rlang::sym(cols[2]))
        } else {
          rlang::expr(NA)
        }
      }) %>% stats::setNames(paste0("check_", names(checks)))
    )

  check_cols <- intersect(paste0("check_", names(checks)), names(df_checks))

  df_checks %>%
    dplyr::group_by(.data$YEAR) %>%
    dplyr::summarise(
      dplyr::across(
        dplyr::all_of(check_cols),
        ~ mean(.x, na.rm = TRUE) * 100,
        .names = "pct_{.col}"
      ),
      .groups = "drop"
    ) %>%
    tidyr::pivot_longer(
      cols = dplyr::starts_with("pct_"),
      names_to = "check_type",
      names_prefix = "pct_check_",
      values_to = "pct_coherent"
    ) %>%
    dplyr::filter(!is.na(.data$pct_coherent)) %>%
    dplyr::mutate(
      check_label = dplyr::recode(
        .data$check_type,
        !!!stats::setNames(check_labels, sub("^pct_coherent_", "", names(check_labels)))
      ),
      check_label = factor(.data$check_label, levels = unique(.data$check_label)),
      check_label = forcats::fct_reorder(.data$check_label, .data$pct_coherent, .fun = median, na.rm = TRUE)
    )
}

plot_national_coherence_heatmap <- function(coherency_metrics) {
  ggplot2::ggplot(coherency_metrics, ggplot2::aes(
    x = factor(.data$YEAR),
    y = .data$check_label,
    fill = .data$pct_coherent
  )) +
    ggplot2::geom_tile(color = NA, width = 0.88, height = 0.88) +
    ggplot2::geom_text(
      ggplot2::aes(label = sprintf("%.0f%%", .data$pct_coherent)),
      color = "white",
      fontface = "bold",
      size = 5
    ) +
    viridis::scale_fill_viridis(
      name = "% Coherent",
      option = "viridis",
      limits = c(0, 100),
      direction = -1
    ) +
    ggplot2::labs(
      title = "Controles de coherence des donnees (niveau national)",
      x = "Annee",
      y = NULL
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      panel.grid = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(size = 22, face = "bold", hjust = 0.5),
      axis.text.y = ggplot2::element_text(size = 16, hjust = 0),
      axis.text.x = ggplot2::element_text(size = 16),
      legend.title = ggplot2::element_text(size = 16, face = "bold"),
      legend.text = ggplot2::element_text(size = 14),
      legend.key.width = grid::unit(0.7, "cm"),
      legend.key.height = grid::unit(1.2, "cm")
    )
}

compute_adm_coherence_long <- function(df, checks, check_labels, min_reports = 5) {
  df_checks <- df %>%
    dplyr::mutate(
      !!!lapply(names(checks), function(check_name) {
        cols <- checks[[check_name]]
        if (all(cols %in% names(df))) {
          rlang::expr(!!rlang::sym(cols[1]) >= !!rlang::sym(cols[2]))
        } else {
          rlang::expr(NA_real_)
        }
      }) %>% stats::setNames(paste0("check_", names(checks)))
    )

  check_cols <- names(df_checks)[grepl("^check_", names(df_checks))]
  valid_checks <- check_cols[
    purrr::map_lgl(df_checks[check_cols], ~ !all(is.na(.x)))
  ]

  adm_coherence <- df_checks %>%
    dplyr::group_by(.data$ADM1_NAME, .data$ADM2_NAME, .data$ADM2_ID, .data$YEAR) %>%
    dplyr::summarise(
      total_reports = dplyr::n(),
      !!!purrr::map(
        valid_checks,
        ~ rlang::expr(100 * mean(.data[[.x]], na.rm = TRUE))
      ) %>%
        stats::setNames(paste0("pct_coherent_", sub("^check_", "", valid_checks))),
      .groups = "drop"
    ) %>%
    dplyr::filter(.data$total_reports >= min_reports)

  adm_long <- adm_coherence %>%
    tidyr::pivot_longer(
      cols = dplyr::starts_with("pct_coherent_"),
      names_to = "check_type",
      values_to = "pct_coherent"
    ) %>%
    dplyr::filter(!is.na(.data$pct_coherent)) %>%
    dplyr::mutate(check_label = dplyr::recode(.data$check_type, !!!check_labels))

  list(adm_coherence = adm_coherence, adm_long = adm_long)
}

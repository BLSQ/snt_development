# Report helpers for mean outliers imputation pipeline.
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
.this_dir <- if (!is.na(.this_file)) dirname(.this_file) else getwd()
source(file.path(.this_dir, "bootstrap.R"))
source(file.path(.this_dir, "reporting_utils.R"))

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


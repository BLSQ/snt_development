# Report helpers for median outliers imputation pipeline.
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
.this_dir <- if (exists("PIPELINE_PATH", inherits = TRUE)) {
    file.path(get("PIPELINE_PATH", inherits = TRUE), "utils")
} else if (!is.na(.this_file)) {
    dirname(.this_file)
} else {
    getwd()
}
source(file.path(.this_dir, "snt_dhis2_outliers_imputation_median.r"))

printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

plot_outliers <- function(ind_name, df, outlier_col) {
    df_ind <- df %>% dplyr::filter(INDICATOR == ind_name)
    df_ind <- df_ind %>% dplyr::filter(!is.na(YEAR), !is.na(VALUE), is.finite(VALUE))
    ggplot2::ggplot(df_ind, ggplot2::aes(x = YEAR, y = VALUE)) +
        ggplot2::geom_point(alpha = 0.25, color = "grey40", na.rm = TRUE) +
        ggplot2::geom_point(
            data = df_ind %>% dplyr::filter(.data[[outlier_col]] == TRUE),
            ggplot2::aes(x = YEAR, y = VALUE),
            color = "red",
            size = 2.8,
            alpha = 0.85,
            na.rm = TRUE
        ) +
        ggplot2::labs(
            title = paste("Outliers for indicator:", ind_name),
            subtitle = "Grey = all values, red = detected outliers",
            x = "Year",
            y = "Value"
        ) +
        ggplot2::theme_minimal(base_size = 14)
}

plot_outliers_by_district_facet_year <- function(ind_name, df, outlier_col) {
    df_ind <- df %>%
        dplyr::filter(
            INDICATOR == ind_name,
            !is.na(YEAR),
            !is.na(VALUE),
            is.finite(VALUE)
        )
    if (nrow(df_ind) == 0) {
        return(NULL)
    }
    ggplot2::ggplot(df_ind, ggplot2::aes(x = ADM2_ID, y = VALUE)) +
        ggplot2::geom_point(color = "grey60", alpha = 0.3) +
        ggplot2::geom_point(
            data = df_ind %>% dplyr::filter(.data[[outlier_col]] == TRUE),
            color = "red",
            size = 2.8,
            alpha = 0.85
        ) +
        ggplot2::facet_wrap(~ YEAR, scales = "free_y") +
        ggplot2::labs(
            title = paste("Outliers by district and year:", ind_name),
            x = "District",
            y = "Value"
        ) +
        ggplot2::theme_minimal(base_size = 12)
}

plot_coherence_heatmap <- function(df, selected_year, agg_level = "ADM1_NAME", filename = NULL, do_plot = TRUE) {
    if (!(agg_level %in% c("ADM1_NAME", "ADM2_NAME"))) stop("agg_level must be ADM1_NAME or ADM2_NAME")
    if (!all(c("INDICATOR", "YEAR", agg_level, "VALUE", "VALUE_IMPUTED") %in% colnames(df))) {
        stop("Data frame is missing required columns.")
    }
    comp <- df %>%
        dplyr::filter(YEAR == selected_year) %>%
        dplyr::group_by(INDICATOR, !!rlang::sym(agg_level)) %>%
        dplyr::summarise(
            coherence = ifelse(sum(!is.na(VALUE)) == 0, NA, sum(VALUE == VALUE_IMPUTED, na.rm = TRUE) / sum(!is.na(VALUE))),
            n = dplyr::n(),
            .groups = "drop"
        )
    p <- ggplot2::ggplot(comp, ggplot2::aes(x = .data[[agg_level]], y = INDICATOR, fill = coherence)) +
        ggplot2::geom_tile(color = "white", linewidth = 0.2) +
        ggplot2::scale_fill_gradient(low = "#fee5d9", high = "#a50f15", na.value = "grey90", limits = c(0, 1)) +
        ggplot2::labs(
            title = paste("Coherence heatmap -", agg_level, "-", selected_year),
            x = agg_level,
            y = "Indicator",
            fill = "Coherence"
        ) +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
    if (!is.null(filename)) ggplot2::ggsave(filename, p, width = 12, height = 6)
    if (isTRUE(do_plot)) print(p)
    invisible(p)
}

plot_coherence_map <- function(map_data, col_name, indicator_label = NULL) {
    if (!inherits(map_data, "sf")) stop("map_data must be an sf object.")
    if (!(col_name %in% names(map_data))) stop(paste("Column", col_name, "not found in map_data."))
    ttl <- ifelse(is.null(indicator_label), paste("Map of", col_name), paste("Map of", col_name, "-", indicator_label))
    ggplot2::ggplot(map_data) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data[[col_name]]), color = "grey30", linewidth = 0.1) +
        ggplot2::scale_fill_viridis_c(option = "C", na.value = "grey90") +
        ggplot2::labs(title = ttl, fill = col_name) +
        ggplot2::theme_minimal(base_size = 12)
}

get_coherence_definitions <- function() {
    checks <- list(
        "long_term" = function(x) (x >= 0.95),
        "short_term" = function(x) (x >= 0.95),
        "cyclicality" = function(x) (x >= 0.90),
        "volatility" = function(x) (x >= 0.90),
        "rolling_sd" = function(x) (x <= 0.80),
        "spatial" = function(x) (x <= 0.80),
        "residual" = function(x) (x <= 2),
        "trend_strength" = function(x) (x >= 0.20)
    )
    check_labels <- c(
        "long_term" = "Long-term (>= 95%)",
        "short_term" = "Short-term (>= 95%)",
        "cyclicality" = "Cyclicality (>= 90%)",
        "volatility" = "Volatility (>= 90%)",
        "rolling_sd" = "Rolling SD (<= 80%)",
        "spatial" = "Spatial (<= 80%)",
        "residual" = "Residual (<= 2)",
        "trend_strength" = "Trend strength (>= 20%)"
    )
    list(checks = checks, check_labels = check_labels)
}

compute_national_coherency_metrics <- function(df, checks, check_labels) {
    coherency_metrics <- purrr::imap_dfr(checks, function(cond, check_name) {
        vals <- df[[check_name]]
        tibble::tibble(
            check = check_name,
            label = check_labels[[check_name]],
            percent = round(100 * mean(cond(vals), na.rm = TRUE), 1)
        )
    })
    coherency_metrics$label <- factor(coherency_metrics$label, levels = rev(check_labels))
    coherency_metrics
}

plot_national_coherence_heatmap <- function(coherency_metrics) {
    ggplot2::ggplot(coherency_metrics, ggplot2::aes(x = 1, y = label, fill = percent)) +
        ggplot2::geom_tile(color = "white", width = 0.95, height = 0.9) +
        ggplot2::geom_text(ggplot2::aes(label = paste0(percent, "%")), size = 4, color = "black", fontface = "bold") +
        ggplot2::scale_fill_gradient2(
            low = "#f7fcf5", mid = "#74c476", high = "#00441b",
            midpoint = 85, limits = c(0, 100), name = "% indicators pass"
        ) +
        ggplot2::scale_x_continuous(expand = c(0, 0)) +
        ggplot2::labs(
            title = "National coherence overview",
            subtitle = "Percentage of indicators meeting each coherence criterion",
            x = NULL, y = NULL
        ) +
        ggplot2::theme_minimal(base_size = 13) +
        ggplot2::theme(
            axis.text.x = ggplot2::element_blank(),
            axis.ticks = ggplot2::element_blank(),
            panel.grid = ggplot2::element_blank(),
            legend.position = "right",
            plot.title = ggplot2::element_text(face = "bold"),
            plot.subtitle = ggplot2::element_text(color = "gray30"),
            axis.text.y = ggplot2::element_text(face = "bold")
        )
}

compute_adm_coherence_long <- function(df, checks, check_labels, min_reports = 5) {
    ADM_levels <- c("ADM1_NAME", "ADM2_NAME", "OU_NAME")
    adm_long <- lapply(ADM_levels, function(level) {
        df %>%
            dplyr::filter(!is.na(.data[[level]]), !is.na(INDICATOR)) %>%
            dplyr::group_by(.data[[level]], INDICATOR) %>%
            dplyr::summarise(
                dplyr::across(dplyr::all_of(names(checks)), ~ mean(checks[[cur_column()]](.x), na.rm = TRUE)),
                n_reports = dplyr::n(),
                .groups = "drop"
            ) %>%
            dplyr::filter(n_reports >= min_reports) %>%
            tidyr::pivot_longer(cols = dplyr::all_of(names(checks)), names_to = "check", values_to = "coherence_rate") %>%
            dplyr::mutate(level = level, label = check_labels[check])
    }) %>% dplyr::bind_rows()
    adm_long$label <- factor(adm_long$label, levels = rev(check_labels))
    adm_long
}

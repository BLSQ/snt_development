# ================================================
# Title: Report helpers for the mean outliers imputation pipeline
# Description: Plotting and coherence-metric helpers sourced by the reporting notebook.
# Dependencies: ggplot2, dplyr, tidyr, tibble, purrr, rlang, forcats, viridis, grid, stats, sf
# ================================================

# Load base snt utils
source(file.path("~/workspace", "code", "snt_utils.r"))

#' Null-Coalescing Operator
#'
#' Returns the left-hand side when it is not NULL, otherwise the right-hand side.
#'
#' @param x Any R object. Value returned when not NULL.
#' @param y Any R object. Fallback value used when `x` is NULL.
#' @return `x` when it is not NULL, otherwise `y`.
#'
#' @export
`%||%` <- function(x, y) if (!is.null(x)) x else y


#' Print the Dimensions of a Data Frame
#'
#' Writes the number of rows and columns of a table to the console, prefixed by
#' the table name, for quick inspection inside reporting notebooks.
#'
#' @param df Data frame. Table whose dimensions are printed.
#' @param name Character. Label used in the printed message. Defaults to the
#'   deparsed expression passed as `df`.
#' @return No return value, called for its console output.
#'
#' @export
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}


#' Plot Detected Outliers for One Indicator
#'
#' Draws a scatter plot of all values of a single indicator against the year,
#' highlighting in red the observations flagged as outliers.
#'
#' @param ind_name Character. Name of the indicator to plot (matched on INDICATOR).
#' @param df Data frame. Routine data containing INDICATOR, YEAR, VALUE and the
#'   outlier flag column.
#' @param outlier_col Character. Name of the logical column flagging outliers.
#' @return A ggplot object.
#'
#' @export
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


#' Plot Detected Outliers by District, Faceted by Year
#'
#' Draws one panel per year showing the values of a single indicator across
#' districts, highlighting in red the observations flagged as outliers.
#'
#' @param ind_name Character. Name of the indicator to plot (matched on INDICATOR).
#' @param df Data frame. Routine data containing INDICATOR, ADM2_ID, YEAR, VALUE
#'   and the outlier flag column.
#' @param outlier_col Character. Name of the logical column flagging outliers.
#' @return A ggplot object, or NULL when the indicator has no plottable rows.
#'
#' @export
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


#' Plot a Coherence Heatmap for a Single Year
#'
#' Draws a heatmap of the percentage of coherent records per coherence check and
#' aggregation unit for one year, optionally saving it to disk.
#'
#' @param df Data frame. Long coherence table with YEAR, check_label, pct_coherent
#'   and the aggregation column.
#' @param selected_year Integer or character. Year to display.
#' @param agg_level Character. Name of the aggregation column (e.g. "ADM1_NAME").
#' @param filename Character. Optional output path; the plot is saved when provided.
#' @param do_plot Logical. Print the plot when TRUE.
#' @return Invisibly, a ggplot object, or NULL when the required columns or rows
#'   are missing.
#'
#' @export
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


#' Map a Coherence Indicator
#'
#' Draws a choropleth map of one coherence column over the supplied spatial
#' features, on a fixed 0-100 scale.
#'
#' @param map_data sf object. Spatial features carrying the coherence column.
#' @param col_name Character. Name of the column to map.
#' @param indicator_label Character. Optional label used for the title and legend;
#'   defaults to `col_name`.
#' @return A ggplot object, or NULL when `map_data` is not an sf object or the
#'   column is missing.
#'
#' @export
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


#' Get the Coherence Check Definitions
#'
#' Returns the pairs of indicators compared by each coherence check, together
#' with the human-readable labels used in reports.
#'
#' @return Named list with two elements: `checks`, a named list of indicator
#'   pairs, and `check_labels`, a named character vector of display labels.
#'
#' @export
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


#' Compute National Coherence Metrics
#'
#' Evaluates each coherence check on the routine data and summarises, per year,
#' the percentage of records satisfying it.
#'
#' @param df Data frame. Routine data in wide format, with YEAR and one column
#'   per indicator involved in the checks.
#' @param checks Named list. Indicator pairs per check, as returned by
#'   `get_coherence_definitions()`.
#' @param check_labels Named character vector. Display labels per check.
#' @return Data frame with YEAR, check_type, pct_coherent and check_label; empty
#'   when none of the checks can be evaluated.
#'
#' @export
compute_national_coherence_metrics <- function(df, checks, check_labels) {
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
    if (length(check_cols) == 0) {
        return(tibble::tibble(
            YEAR = integer(),
            check_type = character(),
            pct_coherent = numeric(),
            check_label = factor()
        ))
    }

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
            check_label = forcats::fct_reorder(
                .data$check_label,
                .data$pct_coherent,
                .fun = median,
                na.rm = TRUE
            )
        )
}


#' Plot the National Coherence Heatmap
#'
#' Draws a year-by-check heatmap of national coherence percentages, annotated
#' with the rounded percentage inside each tile.
#'
#' @param coherence_metrics Data frame. Output of
#'   `compute_national_coherence_metrics()`.
#' @return A ggplot object.
#'
#' @export
plot_national_coherence_heatmap <- function(coherence_metrics) {
    ggplot2::ggplot(coherence_metrics, ggplot2::aes(
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


#' Compute District-Level Coherence Metrics
#'
#' Evaluates each coherence check per district and year, keeping only districts
#' with enough reports, and returns both the wide and long representations.
#'
#' @param df Data frame. Routine data in wide format, with ADM1_NAME, ADM2_NAME,
#'   ADM2_ID, YEAR and one column per indicator involved in the checks.
#' @param checks Named list. Indicator pairs per check, as returned by
#'   `get_coherence_definitions()`.
#' @param check_labels Named character vector. Display labels per check.
#' @param min_reports Integer. Minimum number of reports required to keep a
#'   district-year. Defaults to 5.
#' @return Named list with `adm_coherence` (one row per district-year, one
#'   percentage column per check) and `adm_long` (the same data pivoted long).
#'
#' @export
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
    if (length(valid_checks) == 0) {
        adm_coherence <- df_checks %>%
            dplyr::group_by(.data$ADM1_NAME, .data$ADM2_NAME, .data$ADM2_ID, .data$YEAR) %>%
            dplyr::summarise(total_reports = dplyr::n(), .groups = "drop") %>%
            dplyr::filter(.data$total_reports >= min_reports)
        adm_long <- tibble::tibble(
            ADM1_NAME = character(),
            ADM2_NAME = character(),
            ADM2_ID = character(),
            YEAR = integer(),
            total_reports = integer(),
            check_type = character(),
            pct_coherent = numeric(),
            check_label = character()
        )
        return(list(adm_coherence = adm_coherence, adm_long = adm_long))
    }

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

# ================================================
# Title: Report helpers for the Magic Glasses outliers imputation pipeline
# Description: Method label, outlier summary and bar-chart helpers sourced by the reporting
#   notebook.
# Dependencies: dplyr, ggplot2, scales, stats
# ================================================

# Load base utils
source(file.path("~/workspace", "code", "snt_utils.r"))


#' Get the Display Label of a Magic Glasses Method
#'
#' Maps the OUTLIER_METHOD value written by the MG pipeline to the French label used
#' in the report. Unknown values are returned unchanged.
#'
#' @param method Character. OUTLIER_METHOD value (e.g. "MAGIC_GLASSES_PARTIAL").
#' @return Character. Label for display.
#'
#' @export
get_mg_method_label <- function(method) {
    labels <- c(
        MAGIC_GLASSES_PARTIAL = "MG partiel (MAD15 → MAD10)",
        MAGIC_GLASSES_COMPLETE = "MG complet (MAD15 → MAD10 → seasonal5 → seasonal3)"
    )
    ifelse(method %in% names(labels), labels[method], method)
}


#' Summarise Detected Outliers by a Grouping Column
#'
#' Counts, for each value of `by_col` (or overall when `by_col` is NULL), the number of
#' reported values, the number of outliers and the share of reported values flagged as
#' outliers. Missing values cannot be flagged, so they are excluded from the
#' denominator.
#'
#' @param detected_tbl Data frame. Detection table with VALUE and OUTLIER_DETECTED.
#' @param by_col Character or NULL. Column to group by (e.g. "INDICATOR", "YEAR").
#'   Default: NULL (one overall row).
#' @return Data frame with `by_col` (if given), N_VALUES, N_OUTLIERS and PCT_OUTLIERS,
#'   sorted by decreasing N_OUTLIERS when grouped by INDICATOR, by `by_col` otherwise.
#'
#' @export
summarise_outliers_by <- function(detected_tbl, by_col = NULL) {
    group_cols <- if (is.null(by_col)) character(0) else by_col
    summary_tbl <- detected_tbl %>%
        dplyr::group_by(dplyr::across(dplyr::all_of(group_cols))) %>%
        dplyr::summarise(
            N_VALUES = sum(!is.na(VALUE)),
            N_OUTLIERS = sum(OUTLIER_DETECTED, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        dplyr::mutate(PCT_OUTLIERS = ifelse(N_VALUES > 0, round(N_OUTLIERS / N_VALUES * 100, 3), NA_real_))

    if (identical(by_col, "INDICATOR")) {
        summary_tbl <- summary_tbl %>% dplyr::arrange(dplyr::desc(N_OUTLIERS))
    } else if (!is.null(by_col)) {
        summary_tbl <- summary_tbl %>% dplyr::arrange(.data[[by_col]])
    }
    summary_tbl
}


#' Plot the Number of Outliers by a Grouping Column
#'
#' Draws a bar chart of N_OUTLIERS for each value of `by_col`, from the output of
#' `summarise_outliers_by()`.
#'
#' @param summary_tbl Data frame. Output of `summarise_outliers_by()` with `by_col`.
#' @param by_col Character. Grouping column shown on the x axis.
#' @param title Character. Plot title.
#' @param subtitle Character. Plot subtitle.
#' @param x_label Character. Label of the grouping axis.
#' @param horizontal Logical. Draw horizontal bars, ordered by count. Default: FALSE.
#' @return ggplot object.
#'
#' @export
plot_outliers_by <- function(summary_tbl, by_col, title, subtitle, x_label, horizontal = FALSE) {
    plot_tbl <- summary_tbl %>% dplyr::mutate(GROUP = as.character(.data[[by_col]]))
    if (horizontal) {
        plot_tbl <- plot_tbl %>% dplyr::mutate(GROUP = stats::reorder(GROUP, N_OUTLIERS))
    }

    p <- ggplot2::ggplot(plot_tbl, ggplot2::aes(x = GROUP, y = N_OUTLIERS)) +
        ggplot2::geom_col(fill = "#2c7fb8", alpha = 0.85, width = 0.7) +
        ggplot2::scale_y_continuous(labels = scales::label_number(big.mark = " ")) +
        ggplot2::labs(title = title, subtitle = subtitle, x = x_label, y = "Nombre d'outliers") +
        ggplot2::theme_minimal(base_size = 12)

    if (horizontal) {
        p <- p + ggplot2::coord_flip()
    }
    p
}

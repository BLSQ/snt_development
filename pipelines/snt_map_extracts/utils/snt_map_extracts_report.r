# ================================================
# Title: Report helpers for the MAP extracts pipeline
# Description: Selection of the MAP metrics to map and choropleth map helpers sourced by the
#   reporting notebook.
# Dependencies: dplyr, sf, ggplot2, gridExtra, glue
# Requires: code/snt_utils.r, loaded by the reporting notebook before this file.
# ================================================


#' Get the MAP Metric Columns
#'
#' Returns the metric columns of the wide MAP table: every column that has a population-weighted
#' companion `{METRIC}_POP_W`. The order follows the table's column order.
#'
#' @param map_data Data frame. Wide MAP table, one value column and one `_POP_W` column per metric.
#' @return Character vector of metric column names, without the `_POP_W` suffix.
#'
#' @export
get_map_metric_columns <- function(map_data) {
    pop_w_cols <- grep("_POP_W$", names(map_data), value = TRUE)
    metric_cols <- sub("_POP_W$", "", pop_w_cols)
    intersect(names(map_data), metric_cols)
}


#' Select the MAP Rows of One Year
#'
#' Filters the wide MAP table to the given `YEAR`. Stops with an `[ERROR]` message if no row
#' matches.
#'
#' @param map_data Data frame. Wide MAP table with a `YEAR` column.
#' @param year Integer. Year to keep.
#' @return Data frame. One row per ADM2.
#'
#' @export
select_map_year_data <- function(map_data, year) {
    year_data <- map_data %>%
        dplyr::filter(YEAR == year)
    if (nrow(year_data) == 0) {
        stop(glue::glue("[ERROR] No MAP data for YEAR {year}."))
    }
    year_data
}


#' Build One Choropleth Map
#'
#' Draws `value_col` of `map_sf` per ADM2, with a viridis fill scale bounded by `limits` so that
#' several maps can share the same colour scale.
#'
#' @param map_sf sf object. ADM2 shapes joined with the values to map.
#' @param value_col Character. Column to map.
#' @param title Character. Map title.
#' @param subtitle Character. Map subtitle.
#' @param limits Numeric vector of length 2. Fill scale limits.
#' @param na_color Character. Fill colour for missing values. Default: "#D3D3D3".
#' @return ggplot object.
#'
#' @export
build_metric_map <- function(map_sf, value_col, title, subtitle, limits, na_color = "#D3D3D3") {
    ggplot2::ggplot(map_sf) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data[[value_col]]), color = "white") +
        ggplot2::scale_fill_viridis_c(option = "C", limits = limits, na.value = na_color) +
        ggplot2::labs(title = title, subtitle = subtitle, fill = "Valeur") +
        ggplot2::theme_void() +
        ggplot2::theme(
            plot.title = ggplot2::element_text(face = "bold", size = 10, margin = ggplot2::margin(b = 4)),
            plot.subtitle = ggplot2::element_text(size = 8, colour = "grey40", margin = ggplot2::margin(b = 8)),
            legend.position = "right",
            legend.text = ggplot2::element_text(size = 8),
            plot.margin = ggplot2::margin(10, 10, 10, 10)
        )
}


#' Build and Save the MAP Metric Maps
#'
#' For each metric, draws the median (`{METRIC}`) and the population-weighted median
#' (`{METRIC}_POP_W`) side by side on a shared colour scale, and saves the pair as
#' `{country_code}_map_{METRIC}_{year}.png`. A metric with no value in either column is skipped
#' with a warning.
#'
#' @param map_sf sf object. ADM2 shapes joined with one year of the wide MAP table.
#' @param metric_cols Character vector. Metric columns to map, without the `_POP_W` suffix.
#' @param year Integer. Year shown in the map subtitles and the filenames.
#' @param figures_path Character. Folder where the maps are saved.
#' @param country_code Character. Country code used as the filename prefix.
#' @return Character vector. Paths of the saved maps, in the order of `metric_cols`.
#'
#' @export
save_map_metric_plots <- function(map_sf, metric_cols, year, figures_path, country_code) {
    out_files <- c()
    for (metric in metric_cols) {
        pop_w_col <- paste0(metric, "_POP_W")
        values <- c(map_sf[[metric]], map_sf[[pop_w_col]])
        if (all(is.na(values))) {
            log_msg(glue::glue("No value for {metric} in {year}. Map skipped."), "warning")
            next
        }
        limits <- range(values, na.rm = TRUE)

        median_map <- build_metric_map(map_sf, metric, metric, glue::glue("Médiane - {year}"), limits)
        pop_w_map <- build_metric_map(
            map_sf, pop_w_col, metric, glue::glue("Médiane pondérée par la population - {year}"), limits
        )
        combined_plot <- gridExtra::arrangeGrob(median_map, pop_w_map, ncol = 2)

        out_file <- file.path(figures_path, glue::glue("{country_code}_map_{metric}_{year}.png"))
        ggplot2::ggsave(out_file, plot = combined_plot, width = 14, height = 6, dpi = 150, bg = "white")
        log_msg(glue::glue("Map saved: {out_file}"))
        out_files <- c(out_files, out_file)
    }
    out_files
}

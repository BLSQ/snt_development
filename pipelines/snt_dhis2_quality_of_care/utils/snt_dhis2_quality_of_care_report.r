# ================================================
# Title: Report helpers for the Quality of Care pipeline
# Description: Summary table, summary outputs and year-level chart helpers sourced by the
#   reporting notebook.
# Dependencies: data.table, arrow, ggplot2, gridExtra, scales, glue
# Requires: code/snt_utils.r, loaded by the reporting notebook before this file.
# ================================================


#' Load the Latest Quality of Care District-Year Output
#'
#' Finds the `{country_code}_quality_of_care_district_year_{imputed|removed}.parquet`
#' files in the output folder and reads the most recently modified one. Stops
#' with an `[ERROR]` message if none is found.
#'
#' @param output_data_path Character. Folder holding the quality-of-care data outputs.
#' @param country_code Character. Country code used as the filename prefix.
#' @return Named list with `qoc` (data.table, the loaded output) and `latest_file`
#'   (character, the path of the file read).
#'
#' @export
load_latest_quality_of_care_output <- function(output_data_path, country_code) {
    files <- list.files(
        output_data_path,
        pattern = paste0("^", country_code, "_quality_of_care_district_year_(imputed|removed)\\.parquet$"),
        full.names = TRUE
    )
    if (length(files) == 0) {
        stop(glue::glue("[ERROR] No quality_of_care parquet found in {output_data_path}"))
    }
    latest_file <- files[which.max(file.info(files)$mtime)]
    qoc <- data.table::as.data.table(arrow::read_parquet(latest_file))
    list(qoc = qoc, latest_file = latest_file)
}


#' Build the Year-Level Quality of Care Summary Table
#'
#' Aggregates the district-year table to one row per year: rate indicators are
#' averaged across districts and absolute indicators are summed, ignoring NA
#' values. Indicators absent from the input are skipped.
#'
#' @param qoc_dt data.table. District-year quality-of-care indicators.
#' @return data.table. One row per `YEAR`, ordered by `YEAR`.
#'
#' @export
build_quality_of_care_summary <- function(qoc_dt) {
    mean_cols <- c("TESTING_RATE", "TREATMENT_RATE", "CASE_FATALITY_RATE", "PROP_ADM_MALARIA", "PROP_MALARIA_DEATHS")
    sum_cols  <- c("NON_MALARIA_ALL_CAUSE_OUTPATIENTS", "PRESUMED_CASES")

    summary_tbl <- unique(qoc_dt[, .(YEAR)])

    for (col in intersect(mean_cols, names(qoc_dt))) {
        agg <- qoc_dt[, setNames(list(mean(get(col), na.rm = TRUE)), col), by = .(YEAR)]
        summary_tbl <- merge(summary_tbl, agg, by = "YEAR", all.x = TRUE)
    }

    for (col in intersect(sum_cols, names(qoc_dt))) {
        agg <- qoc_dt[, setNames(list(sum(get(col), na.rm = TRUE)), col), by = .(YEAR)]
        summary_tbl <- merge(summary_tbl, agg, by = "YEAR", all.x = TRUE)
    }

    summary_tbl[order(YEAR)]
}


#' Save the Year-Level Quality of Care Summary Outputs
#'
#' Writes the summary table as `{country_code}_quality_of_care_summary` in
#' parquet and csv format (no Excel, to avoid extra dependencies) and logs the
#' saved paths.
#'
#' @param summary_tbl data.table. Year-level summary table.
#' @param report_outputs_path Character. Reporting outputs folder.
#' @param country_code Character. Country code used as the filename prefix.
#' @return Named list with the `summary_parquet` and `summary_csv` file paths.
#'
#' @export
save_quality_of_care_summary_outputs <- function(summary_tbl, report_outputs_path, country_code) {
    summary_parquet <- file.path(report_outputs_path, glue::glue("{country_code}_quality_of_care_summary.parquet"))
    summary_csv     <- file.path(report_outputs_path, glue::glue("{country_code}_quality_of_care_summary.csv"))

    arrow::write_parquet(summary_tbl, summary_parquet)
    data.table::fwrite(summary_tbl, summary_csv)

    log_msg(glue::glue("Summary data saved to: {summary_parquet}, {summary_csv}"))
    list(summary_parquet = summary_parquet, summary_csv = summary_csv)
}


#' Build and Save the Year-Level Quality of Care Chart Panel
#'
#' Draws one bar chart per available indicator (rates as percentages, absolute
#' indicators as counts), arranges them in a two-column panel and saves it as
#' `{country_code}_quality_of_care_by_year.png`.
#'
#' @param summary_tbl data.table. Year-level summary table.
#' @param figures_path Character. Folder where the chart panel is saved.
#' @param country_code Character. Country code used as the filename prefix.
#' @return Character. Path of the saved chart, or NULL if the summary is empty or
#'   no indicator column is available.
#'
#' @export
save_quality_of_care_summary_charts <- function(summary_tbl, figures_path, country_code) {
    plot_data <- data.table::copy(summary_tbl)
    if (nrow(plot_data) == 0) return(NULL)

    make_pct_plot <- function(col_name, title_name) {
        ggplot2::ggplot(plot_data, ggplot2::aes(x = factor(YEAR), y = .data[[col_name]] * 100)) +
            ggplot2::geom_bar(stat = "identity", fill = "#2563eb", color = "#1e40af", width = 0.7) +
            ggplot2::geom_text(ggplot2::aes(label = paste0(round(.data[[col_name]] * 100, 1), "%")), vjust = -0.5, size = 2.5) +
            ggplot2::labs(title = title_name, x = "Annee", y = "%") +
            ggplot2::theme_minimal() +
            ggplot2::theme(
                plot.title = ggplot2::element_text(face = "bold", size = 10),
                axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 9),
                panel.grid.major.y = ggplot2::element_line(linetype = "dashed", color = scales::alpha("grey", 0.7)),
                plot.background = ggplot2::element_rect(fill = "#fafafa", color = NA),
                panel.background = ggplot2::element_rect(fill = "#fafafa", color = NA),
                plot.margin = ggplot2::margin(5, 5, 5, 5)
            ) +
            ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.1)))
    }

    make_abs_plot <- function(col_name, title_name) {
        format_label <- function(v) {
            ifelse(
                is.na(v) | v == 0,
                "0",
                ifelse(v >= 1e6, paste0(round(v / 1e6, 2), "M"), format(round(v), big.mark = " ", scientific = FALSE))
            )
        }
        ggplot2::ggplot(plot_data, ggplot2::aes(x = factor(YEAR), y = .data[[col_name]])) +
            ggplot2::geom_bar(stat = "identity", fill = "#2563eb", color = "#1e40af", width = 0.7) +
            ggplot2::geom_text(ggplot2::aes(label = format_label(.data[[col_name]])), vjust = -0.5, size = 2.5) +
            ggplot2::labs(title = title_name, x = "Annee", y = "Nombre") +
            ggplot2::theme_minimal() +
            ggplot2::theme(
                plot.title = ggplot2::element_text(face = "bold", size = 10),
                axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, size = 9),
                panel.grid.major.y = ggplot2::element_line(linetype = "dashed", color = scales::alpha("grey", 0.7)),
                plot.background = ggplot2::element_rect(fill = "#fafafa", color = NA),
                panel.background = ggplot2::element_rect(fill = "#fafafa", color = NA),
                plot.margin = ggplot2::margin(5, 5, 5, 5)
            ) +
            ggplot2::scale_y_continuous(labels = scales::comma, expand = ggplot2::expansion(mult = c(0, 0.1)))
    }

    plots_list <- list()
    if ("TESTING_RATE" %in% names(plot_data)) plots_list[["TESTING_RATE"]] <- make_pct_plot("TESTING_RATE", "Testing rate (TEST / SUSP)")
    if ("TREATMENT_RATE" %in% names(plot_data)) plots_list[["TREATMENT_RATE"]] <- make_pct_plot("TREATMENT_RATE", "Treatment rate (MALTREAT / CONF)")
    if ("CASE_FATALITY_RATE" %in% names(plot_data)) plots_list[["CASE_FATALITY_RATE"]] <- make_pct_plot("CASE_FATALITY_RATE", "Case fatality rate (MALDTH / MALADM)")
    if ("PROP_ADM_MALARIA" %in% names(plot_data)) plots_list[["PROP_ADM_MALARIA"]] <- make_pct_plot("PROP_ADM_MALARIA", "Prop. admissions paludisme (MALADM / ALLADM)")
    if ("PROP_MALARIA_DEATHS" %in% names(plot_data)) plots_list[["PROP_MALARIA_DEATHS"]] <- make_pct_plot("PROP_MALARIA_DEATHS", "Prop. deces paludisme (MALDTH / ALLDTH)")
    if ("PRESUMED_CASES" %in% names(plot_data)) plots_list[["PRESUMED_CASES"]] <- make_abs_plot("PRESUMED_CASES", "Cas presumes (PRES)")
    if ("NON_MALARIA_ALL_CAUSE_OUTPATIENTS" %in% names(plot_data)) plots_list[["NON_MALARIA_ALL_CAUSE_OUTPATIENTS"]] <- make_abs_plot("NON_MALARIA_ALL_CAUSE_OUTPATIENTS", "Consultations externes non-paludisme (ALLOUT)")

    if (length(plots_list) == 0) return(NULL)

    plot_order <- c("TESTING_RATE", "TREATMENT_RATE", "CASE_FATALITY_RATE", "PROP_ADM_MALARIA", "PROP_MALARIA_DEATHS", "PRESUMED_CASES", "NON_MALARIA_ALL_CAUSE_OUTPATIENTS")
    available_plots <- plots_list[intersect(plot_order, names(plots_list))]
    n_plots <- length(available_plots)
    ncol_layout <- 2
    nrow_layout <- ceiling(n_plots / ncol_layout)

    combined_plot <- do.call(gridExtra::grid.arrange, c(available_plots, ncol = ncol_layout, nrow = nrow_layout))
    out_file <- file.path(figures_path, glue::glue("{country_code}_quality_of_care_by_year.png"))
    ggplot2::ggsave(out_file, plot = combined_plot, width = 18, height = max(8, 5.2 * nrow_layout), dpi = 300, bg = "white", units = "in")
    log_msg(glue::glue("Combined bar charts saved: {out_file}"))
    out_file
}

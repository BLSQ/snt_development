# Quality of Care **reporting** helpers (used by the reporting notebook).

source(
    file.path(
        "~/workspace",
        "pipelines",
        "snt_dhis2_quality_of_care",
        "utils",
        "snt_dhis2_quality_of_care.r"
    )
)


#' Load latest Quality of Care district-year output.
load_latest_quality_of_care_output <- function(output_data_path, country_code) {
    files <- list.files(output_data_path, pattern = paste0("^", country_code, "_quality_of_care_district_year_(imputed|removed)\\.parquet$"), full.names = TRUE)
    if (length(files) == 0) {
        stop(glue::glue("[ERROR] No quality_of_care parquet found in {output_data_path}"))
    }
    latest_file <- files[which.max(file.info(files)$mtime)]
    qoc <- data.table::as.data.table(arrow::read_parquet(latest_file))
    list(qoc = qoc, latest_file = latest_file)
}


#' Build year-level Quality of Care summary table.
build_quality_of_care_summary <- function(qoc_dt) {
    summary_tbl <- unique(qoc_dt[, .(YEAR)])

    if ("testing_rate" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(testing_rate = mean(testing_rate, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)
    if ("treatment_rate" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(treatment_rate = mean(treatment_rate, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)
    if ("case_fatality_rate" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(case_fatality_rate = mean(case_fatality_rate, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)
    if ("prop_adm_malaria" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(prop_adm_malaria = mean(prop_adm_malaria, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)
    if ("prop_malaria_deaths" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(prop_malaria_deaths = mean(prop_malaria_deaths, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)
    if ("non_malaria_all_cause_outpatients" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(non_malaria_all_cause_outpatients = sum(non_malaria_all_cause_outpatients, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)
    if ("presumed_cases" %in% names(qoc_dt)) summary_tbl <- merge(summary_tbl, qoc_dt[, .(presumed_cases = sum(presumed_cases, na.rm = TRUE)), by = .(YEAR)], by = "YEAR", all.x = TRUE)

    summary_tbl[order(YEAR)]
}


#' Save year-level summary outputs for report consumption.
save_quality_of_care_summary_outputs <- function(summary_tbl, report_outputs_path, country_code) {
    summary_parquet <- file.path(report_outputs_path, glue::glue("{country_code}_quality_of_care_summary.parquet"))
    summary_csv <- file.path(report_outputs_path, glue::glue("{country_code}_quality_of_care_summary.csv"))
    summary_xlsx <- file.path(report_outputs_path, glue::glue("{country_code}_quality_of_care_summary.xlsx"))

    arrow::write_parquet(summary_tbl, summary_parquet)
    data.table::fwrite(summary_tbl, summary_csv)
    writexl::write_xlsx(list(summary = as.data.frame(summary_tbl)), summary_xlsx)

    log_msg(glue::glue("Summary data saved to: {summary_parquet}, {summary_csv}, {summary_xlsx}"))
    list(summary_parquet = summary_parquet, summary_csv = summary_csv, summary_xlsx = summary_xlsx)
}


#' Build and save year-level bar chart panel for QoC indicators.
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
    if ("testing_rate" %in% names(plot_data)) plots_list[["testing_rate"]] <- make_pct_plot("testing_rate", "Testing rate (TEST / SUSP)")
    if ("treatment_rate" %in% names(plot_data)) plots_list[["treatment_rate"]] <- make_pct_plot("treatment_rate", "Treatment rate (MALTREAT / CONF)")
    if ("case_fatality_rate" %in% names(plot_data)) plots_list[["case_fatality_rate"]] <- make_pct_plot("case_fatality_rate", "Case fatality rate (MALDTH / MALADM)")
    if ("prop_adm_malaria" %in% names(plot_data)) plots_list[["prop_adm_malaria"]] <- make_pct_plot("prop_adm_malaria", "Prop. admissions paludisme (MALADM / ALLADM)")
    if ("prop_malaria_deaths" %in% names(plot_data)) plots_list[["prop_malaria_deaths"]] <- make_pct_plot("prop_malaria_deaths", "Prop. deces paludisme (MALDTH / ALLDTH)")
    if ("presumed_cases" %in% names(plot_data)) plots_list[["presumed_cases"]] <- make_abs_plot("presumed_cases", "Cas presumes (PRES)")
    if ("non_malaria_all_cause_outpatients" %in% names(plot_data)) plots_list[["non_malaria_all_cause_outpatients"]] <- make_abs_plot("non_malaria_all_cause_outpatients", "Consultations externes non-paludisme (ALLOUT)")

    if (length(plots_list) == 0) return(NULL)

    plot_order <- c("testing_rate", "treatment_rate", "case_fatality_rate", "prop_adm_malaria", "prop_malaria_deaths", "presumed_cases", "non_malaria_all_cause_outpatients")
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

# Load shared SNT helpers.
source(file.path("~/workspace", "code", "snt_utils.r"))


#' Bootstrap context for Quality of Care notebooks.
#'
#' Centralizes path definitions, package loading, and OpenHEXA import.
#'
#' @param root_path Workspace root path. Defaults to `~/workspace`.
#' @param required_packages Character vector of packages to install/load.
#' @return Named list containing paths, config, dataset IDs, and country code.
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate")
) {
    install_and_load(packages)

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)

    list(
        CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),
        FORMATTED_DATA_PATH = file.path(SNT_ROOT_PATH, "data", "dhis2", "extracts_formatted"),
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads")
    )
}

#' Validate quality-of-care action parameter.
#'
#' @param data_action Action string expected to be `imputed` or `removed`.
#' @return Validated action string.
validate_quality_of_care_action <- function(data_action) {
    if (is.null(data_action) || !nzchar(data_action)) {
        return("imputed")
    }
    allowed_actions <- c("imputed", "removed")
    if (!(data_action %in% allowed_actions)) {
        stop(glue::glue("[ERROR] Invalid data_action `{data_action}`. Allowed: {paste(allowed_actions, collapse = ', ')}"))
    }
    data_action
}

#' Compute district-year Quality of Care indicators.
#'
#' @param routine Routine dataframe loaded from outliers dataset.
#' @return Data table with district-year indicators.
normalize_qoc_routine_types <- function(routine) {
    data.table::setDT(routine)
    indicator_cols <- c("TEST", "SUSP", "MALTREAT", "CONF", "MALDTH", "MALADM", "ALLADM", "ALLDTH", "ALLOUT", "PRES")
    available_cols <- intersect(indicator_cols, names(routine))

    for (col in available_cols) {
        col_vals <- as.character(routine[[col]])
        col_vals[is.na(col_vals) | col_vals == "" | col_vals == "-"] <- NA_character_
        routine[, (col) := as.numeric(col_vals)]
    }

    routine[, YEAR := as.integer(YEAR)]
    routine[, ADM2_ID := as.character(ADM2_ID)]
    routine
}

#' Aggregate QoC routine indicators by district and year.
#'
#' @param routine Routine data table with normalized types.
#' @return Aggregated district-year data table.
aggregate_qoc_district_year <- function(routine) {
    indicator_cols <- c("TEST", "SUSP", "MALTREAT", "CONF", "MALDTH", "MALADM", "ALLADM", "ALLDTH", "ALLOUT", "PRES")
    available_cols <- intersect(indicator_cols, names(routine))

    if (length(available_cols) > 0) {
        routine[, lapply(.SD, function(x) sum(x, na.rm = TRUE)), .SDcols = available_cols, by = .(ADM2_ID, YEAR)]
    } else {
        unique(routine[, .(ADM2_ID, YEAR)])
    }
}

#' Add derived quality-of-care indicators to aggregated district-year data.
#'
#' @param qoc Aggregated district-year data table.
#' @return Data table with derived indicators.
add_quality_of_care_derived_indicators <- function(qoc) {
    if ("TEST" %in% names(qoc) && "SUSP" %in% names(qoc)) qoc[, testing_rate := data.table::fifelse(SUSP > 0, TEST / SUSP, NA_real_)]
    if ("MALTREAT" %in% names(qoc) && "CONF" %in% names(qoc)) qoc[, treatment_rate := data.table::fifelse(CONF > 0, MALTREAT / CONF, NA_real_)]
    if ("MALDTH" %in% names(qoc) && "MALADM" %in% names(qoc)) qoc[, case_fatality_rate := data.table::fifelse(MALADM > 0, MALDTH / MALADM, NA_real_)]
    if ("MALADM" %in% names(qoc) && "ALLADM" %in% names(qoc)) qoc[, prop_adm_malaria := data.table::fifelse(ALLADM > 0, MALADM / ALLADM, NA_real_)]
    if ("MALDTH" %in% names(qoc) && "ALLDTH" %in% names(qoc)) {
        qoc[, prop_malaria_deaths := data.table::fifelse(ALLDTH > 0, MALDTH / ALLDTH, NA_real_)]
        qoc[, prop_deaths_malaria := prop_malaria_deaths]
    }
    if ("ALLOUT" %in% names(qoc)) qoc[, non_malaria_all_cause_outpatients := ALLOUT]
    if ("PRES" %in% names(qoc)) qoc[, presumed_cases := PRES]

    qoc
}

#' Compute district-year Quality of Care indicators.
#'
#' @param routine Routine dataframe loaded from outliers dataset.
#' @return Data table with district-year indicators.
compute_quality_of_care_indicators <- function(routine) {
    core_cols <- c("ADM2_ID", "YEAR")
    core_missing <- setdiff(core_cols, names(routine))
    if (length(core_missing) > 0) {
        stop(glue::glue("[ERROR] Missing core columns: {paste(core_missing, collapse = ', ')}"))
    }

    indicator_cols <- c("TEST", "SUSP", "MALTREAT", "CONF", "MALDTH", "MALADM", "ALLADM", "ALLDTH", "ALLOUT", "PRES")
    missing_cols <- setdiff(indicator_cols, names(routine))
    if (length(missing_cols) > 0) {
        log_msg(glue::glue("[WARNING] Missing indicator columns: {paste(missing_cols, collapse = ', ')}"), level = "warning")
    }

    routine <- normalize_qoc_routine_types(routine)
    qoc <- aggregate_qoc_district_year(routine)
    qoc <- add_quality_of_care_derived_indicators(qoc)
    qoc
}

#' Merge ADM2 labels into Quality of Care outputs.
#'
#' @param qoc_dt Quality-of-care data table.
#' @param shapes_sf Shapes sf table.
#' @return Data table with optional ADM2_NAME.
attach_quality_of_care_shapes <- function(qoc_dt, shapes_sf) {
    shapes_dt <- data.table::as.data.table(sf::st_drop_geometry(shapes_sf))
    if ("ADM2_ID" %in% names(shapes_dt) && "ADM2_NAME" %in% names(shapes_dt)) {
        shapes_dt[, ADM2_ID := as.character(ADM2_ID)]
        qoc_dt <- merge(qoc_dt, unique(shapes_dt[, .(ADM2_ID, ADM2_NAME)]), by = "ADM2_ID", all.x = TRUE)
    }
    qoc_dt
}

#' Save district-year Quality of Care outputs.
#'
#' @param qoc_dt Computed quality-of-care data table.
#' @param output_data_path Output directory path.
#' @param country_code Country code.
#' @param data_action Action suffix for output naming.
#' @return Named list with `parquet` and `csv` output file paths.
save_quality_of_care_outputs <- function(qoc_dt, output_data_path, country_code, data_action) {
    out_district_parquet <- file.path(output_data_path, glue::glue("{country_code}_quality_of_care_district_year_{data_action}.parquet"))
    out_district_csv <- file.path(output_data_path, glue::glue("{country_code}_quality_of_care_district_year_{data_action}.csv"))

    arrow::write_parquet(qoc_dt, out_district_parquet)
    data.table::fwrite(qoc_dt, out_district_csv)
    log_msg(glue::glue("Saved outputs: {out_district_parquet}, {out_district_csv}"))

    list(parquet = out_district_parquet, csv = out_district_csv)
}

#' Generate and save yearly district maps for QoC indicators.
#'
#' @param qoc_dt Quality-of-care data table.
#' @param shapes_sf District shapes sf.
#' @param figures_path Folder where PNG maps are written.
#' @return Invisibly returns `TRUE`.
save_quality_of_care_maps <- function(qoc_dt, shapes_sf, figures_path) {
    shapes_sf$ADM2_ID <- as.character(shapes_sf$ADM2_ID)
    qoc_dt$ADM2_ID <- as.character(qoc_dt$ADM2_ID)

    plot_yearly_map <- function(df, sf_shapes, value_col, title_prefix, filename_prefix, is_rate = TRUE) {
        if (!(value_col %in% names(df))) return(invisible(NULL))
        sf_shapes_local <- sf_shapes
        years <- sort(unique(df$YEAR))

        for (yr in years) {
            tryCatch(
                {
                    df_y <- df[YEAR == yr]
                    if (nrow(df_y) == 0) return(invisible(NULL))
                    df_y$ADM2_ID <- as.character(df_y$ADM2_ID)
                    map_df <- dplyr::left_join(sf_shapes_local, df_y, by = "ADM2_ID")
                    if (!(value_col %in% names(map_df))) return(invisible(NULL))

                    vals <- map_df[[value_col]]
                    finite_vals <- vals[is.finite(vals) & !is.na(vals)]
                    if (length(finite_vals) == 0) return(invisible(NULL))

                    if (is_rate) {
                        cat_vals <- cut(vals, breaks = c(-Inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0, Inf), labels = c("<0", "0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", ">1.0"), include.lowest = TRUE)
                        fill_palette <- "YlOrRd"
                    } else {
                        if (length(finite_vals) > 4) {
                            br <- unique(as.numeric(quantile(finite_vals, probs = seq(0, 1, 0.2), na.rm = TRUE)))
                            if (length(br) < 2) {
                                cat_vals <- as.factor(rep("all", nrow(map_df)))
                            } else {
                                cat_vals <- cut(vals, breaks = br, include.lowest = TRUE)
                            }
                        } else {
                            cat_vals <- as.factor(vals)
                        }
                        fill_palette <- "Blues"
                    }

                    map_df <- dplyr::mutate(map_df, cat = as.factor(cat_vals))
                    p <- ggplot2::ggplot(map_df) +
                        ggplot2::geom_sf(ggplot2::aes(fill = cat), color = "grey60", size = 0.1) +
                        ggplot2::scale_fill_brewer(palette = fill_palette, na.value = "white", drop = FALSE) +
                        ggplot2::theme_void() +
                        ggplot2::labs(title = paste0(title_prefix, " - ", yr), fill = value_col, caption = "Source: SNT DHIS2 outliers-imputed routine data") +
                        ggplot2::theme(legend.position = "bottom", plot.title = ggplot2::element_text(face = "bold", size = 12))

                    out_png <- file.path(figures_path, glue::glue("{filename_prefix}_{yr}.png"))
                    ggplot2::ggsave(out_png, plot = p, width = 9, height = 7, dpi = 300, bg = "white")
                    log_msg(glue::glue("Saved map: {out_png}"))
                },
                error = function(e) {
                    log_msg(glue::glue("[WARNING] Failed to build/save map for `{value_col}` year `{yr}`: {conditionMessage(e)}"), level = "warning")
                }
            )
        }
    }

    if ("testing_rate" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "testing_rate", "Testing rate (TEST / SUSP)", "testing_rate", TRUE)
    if ("treatment_rate" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "treatment_rate", "Treatment rate (MALTREAT / CONF)", "treatment_rate", TRUE)
    if ("case_fatality_rate" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "case_fatality_rate", "In-hospital case fatality rate (MALDTH / MALADM)", "case_fatality_rate", TRUE)
    if ("prop_adm_malaria" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "prop_adm_malaria", "Proportion admitted for malaria (MALADM / ALLADM)", "prop_adm_malaria", TRUE)
    if ("prop_malaria_deaths" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "prop_malaria_deaths", "Proportion of malaria deaths (MALDTH / ALLDTH)", "prop_malaria_deaths", TRUE)
    if ("non_malaria_all_cause_outpatients" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "non_malaria_all_cause_outpatients", "Non-malaria all-cause outpatients (ALLOUT)", "allout", FALSE)
    if ("presumed_cases" %in% names(qoc_dt)) plot_yearly_map(qoc_dt, shapes_sf, "presumed_cases", "Presumed cases (PRES)", "presumed_cases", FALSE)

    log_msg(glue::glue("Saved yearly maps in: {figures_path}"))
    invisible(TRUE)
}

#' Load latest Quality of Care district-year output.
#'
#' @param output_data_path Path to quality-of-care data outputs.
#' @param country_code Country code.
#' @return Data table loaded from latest matching parquet.
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
#'
#' @param qoc_dt Quality-of-care district-year data table.
#' @return Year-level summary table.
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
#'
#' @param summary_tbl Summary table.
#' @param report_outputs_path Reporting outputs folder.
#' @param country_code Country code.
#' @return Named list with summary file paths.
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
#'
#' @param summary_tbl Year-level summary table.
#' @param figures_path Folder where the combined chart is saved.
#' @param country_code Country code used in output file name.
#' @return Path to saved chart (or NULL if nothing to plot).
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

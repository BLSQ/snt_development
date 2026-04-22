# Quality of Care shared R helpers.
#
# This file is the module used by the QoC *pipeline code* notebook:
# - bootstrap paths + install/load packages
# - config + dataset loading
# - map export utility used by both the pipeline and reporting notebooks
# - compute helpers for district-year outputs
#
# Reporting-only helpers live in:
# - snt_dhis2_quality_of_care_report.r

# Load shared SNT helpers.
source(file.path("~/workspace", "code", "snt_utils.r"))


#' Bootstrap context for Quality of Care notebooks.
#'
#' Returns base workspace paths only.
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c("arrow", "dplyr", "tidyr", "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate")
) {
    install_and_load(packages)

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)

    paths_to_check <- list(
        CONFIG_PATH = file.path(SNT_ROOT_PATH, "configuration"),
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH = file.path(SNT_ROOT_PATH, "data"),
        PIPELINES_PATH = file.path(SNT_ROOT_PATH, "pipelines")
    )

    for (target_path in paths_to_check) {
        dir.create(target_path, recursive = TRUE, showWarnings = FALSE)
    }

    setup_variable <- c(
        list(paths_to_check = paths_to_check),
        paths_to_check
    )

    return(setup_variable)
}


#' Load SNT configuration file.
load_snt_config <- function(config_path, config_file_name = "SNT_config.json") {
    config_file <- file.path(config_path, config_file_name)
    config_json <- tryCatch(
        {
            jsonlite::fromJSON(config_file)
        },
        error = function(e) {
            stop(glue::glue("[ERROR] Error while loading configuration from `{config_file}`: {conditionMessage(e)}"))
        }
    )
    log_msg(paste0("SNT configuration loaded from: ", config_file))
    return(config_json)
}


#' Load dataset file from OpenHEXA.
load_dataset_file <- function(dataset_id, filename, verbose = TRUE) {
    if (!exists("openhexa", inherits = TRUE) || is.null(get("openhexa", inherits = TRUE))) {
        stop("[ERROR] OpenHEXA SDK is not available. Run `get_setup_variables()` before loading dataset files.")
    }

    data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_id, filename)
        },
        error = function(e) {
            stop(glue::glue("[ERROR] Error while loading {filename} file from dataset: {dataset_id}"))
        }
    )

    if (verbose) {
        log_msg(glue::glue(
            "{filename} data loaded from dataset : {dataset_id} dataframe dimensions: [{paste(dim(data), collapse = ', ')}]"
        ))
    }

    return(data)
}


#' Generate and save yearly district maps for QoC indicators.
#'
#' This is used by both the pipeline notebook and the reporting notebook.
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


#' Validate quality-of-care action parameter.
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


#' Coerce `indicator_cols` to numeric; YEAR and ADM2_ID types if those columns exist.
normalize_qoc_routine_types <- function(routine, indicator_cols) {
    data.table::setDT(routine)
    indicator_cols <- as.character(indicator_cols)
    available_cols <- intersect(indicator_cols, names(routine))

    for (col in available_cols) {
        col_vals <- as.character(routine[[col]])
        col_vals[is.na(col_vals) | col_vals == "" | col_vals == "-"] <- NA_character_
        routine[, (col) := suppressWarnings(as.numeric(col_vals))]
    }

    if ("YEAR" %in% names(routine)) routine[, YEAR := as.integer(YEAR)]
    if ("ADM2_ID" %in% names(routine)) routine[, ADM2_ID := as.character(ADM2_ID)]
    routine
}


#' Sum `indicator_cols` by `group_cols` (default ADM2_ID, YEAR).
aggregate_qoc_district_year <- function(routine, indicator_cols, group_cols = c("ADM2_ID", "YEAR")) {
    group_cols <- as.character(group_cols)
    indicator_cols <- as.character(indicator_cols)
    available_cols <- intersect(indicator_cols, names(routine))

    if (length(available_cols) > 0) {
        routine[, lapply(.SD, function(x) sum(x, na.rm = TRUE)), .SDcols = available_cols, by = group_cols]
    } else {
        unique(routine[, ..group_cols])
    }
}


#' Add derived quality-of-care indicators to aggregated district-year data.
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


#' Merge ADM2 labels into Quality of Care outputs.
attach_quality_of_care_shapes <- function(qoc_dt, shapes_sf) {
    shapes_dt <- data.table::as.data.table(sf::st_drop_geometry(shapes_sf))
    if ("ADM2_ID" %in% names(shapes_dt) && "ADM2_NAME" %in% names(shapes_dt)) {
        shapes_dt[, ADM2_ID := as.character(ADM2_ID)]
        qoc_dt <- merge(qoc_dt, unique(shapes_dt[, .(ADM2_ID, ADM2_NAME)]), by = "ADM2_ID", all.x = TRUE)
    }
    qoc_dt
}


#' Save district-year Quality of Care outputs.
save_quality_of_care_outputs <- function(qoc_dt, output_data_path, country_code, data_action) {
    out_district_parquet <- file.path(output_data_path, glue::glue("{country_code}_quality_of_care_district_year_{data_action}.parquet"))
    out_district_csv <- file.path(output_data_path, glue::glue("{country_code}_quality_of_care_district_year_{data_action}.csv"))

    arrow::write_parquet(qoc_dt, out_district_parquet)
    data.table::fwrite(qoc_dt, out_district_csv)
    log_msg(glue::glue("Saved outputs: {out_district_parquet}, {out_district_csv}"))

    list(parquet = out_district_parquet, csv = out_district_csv)
}

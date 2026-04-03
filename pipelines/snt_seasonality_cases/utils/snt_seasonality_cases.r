#' Bootstrap runtime context for seasonality-cases pipeline.
#'
#' Loads shared helpers/packages, initializes OpenHEXA SDK handle, and creates
#' output/intermediate directories used by the notebook workflow.
#'
#' @param root_path Root workspace path.
#' @param required_packages Character vector of required R packages.
#' @param load_openhexa Whether to import OpenHEXA SDK.
#' @return Named list with paths and OpenHEXA handle.
bootstrap_seasonality_cases_context <- function(
    root_path = "~/workspace",
    required_packages = c(
        "jsonlite", "data.table", "ggplot2", "fpp3", "arrow", "glue",
        "sf", "RColorBrewer", "httr", "reticulate"
    ),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    output_data_path <- file.path(root_path, "data", "seasonality_cases")
    intermediate_results_path <- file.path(output_data_path, "intermediate_results")
    dir.create(output_data_path, recursive = TRUE, showWarnings = FALSE)
    dir.create(intermediate_results_path, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }
    assign("openhexa", openhexa, envir = .GlobalEnv)

    list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        OUTPUT_DATA_PATH = output_data_path,
        INTERMEDIATE_RESULTS_PATH = intermediate_results_path,
        openhexa = openhexa
    )
}

#' Load routine and spatial inputs for seasonality-cases.
#'
#' Downloads country-specific `*_shapes.geojson` and `*_routine.parquet` from
#' the configured dataset, with explicit logging and stop-on-error behavior.
#'
#' @param dataset_name Source dataset identifier/name.
#' @param country_code Country code used in filenames.
#' @return List with `spatial_data` and `original_dt`.
load_seasonality_input_data <- function(dataset_name, country_code) {
    spatial_filename <- paste(country_code, "shapes.geojson", sep = "_")
    routine_filename <- paste(country_code, "routine.parquet", sep = "_")

    spatial_data <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, spatial_filename)
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading DHIS2 shapes file for {country_code}: {conditionMessage(e)}")
            log_msg(msg, level = "error")
            stop(msg)
        }
    )
    log_msg(glue::glue("File {spatial_filename} successfully loaded from dataset version: {dataset_name}"))

    original_dt <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, routine_filename)
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading DHIS2 routine file for {country_code}: {conditionMessage(e)}")
            log_msg(msg, level = "error")
            stop(msg)
        }
    )
    log_msg(glue::glue("File {routine_filename} successfully loaded from dataset version: {dataset_name}"))

    list(spatial_data = spatial_data, original_dt = original_dt)
}

#' Compute mean seasonal block proportion of annual cases.
#'
#' For each year and admin unit, computes the maximum forward block sum for the
#' selected duration, divides by annual total cases, and returns the mean ratio.
#'
#' @param admin_id Target administrative unit id.
#' @param block_duration Seasonal block duration in months.
#' @param row_data Row-level seasonality computations.
#' @param annual_data Annual totals by admin/year.
#' @param admin_col Column name for administrative unit id.
#' @param year_column Column name for year.
#' @return Numeric proportion (or NA when not computable).
compute_cases_proportion <- function(admin_id, block_duration, row_data, annual_data, admin_col, year_column) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_real_)
    }

    sum_col <- paste("CASES_SUM", block_duration, "MTH_FW", sep = "_")
    if (!sum_col %in% names(row_data)) {
        return(NA_real_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    admin_annual_data <- annual_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0 || nrow(admin_annual_data) == 0) {
        return(NA_real_)
    }

    yearly_max_block <- admin_row_data[
        !is.na(get(sum_col)),
        .(max_block_sum = if (.N > 0L) max(get(sum_col), na.rm = TRUE) else NA_real_),
        by = year_column
    ]

    yearly_max_block <- yearly_max_block[is.finite(max_block_sum)]
    if (nrow(yearly_max_block) == 0) {
        return(NA_real_)
    }

    merged <- merge(yearly_max_block, admin_annual_data, by = year_column)
    merged <- merged[ANNUAL_TOTAL > 0]
    if (nrow(merged) == 0) {
        return(NA_real_)
    }

    merged[, prop := max_block_sum / ANNUAL_TOTAL]
    mean(merged$prop, na.rm = TRUE)
}


#' Compute dominant seasonal start month for an admin unit.
#'
#' Uses row-level seasonality flags for a selected block duration and returns
#' the most frequent month marked as seasonal.
#'
#' @param admin_id Target administrative unit id.
#' @param block_duration Seasonal block duration in months.
#' @param row_data Row-level seasonality computations.
#' @param admin_col Column name for administrative unit id.
#' @param month_column Column name for month.
#' @return Integer month (1-12) or NA when not computable.
compute_start_month <- function(admin_id, block_duration, row_data, admin_col, month_column) {
    if (is.na(block_duration) || is.infinite(block_duration)) {
        return(NA_integer_)
    }

    seasonality_row_col <- paste("CASES", block_duration, "MTH_ROW_SEASONALITY", sep = "_")
    if (!seasonality_row_col %in% names(row_data)) {
        return(NA_integer_)
    }

    admin_row_data <- row_data[get(admin_col) == admin_id]
    if (nrow(admin_row_data) == 0) {
        return(NA_integer_)
    }

    seasonal_months <- admin_row_data[get(seasonality_row_col) == 1, get(month_column)]
    if (length(seasonal_months) == 0) {
        return(NA_integer_)
    }

    month_counts <- table(seasonal_months)
    as.integer(names(month_counts)[which.max(month_counts)])
}

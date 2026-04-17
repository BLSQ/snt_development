# Load base utils
# Helpers are named so the dataset-method reporting notebook reads like a checklist
# (same idea as `snt_dhis2_reporting_rate_dataelement` utils).
source(file.path("~/workspace/code", "snt_utils.r"))


# JSON reader for this pipeline only (`snt_utils.r` unchanged).
read_workspace_json_file <- function(json_path, resource_label = "JSON file") {
    json_path <- as.character(json_path)[[1L]]
    tryCatch(
        jsonlite::fromJSON(json_path),
        error = function(e) {
            stop(paste0(
                "[ERROR] Error while loading ",
                resource_label,
                " from `",
                json_path,
                "`: ",
                conditionMessage(e)
            ))
        }
    )
}


#' Get Setup Variables for SNT Workspace
#' Initializes workspace paths, loads R packages, and imports OpenHEXA SDK.
#'
#' @param SNT_ROOT_PATH Character. Root path of the SNT workspace. Default: '~/workspace'
#' @param packages Character vector. R packages to install and load.
#' @return List with `paths_to_check` (CONFIG_PATH, UPLOADS_PATH, DATA_PATH) and the
#'   same three paths at the top level for backward compatibility (`setup$CONFIG_PATH`, …).
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c(
        "arrow", "dplyr", "tidyr", "ggplot2",
        "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate"
    )
) {
    paths_to_check <- list(
        CONFIG_PATH  = file.path(SNT_ROOT_PATH, "configuration"),
        UPLOADS_PATH = file.path(SNT_ROOT_PATH, "uploads"),
        DATA_PATH    = file.path(SNT_ROOT_PATH, "data")
    )
    setup_variable <- c(
        list(paths_to_check = paths_to_check),
        paths_to_check
    )

    install_and_load(packages)

    if (Sys.getenv("PROJ_LIB", "") == "") {
        Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    }
    if (Sys.getenv("GDAL_DATA", "") == "") {
        Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    }
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)

    return(setup_variable)
}


#' Load SNT Configuration File
#' @param snt_config_path Character. Full path to `SNT_config.json`.
#' @export
load_snt_config <- function(snt_config_path) {
    config_json <- read_workspace_json_file(snt_config_path, "configuration")
    log_msg(paste0("SNT configuration loaded from: ", snt_config_path))
    return(config_json)
}


#' Fail if Papermill did not inject `ROUTINE_FILE` and `DATASET_ID`.
#' @export
stop_if_dataset_reporting_papermill_params_missing <- function() {
    required <- c("ROUTINE_FILE", "DATASET_ID")
    missing <- required[!vapply(required, exists, logical(1), inherits = TRUE)]
    if (length(missing) > 0) {
        stop(
            "[ERROR] Missing pipeline parameters (Papermill): ",
            paste(missing, collapse = ", "),
            ". Expected only ROUTINE_FILE and DATASET_ID from `snt_dhis2_reporting_rate_dataset`."
        )
    }
}


# Legacy alias.
assert_papermill_reporting_rate_dataset_params <- stop_if_dataset_reporting_papermill_params_missing


#' Country, admins, and product filter from `SNT_config.json` (dataset-method RR).
read_dataset_reporting_identity_from_config <- function(config_json) {
    list(
        COUNTRY_CODE = config_json$SNT_CONFIG$COUNTRY_CODE,
        ADMIN_1 = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_1),
        ADMIN_2 = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_2),
        REPORTING_RATE_PRODUCT_ID = config_json$SNT_CONFIG$REPORTING_RATE_PRODUCT_UID
    )
}


#' Column names kept when trimming routine extracts to reporting-rate grain.
fixed_columns_for_dataset_reporting_rate_routine_slice <- function() {
    c("YEAR", "MONTH", "ADM2_ID", "REPORTING_RATE")
}


#' Build the named settings list used by the dataset-method reporting-rate notebook.
#'
#' Calls `stop_if_dataset_reporting_papermill_params_missing()` first, then reads
#' country / admins / product UID and the fixed routine column list from `config_json`.
#'
#' @export
build_dataset_method_reporting_settings_from_config <- function(config_json) {
    stop_if_dataset_reporting_papermill_params_missing()
    id <- read_dataset_reporting_identity_from_config(config_json)
    c(id, list(fixed_cols_rr = fixed_columns_for_dataset_reporting_rate_routine_slice()))
}


# Legacy alias (same as removed `parse_reporting_rate_dataset_snt_settings`).
parse_reporting_rate_dataset_snt_settings <- build_dataset_method_reporting_settings_from_config


#' Load Dataset File from OpenHEXA
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @param verbose Logical. If TRUE, log dataframe dimensions after a successful load.
#' @export
load_dataset_file <- function(dataset_id, filename, verbose = TRUE) {
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


#' Save final dataset-method reporting-rate table as CSV + Parquet under `data/dhis2/reporting_rate/`.
#' @export
save_dataset_method_reporting_rate_csv_and_parquet <- function(reporting_rate_tbl, snt_environment, country_code) {
    output_dir <- file.path(snt_environment$DATA_PATH, "dhis2", "reporting_rate")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    csv_path <- file.path(output_dir, paste0(country_code, "_reporting_rate_dataset.csv"))
    pq_path <- file.path(output_dir, paste0(country_code, "_reporting_rate_dataset.parquet"))
    utils::write.csv(reporting_rate_tbl, csv_path, row.names = FALSE)
    log_msg(glue::glue("Exported: {csv_path}"))
    arrow::write_parquet(reporting_rate_tbl, pq_path)
    log_msg(glue::glue("Exported: {pq_path}"))
    invisible(list(csv_path = csv_path, parquet_path = pq_path))
}


# Legacy alias.
write_reporting_rate_dataset_outputs <- save_dataset_method_reporting_rate_csv_and_parquet

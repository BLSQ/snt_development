# Load base utils
# Bootstrap matches `snt_dhis2_population_transformation`: fixed-path `source()` of this
# file, `snt_environment <- get_setup_variables()`, then `load_snt_config()`.
# Keep helpers small and reusable; pipeline-specific assignments stay in notebooks.
source(file.path("~/workspace/code", "snt_utils.r"))


# JSON reader for this pipeline only (`snt_utils.r` must stay untouched per project rules).
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
#' @return List with `paths_to_check` plus `CONFIG_PATH`, `UPLOADS_PATH`, `DATA_PATH`
#'   (use as `snt_environment$CONFIG_PATH`, same pattern as population transformation).
#'
#' @export
get_setup_variables <- function(
    SNT_ROOT_PATH = "~/workspace",
    packages = c(
        "arrow", "rlang", "dplyr", "tidyr", "lubridate", "ggplot2",
        "stringr", "stringi", "jsonlite", "httr", "glue", "reticulate", "zoo"
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

    configure_conda_r_spatial_env()
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    reticulate::py_config()$python
    assign("openhexa", reticulate::import("openhexa.sdk"), envir = .GlobalEnv)

    return(setup_variable)
}


#' Load SNT Configuration File
#' Reads and parses a JSON configuration file.
#' @param snt_config_path Character. Path to the configuration JSON file.
#' @return List containing parsed configuration.
#'
#' @export
load_snt_config <- function(snt_config_path) {
    config_json <- read_workspace_json_file(snt_config_path, "configuration")
    log_msg(paste0("SNT configuration loaded from: ", snt_config_path))
    return(config_json)
}


#' Load SNT Metadata File
#' Reads and parses `SNT_metadata.json` (or another workspace metadata JSON).
#' @param snt_metadata_path Character. Path to the metadata JSON file.
#' @return List containing parsed metadata.
#'
#' @export
load_snt_metadata <- function(snt_metadata_path) {
    metadata_json <- read_workspace_json_file(snt_metadata_path, "SNT metadata")
    log_msg(paste0("SNT metadata loaded from: ", snt_metadata_path))
    return(metadata_json)
}


#' Load Dataset File from OpenHEXA
#' Retrieves the latest version of a file from an OpenHEXA dataset.
#'
#' @param dataset_id Character. OpenHEXA dataset identifier.
#' @param filename Character. Name of file to load.
#' @param verbose Logical. If TRUE, log dataframe dimensions after a successful load.
#' @return Dataframe containing the loaded data.
#'
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


#' Conda-friendly defaults for PROJ/GDAL (used when reading spatial data).
configure_conda_r_spatial_env <- function() {
    if (Sys.getenv("PROJ_LIB", "") == "") {
        Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    }
    if (Sys.getenv("GDAL_DATA", "") == "") {
        Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    }
}


#' Fail if Papermill did not inject `ROUTINE_FILE` and `DATASET_ID`.
assert_papermill_dataelement_params <- function() {
    required_pm <- c("ROUTINE_FILE", "DATASET_ID")
    missing_pm <- required_pm[!vapply(required_pm, exists, logical(1), inherits = TRUE)]
    if (length(missing_pm) > 0) {
        stop(
            "[ERROR] Missing pipeline parameters (Papermill): ",
            paste(missing_pm, collapse = ", "),
            ". Expected only ROUTINE_FILE and DATASET_ID from `snt_dhis2_reporting_rate_dataelement`."
        )
    }
}


activity_indicator_list_is_nonempty <- function(activity_indicators) {
    length(activity_indicators) > 0L
}


#' Stop early if the analyst left the activity-indicator list empty.
#' @export
stop_if_activity_indicators_empty <- function(activity_indicators) {
    if (!activity_indicator_list_is_nonempty(activity_indicators)) {
        stop("[ERROR] No activity indicators selected; choose at least one (e.g. CONF).")
    }
    invisible(TRUE)
}


#' Return required columns that are missing from `data`.
#' @export
find_missing_columns <- function(data, required_columns) {
    if (!is.data.frame(data)) {
        stop("[ERROR] `data` must be a data.frame.")
    }
    required_columns <- as.character(unlist(required_columns, use.names = FALSE))
    required_columns <- required_columns[!is.na(required_columns) & nzchar(required_columns)]
    required_columns <- unique(required_columns)
    setdiff(required_columns, names(data))
}


#' Validate that required columns exist in `data`.
#'
#' Returns missing columns invisibly. Behavior on missing columns is controlled by
#' `on_missing`: `"error"`, `"warning"`, or `"none"`.
#' @export
validate_required_columns <- function(
    data,
    required_columns,
    data_label = "data",
    on_missing = c("error", "warning", "none")
) {
    on_missing <- match.arg(on_missing)
    missing_columns <- find_missing_columns(data, required_columns)
    if (length(missing_columns) == 0L) {
        return(invisible(character(0)))
    }

    msg <- glue::glue(
        "{data_label} missing required column(s): {paste(missing_columns, collapse = ', ')}"
    )

    if (on_missing == "error") {
        log_msg(paste0("[ERROR] ", msg), "error")
        stop(paste0("[ERROR] ", msg))
    }
    if (on_missing == "warning") {
        log_msg(paste0("Warning: ", msg), "warning")
    }
    invisible(missing_columns)
}


#' First / last PERIOD in routine and full vector of YYYYMM months in between.
#' @export
summarize_routine_period_range_as_month_vector <- function(dhis2_routine) {
    period_start <- min(dhis2_routine$PERIOD, na.rm = TRUE)
    period_end <- max(dhis2_routine$PERIOD, na.rm = TRUE)
    pv <- format(
        seq(lubridate::ym(period_start), lubridate::ym(period_end), by = "month"),
        "%Y%m"
    )
    list(
        PERIOD_START = period_start,
        PERIOD_END = period_end,
        period_vector = pv
    )
}


#' Pyramid table crossed with every month in the routine period (facility master for RR).
#' @export
build_facilities_crossed_with_monthly_periods <- function(
    dhis2_pyramid_formatted,
    period_vector,
    config_json,
    ADMIN_1,
    ADMIN_2
) {
    dhis2_pyramid_formatted %>%
        dplyr::rename(
            OU_ID = glue::glue("LEVEL_{config_json$SNT_CONFIG$ANALYTICS_ORG_UNITS_LEVEL}_ID"),
            OU_NAME = glue::glue("LEVEL_{config_json$SNT_CONFIG$ANALYTICS_ORG_UNITS_LEVEL}_NAME"),
            ADM2_ID = stringr::str_replace(ADMIN_2, "NAME", "ID"),
            ADM2_NAME = dplyr::all_of(ADMIN_2),
            ADM1_ID = stringr::str_replace(ADMIN_1, "NAME", "ID"),
            ADM1_NAME = dplyr::all_of(ADMIN_1)
        ) %>%
        dplyr::select(ADM1_ID, ADM1_NAME, ADM2_ID, ADM2_NAME, OU_ID, OU_NAME, OPENING_DATE, CLOSED_DATE) %>%
        dplyr::distinct() %>%
        tidyr::crossing(PERIOD = period_vector) %>%
        dplyr::mutate(PERIOD = as.numeric(PERIOD))
}

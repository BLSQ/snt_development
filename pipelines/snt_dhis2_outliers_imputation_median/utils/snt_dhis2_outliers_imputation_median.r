# Main helpers for median outliers imputation pipeline.
bootstrap_outliers_context <- function(
    root_path = "~/workspace",
    required_packages = c(
        "data.table", "arrow", "tidyverse", "jsonlite", "DBI", "RPostgres",
        "reticulate", "glue", "zoo"
    ),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")
    output_dir <- file.path(data_path, "dhis2", "outliers_imputation")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }

    config_json <- tryCatch(
        {
            jsonlite::fromJSON(file.path(config_path, "SNT_config.json"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading configuration {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    return(list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        OUTPUT_DIR = output_dir,
        openhexa = openhexa,
        config_json = config_json
    ))
}

impute_outliers_dt <- function(dt, outlier_col) {
    dt <- data.table::as.data.table(dt)
    data.table::setorder(dt, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, PERIOD, YEAR, MONTH)
    dt[, TO_IMPUTE := data.table::fifelse(get(outlier_col) == TRUE, NA_real_, VALUE)]
    dt[, MEDIAN_IMPUTED := data.table::frollapply(
        TO_IMPUTE,
        n = 3,
        FUN = function(x) ceiling(median(x, na.rm = TRUE)),
        align = "center"
    ), by = .(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)]
    dt[, VALUE_IMPUTED := data.table::fifelse(is.na(TO_IMPUTE), MEDIAN_IMPUTED, TO_IMPUTE)]
    dt[, c("TO_IMPUTE", "MEDIAN_IMPUTED") := NULL]
    return(as.data.frame(data.table::copy(dt)))
}

format_routine_data_selection <- function(
    df,
    outlier_column,
    DHIS2_INDICATORS,
    fixed_cols,
    pyramid_names,
    remove = FALSE
) {
    if (remove) {
        df <- df %>% dplyr::filter(!.data[[outlier_column]])
    }
    target_cols <- c(
        "PERIOD", "YEAR", "MONTH", "ADM1_NAME", "ADM1_ID",
        "ADM2_NAME", "ADM2_ID", "OU_ID", "OU_NAME", DHIS2_INDICATORS
    )
    output <- df %>%
        dplyr::select(-VALUE) %>%
        dplyr::rename(VALUE = VALUE_IMPUTED) %>%
        dplyr::select(dplyr::all_of(fixed_cols), INDICATOR, VALUE) %>%
        dplyr::mutate(VALUE = ifelse(is.nan(VALUE), NA_real_, VALUE)) %>%
        tidyr::pivot_wider(names_from = "INDICATOR", values_from = "VALUE") %>%
        dplyr::left_join(pyramid_names, by = c("ADM1_ID", "ADM2_ID", "OU_ID"))
    return(output %>% dplyr::select(dplyr::all_of(intersect(target_cols, names(output)))))
}


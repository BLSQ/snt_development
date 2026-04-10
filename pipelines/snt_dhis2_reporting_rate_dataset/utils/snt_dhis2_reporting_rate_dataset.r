select_routine_dataset_name_dataset <- function(ROUTINE_FILE, COUNTRY_CODE, config_json) {
    if (ROUTINE_FILE == glue::glue("{COUNTRY_CODE}_routine.parquet")) {
        return(config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED)
    }
    config_json$SNT_DATASET_IDENTIFIERS$DHIS2_OUTLIERS_IMPUTATION
}


load_routine_data_dataset <- function(rountine_dataset_name, ROUTINE_FILE, COUNTRY_CODE, fixed_cols_rr) {
    dhis2_routine <- tryCatch({
        get_latest_dataset_file_in_memory(rountine_dataset_name, ROUTINE_FILE)
    }, error = function(e) {
        msg <- paste("Error while loading DHIS2 routine data file for: ", COUNTRY_CODE, conditionMessage(e))
        cat(msg)
        stop(msg)
    })

    dhis2_routine <- dhis2_routine %>% dplyr::mutate(dplyr::across(c(PERIOD, YEAR, MONTH), as.numeric))
    dhis2_routine <- dhis2_routine %>% dplyr::select(dplyr::any_of(fixed_cols_rr)) %>% dplyr::distinct()

    log_msg(glue::glue(
        "DHIS2 routine file {ROUTINE_FILE} loaded from dataset : {rountine_dataset_name} dataframe dimensions: {paste(dim(dhis2_routine), collapse=', ')}"
    ))
    dhis2_routine
}


load_reporting_data_dataset <- function(config_json, COUNTRY_CODE) {
    dataset_name <- config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED
    file_name <- paste0(COUNTRY_CODE, "_reporting.parquet")

    dhis2_reporting <- tryCatch({
        get_latest_dataset_file_in_memory(dataset_name, file_name)
    }, error = function(e) {
        msg <- paste("[ERROR] Error while loading DHIS2 dataset reporting rates file for: ", COUNTRY_CODE, conditionMessage(e))
        cat(msg)
        stop(msg)
    })
    dhis2_reporting <- dhis2_reporting %>% dplyr::mutate(dplyr::across(c(PERIOD, YEAR, MONTH, VALUE), as.numeric))

    log_msg(paste0(
        "DHIS2 Datatset reporting data loaded from file `", file_name, "` (from dataset : `", dataset_name, "`). Dataframe dimensions: ",
        paste(dim(dhis2_reporting), collapse = ", ")
    ))
    dhis2_reporting
}


compute_reporting_rate_dataset <- function(dhis2_reporting, REPORTING_RATE_PRODUCT_ID, COUNTRY_CODE) {
    if (all(REPORTING_RATE_PRODUCT_ID %in% unique(dhis2_reporting$PRODUCT_UID))) {
        dhis2_reporting <- dhis2_reporting %>% dplyr::filter(PRODUCT_UID %in% REPORTING_RATE_PRODUCT_ID)
    } else {
        log_msg(glue::glue(
            "🚨 Warning: REPORTING_RATE_PRODUCT_UID: {paste(REPORTING_RATE_PRODUCT_ID, collapse=', ')} not found in DHIS2 reporting data. Skipping filtering."
        ), level = "warning")
    }

    dhis2_reporting_wide <- dhis2_reporting %>% tidyr::pivot_wider(names_from = PRODUCT_METRIC, values_from = VALUE)

    dupl_ou_period <- dhis2_reporting_wide %>%
        dplyr::group_by(OU_ID, PERIOD) %>%
        dplyr::filter(dplyr::n() > 1) %>%
        dplyr::ungroup() %>%
        dplyr::select(OU_ID, OU_NAME, PERIOD, PRODUCT_UID, dplyr::ends_with("REPORTS"))

    if (all(dupl_ou_period$ACTUAL_REPORTS %in% c(0, 1)) & all(dupl_ou_period$EXPECTED_REPORTS %in% c(0, 1))) {
        dhis2_reporting_wide <- dhis2_reporting_wide %>%
            dplyr::group_by(PERIOD, OU_ID) %>%
            dplyr::mutate(ACTUAL_REPORTS_deduplicated = ifelse(OU_ID %in% dupl_ou_period$OU_ID, max(ACTUAL_REPORTS), ACTUAL_REPORTS)) %>%
            dplyr::ungroup() %>%
            dplyr::filter(!(OU_ID %in% dupl_ou_period$OU_ID) | (ACTUAL_REPORTS == ACTUAL_REPORTS_deduplicated)) %>%
            dplyr::select(-ACTUAL_REPORTS_deduplicated)
    }

    if (COUNTRY_CODE == "NER") {
        dhis2_reporting_wide <- dhis2_reporting_wide %>%
            dplyr::mutate(
                ACTUAL_REPORTS = ifelse(ACTUAL_REPORTS > 1, 1, ACTUAL_REPORTS),
                EXPECTED_REPORTS = ifelse(EXPECTED_REPORTS > 1, 1, EXPECTED_REPORTS)
            )
    }

    dhis2_reporting_wide_adm2 <- dhis2_reporting_wide %>%
        dplyr::group_by(PERIOD, YEAR, MONTH, ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID) %>%
        dplyr::summarise(
            ACTUAL_REPORTS = sum(ACTUAL_REPORTS, na.rm = TRUE),
            EXPECTED_REPORTS = sum(EXPECTED_REPORTS, na.rm = TRUE),
            .groups = "drop"
        )

    dhis2_reporting_wide_adm2 %>%
        dplyr::mutate(REPORTING_RATE = ACTUAL_REPORTS / EXPECTED_REPORTS)
}


export_reporting_rate_dataset <- function(reporting_rate_dataset, DATA_PATH, COUNTRY_CODE) {
    output_data_path <- file.path(DATA_PATH, "reporting_rate")
    if (!dir.exists(output_data_path)) {
        dir.create(output_data_path, recursive = TRUE)
    }

    file_path <- file.path(output_data_path, paste0(COUNTRY_CODE, "_reporting_rate_dataset.parquet"))
    arrow::write_parquet(reporting_rate_dataset, file_path)
    log_msg(glue::glue("Exported : {file_path}"))

    file_path <- file.path(output_data_path, paste0(COUNTRY_CODE, "_reporting_rate_dataset.csv"))
    write.csv(reporting_rate_dataset, file_path, row.names = FALSE)
    log_msg(glue::glue("Exported : {file_path}"))
}

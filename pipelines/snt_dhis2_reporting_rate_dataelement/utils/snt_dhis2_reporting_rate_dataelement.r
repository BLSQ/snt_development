select_routine_dataset_name_dataelement <- function(ROUTINE_FILE, COUNTRY_CODE, config_json) {
    if (ROUTINE_FILE == glue::glue("{COUNTRY_CODE}_routine.parquet")) {
        return(config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED)
    }
    config_json$SNT_DATASET_IDENTIFIERS$DHIS2_OUTLIERS_IMPUTATION
}


load_routine_data_dataelement <- function(rountine_dataset_name, ROUTINE_FILE, COUNTRY_CODE) {
    dhis2_routine <- tryCatch({
        get_latest_dataset_file_in_memory(rountine_dataset_name, ROUTINE_FILE)
    }, error = function(e) {
        msg <- paste("[ERROR] Error while loading DHIS2 routine data file for: ", COUNTRY_CODE, conditionMessage(e))
        cat(msg)
        stop(msg)
    })

    dhis2_routine <- dhis2_routine %>%
        dplyr::mutate(dplyr::across(c(PERIOD, YEAR, MONTH), as.numeric))

    log_msg(glue::glue(
        "DHIS2 routine file {ROUTINE_FILE} loaded from dataset: {rountine_dataset_name}. Dataframe dimensions: {paste(dim(dhis2_routine), collapse=', ')}"
    ))

    dhis2_routine
}


load_pyramid_data_dataelement <- function(config_json, COUNTRY_CODE) {
    dataset_name <- config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED

    dhis2_pyramid_formatted <- tryCatch({
        get_latest_dataset_file_in_memory(dataset_name, paste0(COUNTRY_CODE, "_pyramid.parquet"))
    }, error = function(e) {
        msg <- paste("Error while loading DHIS2 pyramid FORMATTED data file for: ", COUNTRY_CODE, conditionMessage(e))
        cat(msg)
        stop(msg)
    })

    log_msg(paste0(
        "DHIS2 pyramid FORMATTED data loaded from dataset: `", dataset_name,
        "`. Dataframe dimensions: ", paste(dim(dhis2_pyramid_formatted), collapse = ", ")
    ))

    dhis2_pyramid_formatted
}


build_facility_master_dataelement <- function(
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


compute_reporting_rate_dataelement <- function(
    facility_master,
    dhis2_routine,
    DHIS2_INDICATORS,
    ACTIVITY_INDICATORS,
    VOLUME_ACTIVITY_INDICATORS,
    DATAELEMENT_METHOD_DENOMINATOR,
    USE_WEIGHTED_REPORTING_RATES
) {
    facility_master_routine <- dplyr::left_join(
        facility_master,
        dhis2_routine %>% dplyr::select(OU_ID, PERIOD, dplyr::any_of(DHIS2_INDICATORS)),
        by = c("OU_ID", "PERIOD")
    ) %>%
        dplyr::mutate(
            YEAR = as.numeric(substr(PERIOD, 1, 4)),
            ACTIVE_THIS_PERIOD = ifelse(
                rowSums(!is.na(dplyr::across(dplyr::all_of(ACTIVITY_INDICATORS))) &
                    dplyr::across(dplyr::all_of(ACTIVITY_INDICATORS)) >= 0) > 0, 1, 0
            ),
            COUNT = 1
        ) %>%
        dplyr::mutate(
            period_date = as.Date(zoo::as.yearmon(as.character(PERIOD), "%Y%m")),
            NAME_CLOSED = stringr::str_detect(toupper(OU_NAME), "CLOTUR|FERM(E|EE)?"),
            OPEN_BY_DATE = !(is.na(OPENING_DATE) | as.Date(OPENING_DATE) > period_date |
                (!is.na(CLOSED_DATE) & as.Date(CLOSED_DATE) <= period_date)),
            OPEN = ifelse(!NAME_CLOSED & OPEN_BY_DATE, 1, 0)
        ) %>%
        dplyr::group_by(OU_ID, YEAR) %>%
        dplyr::mutate(ACTIVE_THIS_YEAR = max(ACTIVE_THIS_PERIOD, na.rm = TRUE)) %>%
        dplyr::ungroup()

    mean_monthly_cases <- dhis2_routine %>%
        dplyr::mutate(total_cases_by_hf_month = rowSums(dplyr::across(dplyr::all_of(VOLUME_ACTIVITY_INDICATORS)), na.rm = TRUE)) %>%
        dplyr::group_by(ADM2_ID, OU_ID) %>%
        dplyr::summarise(
            total_cases_by_hf_year = sum(total_cases_by_hf_month, na.rm = TRUE),
            number_of_reporting_months = length(which(total_cases_by_hf_month > 0)),
            .groups = "drop"
        ) %>%
        dplyr::mutate(MEAN_REPORTED_CASES_BY_HF = total_cases_by_hf_year / number_of_reporting_months) %>%
        dplyr::select(ADM2_ID, OU_ID, MEAN_REPORTED_CASES_BY_HF)

    mean_monthly_cases_adm2 <- mean_monthly_cases %>%
        dplyr::select(ADM2_ID, MEAN_REPORTED_CASES_BY_HF) %>%
        dplyr::group_by(ADM2_ID) %>%
        dplyr::summarise(
            SUMMED_MEAN_REPORTED_CASES_BY_ADM2 = sum(MEAN_REPORTED_CASES_BY_HF, na.rm = TRUE),
            NR_OF_HF = dplyr::n()
        )

    hf_weights <- mean_monthly_cases %>%
        dplyr::left_join(mean_monthly_cases_adm2, by = "ADM2_ID") %>%
        dplyr::mutate(WEIGHT = MEAN_REPORTED_CASES_BY_HF / SUMMED_MEAN_REPORTED_CASES_BY_ADM2 * NR_OF_HF)

    facility_master_routine_02 <- facility_master_routine %>%
        dplyr::left_join(hf_weights %>% dplyr::select(OU_ID, WEIGHT), by = c("OU_ID"))

    facility_master_routine_02$ACTIVE_THIS_PERIOD_W <- facility_master_routine_02$ACTIVE_THIS_PERIOD * facility_master_routine_02$WEIGHT
    facility_master_routine_02$COUNT_W <- facility_master_routine_02$COUNT * facility_master_routine_02$WEIGHT
    facility_master_routine_02$OPEN_W <- facility_master_routine_02$OPEN * facility_master_routine_02$WEIGHT
    facility_master_routine_02$ACTIVE_THIS_YEAR_W <- facility_master_routine_02$ACTIVE_THIS_YEAR * facility_master_routine_02$WEIGHT

    reporting_rate_adm2 <- facility_master_routine_02 %>%
        dplyr::group_by(ADM1_ID, ADM1_NAME, ADM2_ID, ADM2_NAME, YEAR, PERIOD) %>%
        dplyr::summarise(
            HF_ACTIVE_THIS_PERIOD_BY_ADM2 = sum(ACTIVE_THIS_PERIOD, na.rm = TRUE),
            NR_OF_HF_BY_ADM2 = sum(COUNT, na.rm = TRUE),
            NR_OF_OPEN_HF_BY_ADM2 = sum(OPEN, na.rm = TRUE),
            HF_ACTIVE_THIS_YEAR_BY_ADM2 = sum(ACTIVE_THIS_YEAR, na.rm = TRUE),
            HF_ACTIVE_THIS_PERIOD_BY_ADM2_WEIGHTED = sum(ACTIVE_THIS_PERIOD_W, na.rm = TRUE),
            NR_OF_HF_BY_ADM2_WEIGHTED = sum(COUNT_W, na.rm = TRUE),
            NR_OF_OPEN_HF_BY_ADM2_WEIGHTED = sum(OPEN_W, na.rm = TRUE),
            HF_ACTIVE_THIS_YEAR_BY_ADM2_WEIGHTED = sum(ACTIVE_THIS_YEAR_W, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        dplyr::mutate(
            RR_TOTAL_HF = HF_ACTIVE_THIS_PERIOD_BY_ADM2 / NR_OF_HF_BY_ADM2,
            RR_OPEN_HF = HF_ACTIVE_THIS_PERIOD_BY_ADM2 / NR_OF_OPEN_HF_BY_ADM2,
            RR_ACTIVE_HF = HF_ACTIVE_THIS_PERIOD_BY_ADM2 / HF_ACTIVE_THIS_YEAR_BY_ADM2,
            RR_TOTAL_HF_W = HF_ACTIVE_THIS_PERIOD_BY_ADM2_WEIGHTED / NR_OF_HF_BY_ADM2_WEIGHTED,
            RR_OPEN_HF_W = HF_ACTIVE_THIS_PERIOD_BY_ADM2_WEIGHTED / NR_OF_OPEN_HF_BY_ADM2_WEIGHTED,
            RR_ACTIVE_HF_W = HF_ACTIVE_THIS_PERIOD_BY_ADM2_WEIGHTED / HF_ACTIVE_THIS_YEAR_BY_ADM2_WEIGHTED
        )

    rr_column_selection <- if (DATAELEMENT_METHOD_DENOMINATOR == "ROUTINE_ACTIVE_FACILITIES") "RR_ACTIVE_HF" else "RR_OPEN_HF"
    if (USE_WEIGHTED_REPORTING_RATES) {
        rr_column_selection <- if (DATAELEMENT_METHOD_DENOMINATOR == "ROUTINE_ACTIVE_FACILITIES") "RR_ACTIVE_HF_W" else "RR_OPEN_HF_W"
    }

    reporting_rate_adm2 %>%
        dplyr::mutate(MONTH = PERIOD %% 100) %>%
        dplyr::rename(REPORTING_RATE = !!rlang::sym(rr_column_selection)) %>%
        dplyr::select(YEAR, MONTH, ADM2_ID, REPORTING_RATE)
}


export_reporting_rate_dataelement <- function(reporting_rate_dataelement, DATA_PATH, COUNTRY_CODE) {
    output_data_path <- file.path(DATA_PATH, "reporting_rate")
    if (!dir.exists(output_data_path)) {
        dir.create(output_data_path, recursive = TRUE)
    }

    file_path <- file.path(output_data_path, paste0(COUNTRY_CODE, "_reporting_rate_dataelement.parquet"))
    arrow::write_parquet(reporting_rate_dataelement, file_path)
    log_msg(glue::glue("Exported : {file_path}"))

    file_path <- file.path(output_data_path, paste0(COUNTRY_CODE, "_reporting_rate_dataelement.csv"))
    write.csv(reporting_rate_dataelement, file_path, row.names = FALSE)
    log_msg(glue::glue("Exported : {file_path}"))
}

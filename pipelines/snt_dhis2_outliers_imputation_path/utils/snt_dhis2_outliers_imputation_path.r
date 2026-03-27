# Helpers for PATH outliers imputation notebook.

build_path_routine_long <- function(dhis2_routine, DHIS2_INDICATORS) {
    dhis2_routine %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM1_NAME", "ADM2_ID", "ADM2_NAME", "OU_ID", "OU_NAME", "PERIOD", DHIS2_INDICATORS))) %>%
        tidyr::pivot_longer(cols = dplyr::all_of(DHIS2_INDICATORS), names_to = "INDICATOR", values_to = "VALUE") %>%
        tidyr::complete(tidyr::nesting(ADM1_ID, ADM1_NAME, ADM2_ID, ADM2_NAME, OU_ID, OU_NAME), PERIOD, INDICATOR) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "INDICATOR", "VALUE")))
}

remove_path_duplicates <- function(dhis2_routine_long) {
    duplicated <- dhis2_routine_long %>%
        dplyr::group_by(ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR) %>%
        dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
        dplyr::filter(n > 1L)

    if (nrow(duplicated) > 0) {
        log_msg(glue::glue("Removing {nrow(duplicated)} duplicated values."))
        dhis2_routine_long <- dhis2_routine_long %>%
            dplyr::distinct(ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR, .keep_all = TRUE)
    }

    list(data = dhis2_routine_long, duplicated = duplicated)
}

detect_possible_stockout <- function(dhis2_routine_outliers, MEAN_DEVIATION) {
    low_testing_periods <- dhis2_routine_outliers %>%
        dplyr::filter(INDICATOR == "TEST") %>%
        dplyr::mutate(
            low_testing = dplyr::case_when(VALUE < MEAN_80 ~ TRUE, TRUE ~ FALSE),
            upper_limit_tested = MEAN_80 + MEAN_DEVIATION * SD_80
        ) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "low_testing", "upper_limit_tested")))

    dhis2_routine_outliers %>%
        dplyr::filter(OUTLIER_TREND == TRUE) %>%
        dplyr::left_join(low_testing_periods, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(POSSIBLE_STKOUT = dplyr::case_when(low_testing == TRUE & INDICATOR == "PRES" & VALUE < upper_limit_tested ~ TRUE, TRUE ~ FALSE)) %>%
        dplyr::filter(POSSIBLE_STKOUT == TRUE) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "POSSIBLE_STKOUT")))
}

detect_possible_epidemic <- function(dhis2_routine_outliers, MEAN_DEVIATION) {
    dhis2_routine_outliers %>%
        dplyr::filter(INDICATOR == "TEST" | INDICATOR == "CONF") %>%
        dplyr::rename(total = VALUE) %>%
        dplyr::mutate(max_value = MEAN_80 + MEAN_DEVIATION * SD_80) %>%
        dplyr::select(-c("MEAN_80", "SD_80")) %>%
        tidyr::pivot_wider(names_from = INDICATOR, values_from = c(total, max_value, OUTLIER_TREND)) %>%
        tidyr::unnest(cols = dplyr::everything()) %>%
        dplyr::mutate(POSSIBLE_EPID = dplyr::case_when(
            OUTLIER_TREND_CONF == TRUE & (OUTLIER_TREND_TEST == TRUE | total_TEST >= total_CONF) ~ TRUE,
            TRUE ~ FALSE
        )) %>%
        dplyr::filter(POSSIBLE_EPID == TRUE) %>%
        dplyr::select(dplyr::all_of(c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "POSSIBLE_EPID")))
}

build_path_clean_outliers <- function(dhis2_routine_outliers, possible_stockout, possible_epidemic) {
    dhis2_routine_outliers %>%
        dplyr::left_join(possible_stockout, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(OUTLIER_TREND_01 = dplyr::case_when(OUTLIER_TREND == TRUE & INDICATOR == "PRES" & POSSIBLE_STKOUT == TRUE ~ FALSE, TRUE ~ OUTLIER_TREND)) %>%
        dplyr::left_join(possible_epidemic, by = c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD")) %>%
        dplyr::mutate(OUTLIER_TREND_02 = dplyr::case_when(OUTLIER_TREND_01 == TRUE & INDICATOR %in% c("CONF", "TEST") & POSSIBLE_EPID == TRUE ~ TRUE, TRUE ~ OUTLIER_TREND_01)) %>%
        dplyr::select(-OUTLIER_TREND) %>%
        dplyr::rename(OUTLIER_TREND = OUTLIER_TREND_02) %>%
        dplyr::mutate(
            YEAR = as.integer(substr(PERIOD, 1, 4)),
            MONTH = as.integer(substr(PERIOD, 5, 6))
        ) %>%
        dplyr::select(dplyr::all_of(c(
            "PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID",
            "INDICATOR", "VALUE", "MEAN_80", "SD_80",
            "OUTLIER_TREND", "POSSIBLE_STKOUT", "POSSIBLE_EPID"
        )))
}

impute_path_outliers <- function(routine_data_outliers_clean) {
    routine_data_outliers_clean %>%
        dplyr::rename(VALUE_OLD = VALUE) %>%
        dplyr::mutate(VALUE_IMPUTED = ifelse(OUTLIER_TREND == TRUE, MEAN_80, VALUE_OLD)) %>%
        dplyr::select(dplyr::all_of(c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR", "VALUE_OLD", "VALUE_IMPUTED", "OUTLIER_TREND"))) %>%
        tidyr::pivot_wider(names_from = INDICATOR, values_from = c(VALUE_OLD, VALUE_IMPUTED, OUTLIER_TREND)) %>%
        dplyr::mutate(reverse_val = dplyr::case_when(
            !is.na(VALUE_IMPUTED_TEST) & !is.na(VALUE_IMPUTED_CONF) &
                VALUE_IMPUTED_TEST < VALUE_IMPUTED_CONF &
                VALUE_OLD_TEST > VALUE_OLD_CONF ~ TRUE,
            TRUE ~ FALSE
        )) %>%
        dplyr::mutate(
            VALUE_IMPUTED_TEST = ifelse(reverse_val == TRUE, VALUE_OLD_TEST, VALUE_IMPUTED_TEST),
            OUTLIER_TREND_TEST = ifelse(reverse_val == TRUE, FALSE, OUTLIER_TREND_TEST)
        ) %>%
        dplyr::mutate(
            VALUE_IMPUTED_CONF = ifelse(reverse_val == TRUE, VALUE_OLD_CONF, VALUE_IMPUTED_CONF),
            OUTLIER_TREND_CONF = ifelse(reverse_val == TRUE, FALSE, OUTLIER_TREND_CONF)
        ) %>%
        dplyr::select(-reverse_val) %>%
        tidyr::pivot_longer(
            cols = dplyr::starts_with("VALUE_OLD_") | dplyr::starts_with("VALUE_IMPUTED_") | dplyr::starts_with("OUTLIER_TREND_"),
            names_to = c(".value", "INDICATOR"),
            names_pattern = "(.*)_(.*)$"
        ) %>%
        dplyr::arrange(ADM1_ID, ADM2_ID, OU_ID, PERIOD, INDICATOR) %>%
        dplyr::select(dplyr::all_of(c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR", "VALUE_OLD", "VALUE_IMPUTED", "OUTLIER_TREND")))
}

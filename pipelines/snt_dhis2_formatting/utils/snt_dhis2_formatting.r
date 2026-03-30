# Shared helpers for snt_dhis2_formatting code notebooks.

build_routine_indicators <- function(routine_data_ind, dhis_indicator_definitions_clean) {
    empty_data_indicators <- c()

    for (indicator in names(dhis_indicator_definitions_clean)) {
        data_element_uids <- dhis_indicator_definitions_clean[[indicator]]
        col_names <- c()

        if (length(data_element_uids) > 0) {
            for (dx in data_element_uids) {
                dx_co <- gsub("\\.", "_", dx)
                if (grepl("_", dx_co)) {
                    col_names <- c(col_names, dx_co)
                } else {
                    if (!any(grepl(dx, colnames(routine_data_ind)))) {
                        msg <- paste0("Data element : ", dx, " of indicator ", indicator, " is missing in the DHIS2 routine data.")
                        log_msg(msg, level = "warning")
                    } else {
                        col_names <- c(col_names, colnames(routine_data_ind)[grepl(dx, colnames(routine_data_ind))])
                    }
                }
            }

            if (length(col_names) == 0) {
                msg <- paste0("No data elements available to build indicator : ", indicator, ", skipped.")
                log_msg(msg, level = "warning")
                empty_data_indicators <- c(empty_data_indicators, indicator)
                next
            }

            msg <- paste0("Building indicator : ", indicator, " -> column selection : ", paste(col_names, collapse = ", "))
            log_msg(msg)

            if (length(col_names) > 1) {
                sums <- rowSums(routine_data_ind[, col_names], na.rm = TRUE)
                all_na <- rowSums(!is.na(routine_data_ind[, col_names])) == 0
                sums[all_na] <- NA
                routine_data_ind[[indicator]] <- sums
            } else {
                routine_data_ind[indicator] <- routine_data_ind[, col_names]
            }
        } else {
            routine_data_ind[indicator] <- NA
            msg <- paste0("Building indicator : ", indicator, " -> column selection : NULL")
            log_msg(msg)
        }
    }

    list(
        routine_data_ind = routine_data_ind,
        empty_data_indicators = empty_data_indicators
    )
}

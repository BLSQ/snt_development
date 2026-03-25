detect_outliers_mad_custom <- function(dt, deviation) {
    flag_col <- paste0("OUTLIER_MAD", deviation)
    dt <- data.table::copy(dt)
    dt[, median_val := median(VALUE, na.rm = TRUE), by = .(YEAR, OU_ID, INDICATOR)]
    dt[, mad_val := mad(VALUE, constant = 1, na.rm = TRUE), by = .(YEAR, OU_ID, INDICATOR)]
    dt[, (flag_col) := (VALUE > (median_val + deviation * mad_val)) | (VALUE < (median_val - deviation * mad_val))]
    dt[is.na(get(flag_col)), (flag_col) := FALSE]
    dt[, c("median_val", "mad_val") := NULL]
    dt
}

detect_seasonal_outliers <- function(dt, deviation, workers = 1) {
    outlier_col <- paste0("OUTLIER_SEASONAL", deviation)
    dt <- data.table::copy(dt)
    data.table::setorder(dt, OU_ID, INDICATOR, PERIOD)

    process_group <- function(sub_dt) {
        n_valid <- sum(!is.na(sub_dt$VALUE))
        if (n_valid < 2) {
            return(data.table::data.table(
                PERIOD = sub_dt$PERIOD,
                OU_ID = sub_dt$OU_ID,
                INDICATOR = sub_dt$INDICATOR,
                OUTLIER_FLAG = rep(FALSE, nrow(sub_dt))
            ))
        }

        values <- as.numeric(sub_dt$VALUE)
        ts_data <- stats::ts(values, frequency = 12)
        cleaned_ts <- tryCatch(
            forecast::tsclean(ts_data, replace.missing = TRUE),
            error = function(e) ts_data
        )
        mad_val <- mad(values, constant = 1, na.rm = TRUE)

        if (is.na(mad_val) || mad_val == 0) {
            return(data.table::data.table(
                PERIOD = sub_dt$PERIOD,
                OU_ID = sub_dt$OU_ID,
                INDICATOR = sub_dt$INDICATOR,
                OUTLIER_FLAG = rep(FALSE, nrow(sub_dt))
            ))
        }

        is_outlier <- abs(as.numeric(ts_data) - as.numeric(cleaned_ts)) / mad_val >= deviation
        is_outlier[is.na(is_outlier)] <- FALSE

        data.table::data.table(
            PERIOD = sub_dt$PERIOD,
            OU_ID = sub_dt$OU_ID,
            INDICATOR = sub_dt$INDICATOR,
            OUTLIER_FLAG = as.logical(is_outlier)
        )
    }

    group_keys <- unique(dt[, .(OU_ID, INDICATOR)])
    group_list <- lapply(seq_len(nrow(group_keys)), function(i) {
        dt[OU_ID == group_keys$OU_ID[i] & INDICATOR == group_keys$INDICATOR[i]]
    })

    if (workers > 1 && requireNamespace("future.apply", quietly = TRUE)) {
        result_list <- future.apply::future_lapply(group_list, process_group, future.seed = TRUE)
    } else {
        result_list <- lapply(group_list, process_group)
    }

    outlier_flags <- data.table::rbindlist(result_list, use.names = TRUE)
    data.table::setnames(outlier_flags, "OUTLIER_FLAG", outlier_col)

    result_dt <- merge(dt, outlier_flags, by = c("PERIOD", "OU_ID", "INDICATOR"), all.x = TRUE)
    result_dt[is.na(get(outlier_col)), (outlier_col) := FALSE]
    result_dt
}

to_routine_wide <- function(dt_long, fixed_cols, indicators_to_keep, pyramid_names) {
    routine_wide <- data.table::dcast(
        dt_long[, .(PERIOD, YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, VALUE)],
        PERIOD + YEAR + MONTH + ADM1_ID + ADM2_ID + OU_ID ~ INDICATOR,
        value.var = "VALUE"
    )

    routine_wide <- merge(routine_wide, unique(pyramid_names), by = c("ADM1_ID", "ADM2_ID", "OU_ID"), all.x = TRUE)

    target_cols <- c("PERIOD", "YEAR", "MONTH", "ADM1_NAME", "ADM1_ID", "ADM2_NAME", "ADM2_ID", "OU_ID", "OU_NAME", indicators_to_keep)
    for (col in setdiff(target_cols, names(routine_wide))) {
        if (col %in% indicators_to_keep) {
            routine_wide[, (col) := NA_real_]
        } else if (col %in% c("YEAR", "MONTH")) {
            routine_wide[, (col) := NA_integer_]
        } else {
            routine_wide[, (col) := NA_character_]
        }
    }
    cols_to_keep <- intersect(target_cols, names(routine_wide))
    routine_wide <- routine_wide[, ..cols_to_keep]
    routine_wide
}

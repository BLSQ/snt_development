# Main helpers for magic glasses outliers imputation pipeline.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))


#' Load DHIS2 Routine Data with Validation
#'
#' Reads the latest routine parquet file from the OpenHEXA dataset, logs its
#' dimensions, optionally casts YEAR and MONTH to integer, and stops with an
#' [ERROR] message if the file cannot be loaded or required indicators are
#' missing.
#'
#' @param dataset_name Character. OpenHEXA dataset identifier.
#' @param country_code Character. Country code used as the routine filename prefix.
#' @param required_indicators Character vector or NULL. Indicator columns that must
#'   be present. Default: NULL (no check).
#' @param cast_year_month Logical. Cast YEAR and MONTH to integer. Default: TRUE.
#' @return Data frame. The validated routine data.
#'
#' @export
load_routine_data <- function(dataset_name, country_code, required_indicators = NULL, cast_year_month = TRUE) {
    dhis2_routine <- tryCatch(
        {
            get_latest_dataset_file_in_memory(dataset_name, paste0(country_code, "_routine.parquet"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading DHIS2 routine data file for {country_code} : {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    log_msg(glue::glue("DHIS2 routine data loaded from dataset : {dataset_name}"))
    log_msg(glue::glue("DHIS2 routine data loaded has dimensions: {nrow(dhis2_routine)} rows, {ncol(dhis2_routine)} columns."))

    if (cast_year_month && all(c("YEAR", "MONTH") %in% colnames(dhis2_routine))) {
        dhis2_routine[c("YEAR", "MONTH")] <- lapply(dhis2_routine[c("YEAR", "MONTH")], as.integer)
    }

    if (!is.null(required_indicators)) {
        missing_indicators <- setdiff(required_indicators, colnames(dhis2_routine))
        if (length(missing_indicators) > 0) {
            msg <- paste("[ERROR] Missing indicator column(s) in routine data:", paste(missing_indicators, collapse = ", "))
            log_msg(msg)
            stop(msg)
        }
    }

    dhis2_routine
}


#' Detect Outliers Using MAD Thresholds
#'
#' Computes the median and MAD (constant = 1) of VALUE by YEAR, OU_ID and
#' INDICATOR, and flags values outside median +/- deviation * MAD. Values that
#' cannot be evaluated (missing VALUE or MAD) are not flagged.
#'
#' @param dt data.table. Long-format routine data with YEAR, OU_ID, INDICATOR and VALUE.
#' @param deviation Numeric. MAD multiplier; also used to name the flag column.
#' @return data.table. A copy of `dt` with a logical OUTLIER_MAD{deviation} column.
#'
#' @export
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


#' Check Whether Any OU_ID x INDICATOR Series Has Gaps in PERIOD
#'
#' Compares each series' periods against the full set of periods present
#' anywhere in the dataset, within that series' own min-max range.
#'
#' @param dt Data frame or data.table. Must contain OU_ID, INDICATOR, PERIOD.
#' @return Logical. TRUE if at least one OU_ID x INDICATOR series has a gap,
#'   FALSE otherwise.
#'
#' @export
has_period_gaps <- function(dt) {
    all_periods <- sort(unique(dt$PERIOD))

    gap_check <- dt[, .(
        n_rows = .N,
        n_expected = sum(all_periods >= min(PERIOD) & all_periods <= max(PERIOD))
    ), by = .(OU_ID, INDICATOR)]

    any(gap_check$n_rows != gap_check$n_expected)
}


#' Detect Seasonal Outliers from Cleaned Time Series Residuals
#'
#' For each OU_ID x INDICATOR series, cleans the series with
#' `forecast::tsclean()` and flags values whose absolute residual, scaled by the
#' series MAD, is at least `deviation`. Series with fewer than 2 values, or with
#' a MAD of 0, are not flagged. Runs in parallel with `future.apply` when
#' `workers` > 1 and the package is available.
#'
#' @param dt data.table. Long-format routine data with PERIOD, OU_ID, INDICATOR and VALUE.
#' @param deviation Numeric. Threshold on scaled residuals; also used to name the flag column.
#' @param frequency Integer. Number of periods per seasonal cycle. Default: 12.
#' @param workers Integer. Number of parallel workers. Default: 1 (sequential).
#' @return data.table. `dt` with a logical OUTLIER_SEASONAL{deviation} column.
#'
#' @export
detect_seasonal_outliers <- function(dt, deviation, frequency = 12, workers = 1) {
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
        ts_data <- stats::ts(values, frequency = frequency)
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


#' Convert Long Routine Data to Wide Export Format
#'
#' Casts indicators to columns, joins ADM1/ADM2/OU names, adds any expected
#' column that is missing as NA of the appropriate type, and orders columns as
#' in the routine export.
#'
#' @param dt_long data.table. Long-format routine data with INDICATOR and VALUE.
#' @param indicators_to_keep Character vector. Indicator columns expected in the output.
#' @param pyramid_names Data frame or data.table. Mapping of ADM1/ADM2/OU IDs to names.
#' @return data.table. Wide routine table: PERIOD, YEAR, MONTH, ADM and OU columns,
#'   then one column per indicator.
#'
#' @export
to_routine_wide <- function(dt_long, indicators_to_keep, pyramid_names) {
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


#' Combine Magic Glasses Outlier Flags into a Single Column
#'
#' Joins the partial flags (MAD15 -> MAD10) onto the long routine data and, in
#' complete mode, the seasonal flags (seasonal5 -> seasonal3). Complete mode is
#' used when `flagged_outliers_seasonal5_seasonal3` is provided. Values already
#' flagged by the partial step are absent from the seasonal input, so they stay
#' flagged in complete mode.
#'
#' @param dhis2_routine_long data.table. Long-format routine data (fixed columns,
#'   INDICATOR, VALUE).
#' @param flagged_outliers_mad15_mad10 data.table. Partial detection output with
#'   OUTLIER_MAD15_MAD10.
#' @param flagged_outliers_seasonal5_seasonal3 data.table or NULL. Complete detection
#'   output with OUTLIER_SEASONAL5_SEASONAL3. Default: NULL (partial mode).
#' @return data.table. `dhis2_routine_long` columns plus OUTLIER_DETECTED (logical,
#'   never NA) and OUTLIER_METHOD ("MAGIC_GLASSES_PARTIAL" or "MAGIC_GLASSES_COMPLETE").
#'
#' @export
build_magic_glasses_flags <- function(
    dhis2_routine_long,
    flagged_outliers_mad15_mad10,
    flagged_outliers_seasonal5_seasonal3 = NULL
) {
    join_cols <- c("PERIOD", "OU_ID", "INDICATOR")
    partial_subset <- flagged_outliers_mad15_mad10[, .(PERIOD, OU_ID, INDICATOR, OUTLIER_DETECTED = OUTLIER_MAD15_MAD10)]
    flags <- merge(dhis2_routine_long, partial_subset, by = join_cols, all.x = TRUE)
    method <- "MAGIC_GLASSES_PARTIAL"

    if (!is.null(flagged_outliers_seasonal5_seasonal3)) {
        method <- "MAGIC_GLASSES_COMPLETE"
        complete_subset <- flagged_outliers_seasonal5_seasonal3[, .(PERIOD, OU_ID, INDICATOR, OUTLIER_COMPLETE = OUTLIER_SEASONAL5_SEASONAL3)]
        flags <- merge(flags, complete_subset, by = join_cols, all.x = TRUE)
        flags[is.na(OUTLIER_COMPLETE) & OUTLIER_DETECTED == TRUE, OUTLIER_COMPLETE := TRUE]
        flags[, OUTLIER_DETECTED := OUTLIER_COMPLETE]
        flags[, OUTLIER_COMPLETE := NULL]
    }

    flags[is.na(OUTLIER_DETECTED), OUTLIER_DETECTED := FALSE]
    flags[, OUTLIER_METHOD := method]
    log_msg(glue::glue("{method}: {sum(flags$OUTLIER_DETECTED)} outliers flagged out of {nrow(flags)} values."))
    flags
}


#' Format the Magic Glasses Detection Output Table
#'
#' Selects the long-format detection columns, joins ADM1/ADM2/OU names, and adds
#' a DATE column (first day of the month).
#'
#' @param flags data.table. Output of `build_magic_glasses_flags()`.
#' @param pyramid_names Data frame or data.table. Mapping of ADM1/ADM2/OU IDs to names.
#' @return data.table. One row per facility, period and indicator, ready to be saved
#'   as `{CC}_routine_outliers_detected.parquet`.
#'
#' @export
format_magic_glasses_detected <- function(flags, pyramid_names) {
    detected <- flags[, .(PERIOD, YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, VALUE, OUTLIER_DETECTED, OUTLIER_METHOD)]
    detected <- merge(detected, unique(pyramid_names), by = c("ADM1_ID", "ADM2_ID", "OU_ID"), all.x = TRUE)
    detected[, DATE := as.Date(sprintf("%04d-%02d-01", YEAR, MONTH))]
    detected
}


#' Format the Magic Glasses Imputed Routine Output Table
#'
#' Replaces flagged outliers with a centered 3-period moving mean (see
#' `impute_outliers()`) and reshapes the result to the wide routine format.
#'
#' @param flags data.table. Output of `build_magic_glasses_flags()`.
#' @param indicators_to_keep Character vector. Indicator columns expected in the output.
#' @param pyramid_names Data frame or data.table. Mapping of ADM1/ADM2/OU IDs to names.
#' @return data.table. Wide routine table ready to be saved as
#'   `{CC}_routine_outliers_imputed.parquet`.
#'
#' @export
format_magic_glasses_imputed <- function(flags, indicators_to_keep, pyramid_names) {
    imputed <- data.table::as.data.table(
        impute_outliers(flags, outlier_col = "OUTLIER_DETECTED", n = 3, stat = "mean")
    )
    imputed[, VALUE := VALUE_IMPUTED]
    to_routine_wide(imputed, indicators_to_keep, pyramid_names)
}


#' Format the Magic Glasses Removed Routine Output Table
#'
#' Sets flagged outliers to NA and reshapes the result to the wide routine
#' format.
#'
#' @param flags data.table. Output of `build_magic_glasses_flags()`.
#' @param indicators_to_keep Character vector. Indicator columns expected in the output.
#' @param pyramid_names Data frame or data.table. Mapping of ADM1/ADM2/OU IDs to names.
#' @return data.table. Wide routine table ready to be saved as
#'   `{CC}_routine_outliers_removed.parquet`.
#'
#' @export
format_magic_glasses_removed <- function(flags, indicators_to_keep, pyramid_names) {
    removed <- data.table::copy(flags)
    removed[OUTLIER_DETECTED == TRUE, VALUE := NA_real_]
    to_routine_wide(removed, indicators_to_keep, pyramid_names)
}


#' Impute Flagged Outliers Using a Centered Moving Statistic
#'
#' For each ADM/OU/indicator time series, values marked as outliers are
#' replaced by a centered moving mean or median (ceiling), preserving
#' non-outlier observations.
#'
#' @param dt Data frame or data.table. Routine data in long format.
#' @param outlier_col Character. Name of the logical outlier flag column.
#' @param n Integer. Size of the centered rolling window, in periods. Default: 3.
#' @param stat Character. Either "mean" or "median". Default: "mean".
#' @return Data frame with a VALUE_IMPUTED column added and the TO_IMPUTE /
#'   MOVING_STAT helper columns removed.
#'
#' @export
impute_outliers <- function(dt, outlier_col, n = 3, stat = c("mean", "median")) {
    stat <- match.arg(stat)
    stat_fun <- if (stat == "mean") mean else median

    dt <- data.table::as.data.table(dt)
    data.table::setorder(dt, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, PERIOD, YEAR, MONTH)
    dt[, TO_IMPUTE := data.table::fifelse(get(outlier_col) == TRUE, NA_real_, VALUE)]
    dt[, MOVING_STAT := as.numeric(data.table::frollapply(
        TO_IMPUTE,
        N = n,
        FUN = function(x) {
            m <- stat_fun(x, na.rm = TRUE)
            if (is.nan(m)) NA_real_ else ceiling(m)
        },
        align = "center"
    )), by = .(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)]
    dt[, VALUE_IMPUTED := data.table::fifelse(is.na(TO_IMPUTE), MOVING_STAT, TO_IMPUTE)]
    dt[, c("TO_IMPUTE", "MOVING_STAT") := NULL]
    return(as.data.frame(data.table::copy(dt)))
}

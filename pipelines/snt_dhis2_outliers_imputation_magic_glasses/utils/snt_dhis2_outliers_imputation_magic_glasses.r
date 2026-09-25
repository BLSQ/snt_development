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


#' Build the ADM1/ADM2/OU Name Lookup for Outliers Output Tables
#'
#' Extracts the location names from the routine data, unchanged (no country-specific
#' cleaning), as character columns, keeping one row per ADM1_ID x ADM2_ID x OU_ID so
#' a join on those IDs can never duplicate rows. Stops with an [ERROR] message if a
#' name or ID column is missing.
#'
#' @param routine_df Data frame or data.table. Formatted routine data.
#' @return Data frame with ADM1_ID, ADM1_NAME, ADM2_ID, ADM2_NAME, OU_ID and OU_NAME.
#'
#' @export
get_outliers_pyramid_names <- function(routine_df) {
    name_cols <- c("ADM1_ID", "ADM1_NAME", "ADM2_ID", "ADM2_NAME", "OU_ID", "OU_NAME")
    missing_name_cols <- setdiff(name_cols, colnames(routine_df))
    if (length(missing_name_cols) > 0) {
        msg <- paste("[ERROR] Routine data is missing name column(s):", paste(missing_name_cols, collapse = ", "))
        log_msg(msg, "error")
        stop(msg)
    }

    as.data.frame(routine_df) %>%
        dplyr::select(dplyr::all_of(name_cols)) %>%
        dplyr::mutate(dplyr::across(dplyr::everything(), as.character)) %>%
        dplyr::distinct(ADM1_ID, ADM2_ID, OU_ID, .keep_all = TRUE)
}


#' Cast PERIOD Values to Integer
#'
#' Converts YYYYMM periods (character, factor or numeric) to integer, and stops with
#' an [ERROR] message if any non-missing value cannot be converted, rather than
#' silently producing NA periods.
#'
#' @param period Vector. PERIOD values.
#' @return Integer vector of the same length.
#'
#' @export
cast_period_to_integer <- function(period) {
    period_int <- suppressWarnings(as.integer(as.character(period)))
    n_bad_periods <- sum(is.na(period_int) & !is.na(period))
    if (n_bad_periods > 0) {
        msg <- glue::glue("[ERROR] {n_bad_periods} PERIOD value(s) cannot be converted to integer (expected YYYYMM).")
        log_msg(msg, "error")
        stop(msg)
    }
    period_int
}


#' Standardize Long-Format Outliers Data Before Formatting
#'
#' Shared first step of the outliers output formatters, so every output table types its
#' key columns the same way. Checks that the key columns (and any `extra_cols`) are
#' present, casts PERIOD with `cast_period_to_integer()`, YEAR and MONTH to integer,
#' ADM1_ID, ADM2_ID, OU_ID and INDICATOR to character and VALUE to double, and writes
#' NaN values as NA. Other columns are kept unchanged. Duplicated keys are not checked
#' here: deduplicate upstream, in the notebook.
#'
#' @param outliers_long Data frame or data.table. Long-format data with PERIOD, YEAR,
#'   MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR and VALUE.
#' @param extra_cols Character vector or NULL. Additional columns that must be present
#'   (e.g. the outlier flag column). Default: NULL.
#' @return Data frame. `outliers_long` with standardized key column types.
#'
#' @export
standardize_outliers_long <- function(outliers_long, extra_cols = NULL) {
    key_cols <- c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID", "INDICATOR", "VALUE")
    missing_cols <- setdiff(c(key_cols, extra_cols), colnames(outliers_long))
    if (length(missing_cols) > 0) {
        msg <- paste("[ERROR] Outliers table is missing column(s):", paste(missing_cols, collapse = ", "))
        log_msg(msg, "error")
        stop(msg)
    }

    outliers_long <- as.data.frame(outliers_long)
    outliers_long$PERIOD <- cast_period_to_integer(outliers_long$PERIOD)

    outliers_long %>%
        dplyr::mutate(
            YEAR = as.integer(YEAR),
            MONTH = as.integer(MONTH),
            ADM1_ID = as.character(ADM1_ID),
            ADM2_ID = as.character(ADM2_ID),
            OU_ID = as.character(OU_ID),
            INDICATOR = as.character(INDICATOR),
            VALUE = as.double(VALUE),
            VALUE = dplyr::if_else(is.nan(VALUE), NA_real_, VALUE)
        )
}


#' Format the Routine Outliers Detection Output Table
#'
#' Builds the standard `{CC}_routine_outliers_detected.parquet` table, written to be
#' method-agnostic so it can later move to `code/snt_utils.r` and be shared by all
#' `snt_dhis2_outliers_imputation_*` pipelines. Key columns are typed by
#' `standardize_outliers_long()`. It uses one flag column as OUTLIER_DETECTED (NA
#' counted as not an outlier), stamps the method, adds a DATE column (first day of the
#' month), joins ADM1/ADM2/OU names taken unchanged from the routine data, and fixes
#' column order and row order:
#' PERIOD, YEAR, MONTH (integer), DATE (Date), ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID,
#' OU_ID, OU_NAME, INDICATOR (character), VALUE (double), OUTLIER_DETECTED (logical),
#' OUTLIER_METHOD (character). Location and period columns follow the same order as
#' the imputed and removed routine tables.
#'
#' @param outliers_long Data frame or data.table. Long-format detection results with
#'   PERIOD, YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, VALUE and the flag column.
#' @param outlier_col Character. Name of the logical flag column in `outliers_long`.
#' @param method Character. Value written to OUTLIER_METHOD (e.g. "MAGIC_GLASSES_PARTIAL").
#' @param routine_df Data frame or data.table. Formatted routine data, used as the
#'   source of ADM1_NAME, ADM2_NAME and OU_NAME.
#' @return Data frame. One row per facility, period and indicator, sorted by ADM1_ID,
#'   ADM2_ID, OU_ID, INDICATOR and PERIOD.
#'
#' @export
format_outliers_detected_table <- function(outliers_long, outlier_col, method, routine_df) {
    pyramid_names <- get_outliers_pyramid_names(routine_df)

    detected <- standardize_outliers_long(outliers_long, extra_cols = outlier_col) %>%
        dplyr::mutate(
            OUTLIER_DETECTED = dplyr::coalesce(as.logical(.data[[outlier_col]]), FALSE),
            OUTLIER_METHOD = as.character(method),
            DATE = as.Date(sprintf("%04d-%02d-01", YEAR, MONTH))
        ) %>%
        dplyr::left_join(pyramid_names, by = c("ADM1_ID", "ADM2_ID", "OU_ID")) %>%
        dplyr::select(
            PERIOD, YEAR, MONTH, DATE,
            ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID, OU_ID, OU_NAME,
            INDICATOR, VALUE, OUTLIER_DETECTED, OUTLIER_METHOD
        ) %>%
        dplyr::arrange(ADM1_ID, ADM2_ID, OU_ID, INDICATOR, PERIOD)

    log_msg(glue::glue("{method}: detection table formatted ({sum(detected$OUTLIER_DETECTED)} outliers out of {nrow(detected)} values)."))
    detected
}


#' Pivot Long Outliers Routine Data to the Standard Wide Routine Layout
#'
#' Shared layout step of `format_outliers_imputed_table()` and
#' `format_outliers_removed_table()`, so both tables always have the same structure.
#' Key columns are typed by `standardize_outliers_long()`. Pivots the VALUE column to
#' one column per indicator, joins ADM1/ADM2/OU names unchanged from the routine data,
#' and fixes column order and row order:
#' PERIOD, YEAR, MONTH (integer), ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID, OU_ID,
#' OU_NAME (character), then one double column per indicator in `indicators` order.
#'
#' The rows are taken from `routine_df`, not from `routine_long`: the output has exactly
#' one row per facility x period of the formatted routine data, whatever rows the
#' notebook passes in. A facility x period missing from `routine_long` (e.g. because its
#' outlier rows were dropped instead of set to NA) comes back with NA values; input rows
#' that are not in the routine data are dropped, and their count is logged; if none of
#' the input rows match the routine data, it stops with an [ERROR] (key mismatch). An indicator
#' with no values at all is added as an all-NA column. Facility x period x indicator keys
#' must be unique: deduplicate upstream, in the notebook.
#'
#' @param routine_long Data frame or data.table. Long-format routine data with PERIOD,
#'   YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR and VALUE (the value to publish).
#' @param indicators Character vector. Indicator columns of the output, in order;
#'   indicators of `routine_long` not listed here are dropped.
#' @param routine_df Data frame or data.table. Formatted routine data, used as the
#'   source of the output rows (facility x period) and of ADM1_NAME, ADM2_NAME and OU_NAME.
#' @return Data frame. One row per facility and period of `routine_df`, sorted by
#'   ADM1_ID, ADM2_ID, OU_ID and PERIOD.
#'
#' @export
to_outliers_routine_wide <- function(routine_long, indicators, routine_df) {
    row_cols <- c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID")

    missing_row_cols <- setdiff(row_cols, colnames(routine_df))
    if (length(missing_row_cols) > 0) {
        msg <- paste("[ERROR] Routine data is missing row key column(s):", paste(missing_row_cols, collapse = ", "))
        log_msg(msg, "error")
        stop(msg)
    }

    pyramid_names <- get_outliers_pyramid_names(routine_df)

    # Output rows: every facility x period of the routine data, typed like the long table
    row_keys <- as.data.frame(routine_df) %>%
        dplyr::select(dplyr::all_of(row_cols)) %>%
        dplyr::mutate(
            PERIOD = cast_period_to_integer(PERIOD),
            YEAR = as.integer(YEAR),
            MONTH = as.integer(MONTH),
            ADM1_ID = as.character(ADM1_ID),
            ADM2_ID = as.character(ADM2_ID),
            OU_ID = as.character(OU_ID)
        ) %>%
        dplyr::distinct()

    routine_long <- standardize_outliers_long(routine_long) %>%
        dplyr::select(dplyr::all_of(c(row_cols, "INDICATOR", "VALUE")))

    input_row_keys <- routine_long %>% dplyr::distinct(dplyr::across(dplyr::all_of(row_cols)))
    n_extra_rows <- input_row_keys %>%
        dplyr::anti_join(row_keys, by = row_cols) %>%
        nrow()

    # No input row matching the routine data can only be a key mismatch (e.g. ID types),
    # which would otherwise publish a table with all indicator values NA
    if (nrow(input_row_keys) > 0 && n_extra_rows == nrow(input_row_keys)) {
        msg <- glue::glue("[ERROR] None of the {n_extra_rows} facility x period row(s) of the outliers table match the routine data; check PERIOD, YEAR, MONTH, ADM1_ID, ADM2_ID and OU_ID.")
        log_msg(msg, "error")
        stop(msg)
    }
    if (n_extra_rows > 0) {
        log_msg(glue::glue("{n_extra_rows} facility x period row(s) not present in the routine data were dropped from the wide table."))
    }

    routine_wide <- routine_long %>%
        dplyr::filter(INDICATOR %in% indicators) %>%
        tidyr::pivot_wider(names_from = "INDICATOR", values_from = "VALUE")
    routine_wide <- row_keys %>%
        dplyr::left_join(routine_wide, by = row_cols)

    for (indicator in setdiff(indicators, colnames(routine_wide))) {
        routine_wide[[indicator]] <- NA_real_
    }

    routine_wide %>%
        dplyr::left_join(pyramid_names, by = c("ADM1_ID", "ADM2_ID", "OU_ID")) %>%
        dplyr::select(
            PERIOD, YEAR, MONTH,
            ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID, OU_ID, OU_NAME,
            dplyr::all_of(indicators)
        ) %>%
        dplyr::arrange(ADM1_ID, ADM2_ID, OU_ID, PERIOD)
}


#' Format the Routine Outliers Imputed Output Table
#'
#' Builds the standard `{CC}_routine_outliers_imputed.parquet` table, written to be
#' method-agnostic so it can later move to `code/snt_utils.r` and be shared by all
#' `snt_dhis2_outliers_imputation_*` pipelines. It only formats: the imputation itself
#' is method-specific and must already be in `value_col`. Layout, types and checks are
#' those of `to_outliers_routine_wide()`.
#'
#' @param outliers_long Data frame or data.table. Long-format routine data with PERIOD,
#'   YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR and `value_col`.
#' @param indicators Character vector. Indicator columns of the output, in order.
#' @param routine_df Data frame or data.table. Formatted routine data, used as the
#'   source of ADM1_NAME, ADM2_NAME and OU_NAME.
#' @param value_col Character. Column holding the imputed values. Default: "VALUE_IMPUTED".
#' @return Data frame. Wide routine table ready to be saved as
#'   `{CC}_routine_outliers_imputed.parquet`.
#'
#' @export
format_outliers_imputed_table <- function(outliers_long, indicators, routine_df, value_col = "VALUE_IMPUTED") {
    if (!value_col %in% colnames(outliers_long)) {
        msg <- glue::glue("[ERROR] Outliers imputed table is missing the imputed value column: {value_col}")
        log_msg(msg, "error")
        stop(msg)
    }

    routine_long <- as.data.frame(outliers_long)
    routine_long$VALUE <- routine_long[[value_col]]

    imputed <- to_outliers_routine_wide(routine_long, indicators, routine_df)
    log_msg(glue::glue("Imputed table formatted: {nrow(imputed)} rows, {length(indicators)} indicators."))
    imputed
}


#' Format the Routine Outliers Removed Output Table
#'
#' Builds the standard `{CC}_routine_outliers_removed.parquet` table, written to be
#' method-agnostic so it can later move to `code/snt_utils.r` and be shared by all
#' `snt_dhis2_outliers_imputation_*` pipelines. It only formats: the removal itself
#' (setting outlier values to NA) is done in the notebook and must already be in
#' `value_col`. Layout, types and checks are those of `to_outliers_routine_wide()`.
#'
#' @param outliers_long Data frame or data.table. Long-format routine data with PERIOD,
#'   YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR and `value_col`.
#' @param indicators Character vector. Indicator columns of the output, in order.
#' @param routine_df Data frame or data.table. Formatted routine data, used as the
#'   source of ADM1_NAME, ADM2_NAME and OU_NAME.
#' @param value_col Character. Column holding the values with outliers removed (NA).
#'   Default: "VALUE_REMOVED".
#' @return Data frame. Wide routine table ready to be saved as
#'   `{CC}_routine_outliers_removed.parquet`.
#'
#' @export
format_outliers_removed_table <- function(outliers_long, indicators, routine_df, value_col = "VALUE_REMOVED") {
    if (!value_col %in% colnames(outliers_long)) {
        msg <- glue::glue("[ERROR] Outliers removed table is missing the removed value column: {value_col}")
        log_msg(msg, "error")
        stop(msg)
    }

    routine_long <- as.data.frame(outliers_long)
    routine_long$VALUE <- routine_long[[value_col]]

    removed <- to_outliers_routine_wide(routine_long, indicators, routine_df)
    log_msg(glue::glue("Removed table formatted: {nrow(removed)} rows, {length(indicators)} indicators."))
    removed
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

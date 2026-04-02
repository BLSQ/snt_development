# Main helpers for magic glasses outliers imputation pipeline.
bootstrap_magic_glasses_context <- function(
    root_path = "~/workspace",
    required_packages = c("arrow", "data.table", "jsonlite", "reticulate", "glue"),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")
    output_dir <- file.path(data_path, "dhis2", "outliers_imputation")
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(unique(required_packages))

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }
    # snt_utils::log_msg() relies on a global `openhexa` object.
    # Expose it before any helper function logs messages.
    assign("openhexa", openhexa, envir = .GlobalEnv)

    return(list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        OUTPUT_DIR = output_dir,
        openhexa = openhexa
    ))
}

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

prepare_magic_glasses_input <- function(
    root_path,
    config_file_name = "SNT_config.json",
    run_complete = FALSE,
    seasonal_workers = 1,
    dev_subset = FALSE,
    dev_subset_adm1_n = 2
) {
    required_packages <- c("arrow", "data.table", "jsonlite", "reticulate", "glue")
    if (run_complete) {
        required_packages <- c(required_packages, "forecast")
    }
    if (run_complete && seasonal_workers > 1) {
        required_packages <- c(required_packages, "future", "future.apply")
    }

    setup_ctx <- bootstrap_magic_glasses_context(
        root_path = root_path,
        required_packages = required_packages
    )

    if (run_complete) {
        log_msg("[WARNING] Complete mode: seasonal detection is very computationally intensive and can take several hours to run.", "warning")
    }

    if (run_complete && seasonal_workers > 1) {
        future::plan(future::multisession, workers = seasonal_workers)
        log_msg(glue::glue("Using parallel seasonal detection with {seasonal_workers} workers"))
    }

    config_json <- jsonlite::fromJSON(file.path(setup_ctx$CONFIG_PATH, config_file_name))

    snt_config_mandatory <- c("COUNTRY_CODE", "DHIS2_ADMINISTRATION_1", "DHIS2_ADMINISTRATION_2")
    for (conf in snt_config_mandatory) {
        if (is.null(config_json$SNT_CONFIG[[conf]])) {
            msg <- paste("Missing configuration input:", conf)
            log_msg(msg)
            stop(msg)
        }
    }

    country_code <- config_json$SNT_CONFIG$COUNTRY_CODE
    fixed_cols <- c("PERIOD", "YEAR", "MONTH", "ADM1_ID", "ADM2_ID", "OU_ID")
    indicators_to_keep <- names(config_json$DHIS2_DATA_DEFINITIONS$DHIS2_INDICATOR_DEFINITIONS)

    dataset_name <- config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED
    dhis2_routine <- load_routine_data(
        dataset_name = dataset_name,
        country_code = country_code,
        required_indicators = indicators_to_keep
    )

    cols_to_select <- intersect(c(fixed_cols, indicators_to_keep), names(dhis2_routine))
    dt_routine <- data.table::as.data.table(dhis2_routine)[, ..cols_to_select]

    dhis2_routine_long <- data.table::melt(
        dt_routine,
        id.vars = intersect(fixed_cols, names(dt_routine)),
        measure.vars = intersect(indicators_to_keep, names(dt_routine)),
        variable.name = "INDICATOR",
        value.name = "VALUE",
        variable.factor = FALSE
    )

    dup_keys <- c("ADM1_ID", "ADM2_ID", "OU_ID", "PERIOD", "YEAR", "MONTH", "INDICATOR")
    dup_keys <- intersect(dup_keys, names(dhis2_routine_long))
    if (length(dup_keys) > 0) {
        duplicated <- dhis2_routine_long[, .N, by = dup_keys][N > 1L]
        if (nrow(duplicated) > 0) {
            log_msg(glue::glue("Removing {nrow(duplicated)} duplicated values."))
            data.table::setkeyv(dhis2_routine_long, dup_keys)
            dhis2_routine_long <- unique(dhis2_routine_long)
        }
    }

    if (dev_subset) {
        unique_adm1 <- unique(dhis2_routine_long$ADM1_ID)
        adm1_to_keep <- unique_adm1[seq_len(min(dev_subset_adm1_n, length(unique_adm1)))]
        dhis2_routine_long <- dhis2_routine_long[ADM1_ID %in% adm1_to_keep]
        log_msg(glue::glue("DEV_SUBSET enabled: keeping {length(adm1_to_keep)} ADM1 values"), "warning")
    }

    log_msg(glue::glue("Data loaded: {nrow(dhis2_routine_long)} rows, {length(unique(dhis2_routine_long$OU_ID))} facilities"))

    if (run_complete) {
        n_groups <- data.table::uniqueN(dhis2_routine_long[, .(OU_ID, INDICATOR)])
        log_msg(glue::glue("Complete mode active: seasonal detection will run on up to {n_groups} OU_ID x INDICATOR time series."), "warning")
    } else {
        log_msg("Partial mode active: seasonal detection is skipped.")
    }

    list(
        setup_ctx = setup_ctx,
        config_json = config_json,
        country_code = country_code,
        fixed_cols = fixed_cols,
        indicators_to_keep = indicators_to_keep,
        dhis2_routine = dhis2_routine,
        dhis2_routine_long = dhis2_routine_long
    )
}

run_magic_glasses_outlier_detection <- function(
    dhis2_routine_long,
    deviation_mad15 = 15,
    deviation_mad10 = 10,
    run_complete = FALSE,
    deviation_seasonal5 = 5,
    deviation_seasonal3 = 3,
    seasonal_workers = 1
) {
    log_msg("Starting MAD15 detection...")
    flagged_outliers_mad15 <- detect_outliers_mad_custom(dhis2_routine_long, deviation_mad15)
    flagged_outliers_mad15_filtered <- flagged_outliers_mad15[OUTLIER_MAD15 == FALSE]

    log_msg("Starting MAD10 detection...")
    flagged_outliers_mad10 <- detect_outliers_mad_custom(flagged_outliers_mad15_filtered, deviation_mad10)
    data.table::setnames(flagged_outliers_mad10, paste0("OUTLIER_MAD", deviation_mad10), "OUTLIER_MAD15_MAD10")

    join_cols <- c("PERIOD", "OU_ID", "INDICATOR")
    mad10_subset <- flagged_outliers_mad10[, .(PERIOD, OU_ID, INDICATOR, OUTLIER_MAD15_MAD10)]
    flagged_outliers_mad15_mad10 <- merge(
        flagged_outliers_mad15,
        mad10_subset,
        by = join_cols,
        all.x = TRUE
    )
    flagged_outliers_mad15_mad10[is.na(OUTLIER_MAD15_MAD10), OUTLIER_MAD15_MAD10 := TRUE]
    log_msg(glue::glue("MAD partial done: {sum(flagged_outliers_mad15_mad10$OUTLIER_MAD15_MAD10)} outliers flagged"))

    flagged_outliers_seasonal5_seasonal3 <- NULL
    if (run_complete) {
        flagged_outliers_mad15_mad10_filtered <- flagged_outliers_mad15_mad10[OUTLIER_MAD15_MAD10 == FALSE]

        if (nrow(flagged_outliers_mad15_mad10_filtered) == 0) {
            log_msg("No rows left after MAD partial filtering; seasonal step will be skipped.", "warning")
            flagged_outliers_seasonal5 <- data.table::copy(flagged_outliers_mad15_mad10_filtered)
            flagged_outliers_seasonal5[, OUTLIER_SEASONAL5 := FALSE]
            flagged_outliers_seasonal3 <- data.table::copy(flagged_outliers_seasonal5)
            flagged_outliers_seasonal3[, OUTLIER_SEASONAL3 := FALSE]
        } else {
            log_msg(glue::glue("Starting SEASONAL5 detection on {nrow(flagged_outliers_mad15_mad10_filtered)} rows..."))
            t_seasonal5 <- system.time({
                flagged_outliers_seasonal5 <- detect_seasonal_outliers(
                    flagged_outliers_mad15_mad10_filtered,
                    deviation = deviation_seasonal5,
                    workers = seasonal_workers
                )
            })
            flagged_outliers_seasonal5_filtered <- flagged_outliers_seasonal5[OUTLIER_SEASONAL5 == FALSE]
            log_msg(glue::glue("SEASONAL5 finished in {round(t_seasonal5['elapsed'], 1)}s. Remaining rows: {nrow(flagged_outliers_seasonal5_filtered)}"))

            log_msg(glue::glue("Starting SEASONAL3 detection on {nrow(flagged_outliers_seasonal5_filtered)} rows..."))
            t_seasonal3 <- system.time({
                flagged_outliers_seasonal3 <- detect_seasonal_outliers(
                    flagged_outliers_seasonal5_filtered,
                    deviation = deviation_seasonal3,
                    workers = seasonal_workers
                )
            })
            log_msg(glue::glue("SEASONAL3 finished in {round(t_seasonal3['elapsed'], 1)}s."))
        }

        data.table::setnames(flagged_outliers_seasonal3, paste0("OUTLIER_SEASONAL", deviation_seasonal3), "OUTLIER_SEASONAL5_SEASONAL3")

        seasonal3_subset <- flagged_outliers_seasonal3[, .(PERIOD, OU_ID, INDICATOR, OUTLIER_SEASONAL5_SEASONAL3)]
        flagged_outliers_seasonal5_seasonal3 <- merge(
            flagged_outliers_seasonal5,
            seasonal3_subset,
            by = join_cols,
            all.x = TRUE
        )
        flagged_outliers_seasonal5_seasonal3[is.na(OUTLIER_SEASONAL5_SEASONAL3), OUTLIER_SEASONAL5_SEASONAL3 := TRUE]
        log_msg(glue::glue("SEASONAL complete done: {sum(flagged_outliers_seasonal5_seasonal3$OUTLIER_SEASONAL5_SEASONAL3)} outliers flagged"))
    }

    list(
        flagged_outliers_mad15_mad10 = flagged_outliers_mad15_mad10,
        flagged_outliers_seasonal5_seasonal3 = flagged_outliers_seasonal5_seasonal3
    )
}

export_magic_glasses_outputs <- function(
    dhis2_routine_long,
    flagged_outliers_mad15_mad10,
    flagged_outliers_seasonal5_seasonal3,
    run_complete,
    dhis2_routine,
    fixed_cols,
    indicators_to_keep,
    output_dir,
    country_code
) {
    base_cols <- intersect(c(fixed_cols, "INDICATOR", "VALUE"), names(dhis2_routine_long))
    flagged_outliers_mg <- data.table::copy(dhis2_routine_long[, ..base_cols])
    join_cols <- c("PERIOD", "OU_ID", "INDICATOR")

    partial_subset <- flagged_outliers_mad15_mad10[, .(PERIOD, OU_ID, INDICATOR, OUTLIER_MAD15_MAD10)]
    flagged_outliers_mg <- merge(flagged_outliers_mg, partial_subset, by = join_cols, all.x = TRUE)
    data.table::setnames(flagged_outliers_mg, "OUTLIER_MAD15_MAD10", "OUTLIER_MAGIC_GLASSES_PARTIAL")

    if (run_complete && !is.null(flagged_outliers_seasonal5_seasonal3)) {
        complete_subset <- flagged_outliers_seasonal5_seasonal3[, .(PERIOD, OU_ID, INDICATOR, OUTLIER_SEASONAL5_SEASONAL3)]
        flagged_outliers_mg <- merge(flagged_outliers_mg, complete_subset, by = join_cols, all.x = TRUE)
        data.table::setnames(flagged_outliers_mg, "OUTLIER_SEASONAL5_SEASONAL3", "OUTLIER_MAGIC_GLASSES_COMPLETE")
        flagged_outliers_mg[is.na(OUTLIER_MAGIC_GLASSES_COMPLETE) & OUTLIER_MAGIC_GLASSES_PARTIAL == TRUE, OUTLIER_MAGIC_GLASSES_COMPLETE := TRUE]
    }

    flagged_outliers_mg[is.na(OUTLIER_MAGIC_GLASSES_PARTIAL), OUTLIER_MAGIC_GLASSES_PARTIAL := FALSE]
    if ("OUTLIER_MAGIC_GLASSES_COMPLETE" %in% names(flagged_outliers_mg)) {
        flagged_outliers_mg[is.na(OUTLIER_MAGIC_GLASSES_COMPLETE), OUTLIER_MAGIC_GLASSES_COMPLETE := FALSE]
    }

    active_outlier_col <- if (run_complete && "OUTLIER_MAGIC_GLASSES_COMPLETE" %in% names(flagged_outliers_mg)) {
        "OUTLIER_MAGIC_GLASSES_COMPLETE"
    } else {
        "OUTLIER_MAGIC_GLASSES_PARTIAL"
    }

    if (!(active_outlier_col %in% names(flagged_outliers_mg))) {
        stop(glue::glue("Expected outlier flag column not found: {active_outlier_col}"))
    }

    pyramid_names <- unique(data.table::as.data.table(dhis2_routine)[, .(
        ADM1_NAME, ADM1_ID, ADM2_NAME, ADM2_ID, OU_ID, OU_NAME
    )])

    outlier_method_label <- if (active_outlier_col == "OUTLIER_MAGIC_GLASSES_COMPLETE") "MAGIC_GLASSES_COMPLETE" else "MAGIC_GLASSES_PARTIAL"
    detected_tbl <- flagged_outliers_mg[, .(
        PERIOD, YEAR, MONTH, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, VALUE,
        OUTLIER_DETECTED = get(active_outlier_col),
        OUTLIER_METHOD = outlier_method_label
    )]
    detected_tbl[is.na(OUTLIER_DETECTED), OUTLIER_DETECTED := FALSE]
    detected_tbl <- merge(detected_tbl, unique(pyramid_names), by = c("ADM1_ID", "ADM2_ID", "OU_ID"), all.x = TRUE)
    detected_tbl[, DATE := as.Date(sprintf("%04d-%02d-01", YEAR, MONTH))]
    arrow::write_parquet(detected_tbl, file.path(output_dir, paste0(country_code, "_routine_outliers_detected.parquet")))
    n_out <- sum(detected_tbl$OUTLIER_DETECTED == TRUE)
    log_msg(glue::glue("Exported full detection table ({nrow(detected_tbl)} rows, {n_out} outliers) to {country_code}_routine_outliers_detected.parquet"))

    imputed_long <- data.table::copy(flagged_outliers_mg)
    data.table::setorder(imputed_long, ADM1_ID, ADM2_ID, OU_ID, INDICATOR, PERIOD, YEAR, MONTH)
    imputed_long[, TO_IMPUTE := data.table::fifelse(get(active_outlier_col) == TRUE, NA_real_, VALUE)]
    imputed_long[
        ,
        MOVING_AVG := data.table::frollapply(
            TO_IMPUTE,
            n = 3,
            FUN = function(x) ceiling(mean(x, na.rm = TRUE)),
            align = "center"
        ),
        by = .(ADM1_ID, ADM2_ID, OU_ID, INDICATOR)
    ]
    imputed_long[, VALUE_IMPUTED := data.table::fifelse(is.na(TO_IMPUTE), MOVING_AVG, TO_IMPUTE)]
    imputed_long[, VALUE := VALUE_IMPUTED]
    imputed_long[, c("TO_IMPUTE", "MOVING_AVG", "VALUE_IMPUTED") := NULL]

    routine_imputed <- to_routine_wide(
        dt_long = imputed_long,
        fixed_cols = fixed_cols,
        indicators_to_keep = indicators_to_keep,
        pyramid_names = pyramid_names
    )
    arrow::write_parquet(routine_imputed, file.path(output_dir, paste0(country_code, "_routine_outliers_imputed.parquet")))
    log_msg(glue::glue("Exported routine imputed table to {country_code}_routine_outliers_imputed.parquet"))

    removed_long <- data.table::copy(flagged_outliers_mg)
    removed_long[get(active_outlier_col) == TRUE, VALUE := NA_real_]

    routine_removed <- to_routine_wide(
        dt_long = removed_long,
        fixed_cols = fixed_cols,
        indicators_to_keep = indicators_to_keep,
        pyramid_names = pyramid_names
    )
    arrow::write_parquet(routine_removed, file.path(output_dir, paste0(country_code, "_routine_outliers_removed.parquet")))
    log_msg(glue::glue("Exported routine removed table to {country_code}_routine_outliers_removed.parquet"))

    log_msg("MG outlier tables exported successfully.")
    invisible(list(active_outlier_col = active_outlier_col))
}


# Shared helpers for snt_dhis2_formatting reporting notebook.

printdim <- function(df, name = deparse(substitute(df))) {
    if (is.null(df)) {
        cat("Dimensions of", name, ": NULL (not loaded)\n\n")
        return(invisible(NULL))
    }
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

detect_mad_outliers <- function(data_long, deviation = 15, outlier_column = "mad_flag") {
    data_long %>%
        dplyr::group_by(OU, indicator, YEAR) %>%
        dplyr::mutate(
            median_val = median(value, na.rm = TRUE),
            mad_val = mad(value, na.rm = TRUE),
            "{outlier_column}" := value > (median_val + deviation * mad_val) | value < (median_val - deviation * mad_val)
        ) %>%
        dplyr::ungroup()
}

create_dynamic_labels <- function(breaks) {
    fmt <- function(x) {
        format(x / 1000, big.mark = "'", scientific = FALSE, trim = TRUE)
    }

    c(
        paste0("< ", fmt(breaks[1]), "k"),
        paste0(fmt(breaks[-length(breaks)]), " - ", fmt(breaks[-1]), "k"),
        paste0("> ", fmt(breaks[length(breaks)]), "k")
    )
}


# Meme logique que expand_grid(OU, INDICATOR, DATE) + left_join(long_data), mais une
# grille OU x DATE a la fois par indicateur (RAM). Colonnes attendues = notebook
# snt_dhis2_formatting_report (OU, INDICATOR, DATE, VALUE).
reporting_summary_facility_chunked <- function(long_data) {
    req <- c("OU", "INDICATOR", "DATE", "VALUE")
    miss <- setdiff(req, names(long_data))
    if (length(miss) > 0L) {
        stop(paste0("[ERROR] reporting_summary_facility_chunked: missing columns: ", paste(miss, collapse = ", ")))
    }

    ld <- dplyr::transmute(
        long_data,
        OU = as.character(.data$OU),
        INDICATOR = as.character(.data$INDICATOR),
        DATE = as.Date(.data$DATE),
        VALUE = .data$VALUE
    )
    ld <- dplyr::filter(ld, !is.na(.data$INDICATOR), !is.na(.data$OU), !is.na(.data$DATE))
    ld <- dplyr::distinct(ld, OU, INDICATOR, DATE, .keep_all = TRUE)

    if (nrow(ld) == 0L) {
        return(dplyr::tibble(
            INDICATOR = character(),
            DATE = as.Date(character()),
            n_total = integer(),
            n_missing = integer(),
            n_zero = integer(),
            n_positive = integer(),
            pct_missing = numeric(),
            pct_zero = numeric(),
            pct_positive = numeric()
        ))
    }

    ou_levels <- dplyr::distinct(dplyr::select(ld, "OU"))
    date_levels <- dplyr::distinct(dplyr::select(ld, "DATE"))
    ou_date_grid <- tidyr::crossing(ou_levels, date_levels)
    chunks <- split(ld, ld$INDICATOR, drop = TRUE)

    out <- purrr::imap_dfr(chunks, function(chunk, ind) {
        ind <- as.character(ind)
        rhs <- dplyr::transmute(
            chunk,
            OU = as.character(.data$OU),
            INDICATOR = ind,
            DATE = as.Date(.data$DATE),
            VALUE = .data$VALUE
        )
        rhs <- dplyr::distinct(rhs, OU, INDICATOR, DATE, .keep_all = TRUE)
        dplyr::left_join(
            dplyr::mutate(ou_date_grid, INDICATOR = ind),
            rhs,
            by = c("OU", "INDICATOR", "DATE")
        ) %>%
            dplyr::group_by(.data$INDICATOR, .data$DATE) %>%
            dplyr::summarise(
                n_total = dplyr::n_distinct(.data$OU),
                n_missing = sum(is.na(.data$VALUE)),
                n_zero = sum(.data$VALUE == 0 & !is.na(.data$VALUE)),
                n_positive = sum(.data$VALUE > 0 & !is.na(.data$VALUE)),
                pct_missing = dplyr::if_else(.data$n_total > 0L, 100 * .data$n_missing / .data$n_total, 0),
                pct_zero = dplyr::if_else(.data$n_total > 0L, 100 * .data$n_zero / .data$n_total, 0),
                pct_positive = dplyr::if_else(.data$n_total > 0L, 100 * .data$n_positive / .data$n_total, 0),
                .groups = "drop"
            )
    })

    if (nrow(out) == 0L) {
        return(dplyr::tibble(
            INDICATOR = character(),
            DATE = as.Date(character()),
            n_total = integer(),
            n_missing = integer(),
            n_zero = integer(),
            n_positive = integer(),
            pct_missing = numeric(),
            pct_zero = numeric(),
            pct_positive = numeric()
        ))
    }
    dplyr::arrange(out, .data$INDICATOR, .data$DATE)
}


# Meme idee au niveau ADM2. Colonnes attendues = notebook (ADM2_ID, Date, Indicator, value).
reporting_summary_adm2_chunked <- function(data_long) {
    req <- c("ADM2_ID", "Date", "Indicator", "value")
    miss <- setdiff(req, names(data_long))
    if (length(miss) > 0L) {
        stop(paste0("[ERROR] reporting_summary_adm2_chunked: missing columns: ", paste(miss, collapse = ", ")))
    }

    dl <- dplyr::transmute(
        data_long,
        ADM2_ID = as.character(.data$ADM2_ID),
        Indicator = as.character(.data$Indicator),
        Date = as.Date(.data$Date),
        value = .data$value
    )
    dl <- dplyr::filter(dl, !is.na(.data$Indicator), !is.na(.data$ADM2_ID), !is.na(.data$Date))

    if (nrow(dl) == 0L) {
        return(dplyr::tibble(
            Indicator = character(),
            Date = as.Date(character()),
            n_total = integer(),
            n_missing = integer(),
            n_zero = integer(),
            n_positive = integer(),
            pct_missing = numeric(),
            pct_zero = numeric(),
            pct_positive = numeric()
        ))
    }

    adm_levels <- dplyr::distinct(dplyr::select(dl, "ADM2_ID"))
    date_levels <- dplyr::distinct(dplyr::select(dl, "Date"))
    adm_date_grid <- tidyr::crossing(adm_levels, date_levels)

    reporting_check <- dl %>%
        dplyr::group_by(.data$ADM2_ID, .data$Indicator, .data$Date) %>%
        dplyr::summarise(
            is_missing = all(is.na(.data$value)),
            is_zero = all(.data$value == 0, na.rm = TRUE),
            is_positive = any(.data$value > 0, na.rm = TRUE),
            .groups = "drop"
        )

    rc_chunks <- split(reporting_check, reporting_check$Indicator, drop = TRUE)

    out <- purrr::imap_dfr(rc_chunks, function(rc_chunk, ind) {
        ind <- as.character(ind)
        rc_chunk <- rc_chunk %>%
            dplyr::mutate(Date = as.Date(.data$Date)) %>%
            dplyr::distinct(ADM2_ID, Indicator, Date, .keep_all = TRUE)
        rhs <- dplyr::transmute(
            rc_chunk,
            ADM2_ID = as.character(.data$ADM2_ID),
            Indicator = ind,
            Date = as.Date(.data$Date),
            is_missing = .data$is_missing,
            is_zero = .data$is_zero,
            is_positive = .data$is_positive
        )
        rhs <- dplyr::distinct(rhs, ADM2_ID, Indicator, Date, .keep_all = TRUE)
        dplyr::left_join(
            dplyr::mutate(adm_date_grid, Indicator = ind),
            rhs,
            by = c("ADM2_ID", "Indicator", "Date")
        ) %>%
            dplyr::mutate(
                is_missing = tidyr::replace_na(.data$is_missing, TRUE),
                is_zero = tidyr::replace_na(.data$is_zero, FALSE),
                is_positive = tidyr::replace_na(.data$is_positive, FALSE)
            ) %>%
            dplyr::group_by(.data$Indicator, .data$Date) %>%
            dplyr::summarise(
                n_total = dplyr::n_distinct(.data$ADM2_ID),
                n_missing = sum(.data$is_missing, na.rm = TRUE),
                n_zero = sum(.data$is_zero & !.data$is_missing, na.rm = TRUE),
                n_positive = sum(.data$is_positive, na.rm = TRUE),
                pct_missing = dplyr::if_else(.data$n_total > 0L, 100 * .data$n_missing / .data$n_total, 0),
                pct_zero = dplyr::if_else(.data$n_total > 0L, 100 * .data$n_zero / .data$n_total, 0),
                pct_positive = dplyr::if_else(.data$n_total > 0L, 100 * .data$n_positive / .data$n_total, 0),
                .groups = "drop"
            )
    })

    if (nrow(out) == 0L) {
        return(dplyr::tibble(
            Indicator = character(),
            Date = as.Date(character()),
            n_total = integer(),
            n_missing = integer(),
            n_zero = integer(),
            n_positive = integer(),
            pct_missing = numeric(),
            pct_zero = numeric(),
            pct_positive = numeric()
        ))
    }
    dplyr::arrange(out, .data$Indicator, .data$Date)
}

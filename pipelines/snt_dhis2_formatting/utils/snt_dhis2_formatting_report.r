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


#' Facility-level reporting completeness (same logic as full OU x INDICATOR x DATE grid).
#'
#' Equivalent to expand_grid on distinct OU, INDICATOR, DATE from long_data, left-join
#' long_data, then summarise by INDICATOR and DATE. Processes one INDICATOR at a time so
#' peak memory scales with OU times DATE instead of OU times DATE times INDICATOR.
#'
#' Doublons sur (OU, INDICATOR, DATE): un left_join avec plusieurs lignes a droite pour la
#' meme cle duplique les lignes a gauche et fausse les comptages. Les donnees sont donc
#' dedupliquees avant nest / join (premiere ligne conservee par cle).
#'
#' Optimisations: colonnes minimales, group_nest (un seul passage pour splitter),
#' pas de filter repete sur tout le jeu, comptages et pourcentages en un summarise.
#'
#' @param long_data Tibble with columns OU, INDICATOR, DATE, VALUE (wide routine pivoted long).
#' @return Tibble with INDICATOR, DATE, n_total, n_missing, n_zero, n_positive, pct_*.
reporting_summary_facility_chunked <- function(long_data) {
    ld <- dplyr::transmute(
        long_data,
        OU = as.character(.data$OU),
        INDICATOR = as.character(.data$INDICATOR),
        DATE = as.Date(.data$DATE),
        VALUE = .data$VALUE
    )
    # Cle join unique (evite many-to-many / lignes dupliquees au left_join)
    ld <- dplyr::distinct(ld, OU, INDICATOR, DATE, .keep_all = TRUE)

    ou_levels <- dplyr::distinct(dplyr::select(ld, "OU"))
    date_levels <- dplyr::distinct(dplyr::select(ld, "DATE"))
    ou_date_grid <- tidyr::crossing(ou_levels, date_levels)

    nested <- dplyr::group_nest(ld, INDICATOR)

    out <- purrr::map2_dfr(nested$INDICATOR, nested$data, function(ind, chunk) {
        # group_nest() retire INDICATOR des lignes nested : joindre seulement sur OU + DATE
        dplyr::left_join(
            dplyr::mutate(ou_date_grid, INDICATOR = as.character(ind)),
            chunk,
            by = c("OU", "DATE")
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

    dplyr::arrange(out, .data$INDICATOR, .data$DATE)
}


#' ADM2-level reporting completeness (same logic as full ADM2 x Indicator x Date grid).
#'
#' reporting_check est deja unique sur (ADM2_ID, Indicator, Date) apres le group_by
#' summarise. Le left_join avec adm_date_grid reste au plus une ligne droite par ligne
#' gauche; rc_chunk est quand meme passe en distinct par securite.
#'
#' Optimisations: transmute pour reduire les colonnes, group_nest sur reporting_check
#' pour eviter les filtres repetes, pourcentages dans le meme summarise que les comptages.
#'
#' @param data_long Tibble with ADM2_ID, OU_ID, Date, Indicator, value.
#' @return Tibble with Indicator, Date, n_total, n_missing, n_zero, n_positive, pct_*.
reporting_summary_adm2_chunked <- function(data_long) {
    dl <- dplyr::transmute(
        data_long,
        ADM2_ID = as.character(.data$ADM2_ID),
        OU_ID = as.character(.data$OU_ID),
        Date = as.Date(.data$Date),
        Indicator = as.character(.data$Indicator),
        value = .data$value
    )

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

    nested_rc <- dplyr::group_nest(reporting_check, Indicator)

    out <- purrr::map2_dfr(nested_rc$Indicator, nested_rc$data, function(ind, rc_chunk) {
        # group_nest() retire Indicator des lignes nested : joindre sur ADM2_ID + Date seulement
        rc_chunk <- rc_chunk %>%
            dplyr::mutate(Date = as.Date(.data$Date)) %>%
            dplyr::distinct(ADM2_ID, Date, .keep_all = TRUE)
        dplyr::left_join(
            dplyr::mutate(adm_date_grid, Indicator = as.character(ind)),
            rc_chunk,
            by = c("ADM2_ID", "Date")
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

    dplyr::arrange(out, .data$Indicator, .data$Date)
}

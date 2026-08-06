# Shared helpers for snt_dhis2_extract notebooks.

# To move to  ./code/snt_utils.r ... (see https://bluesquare.atlassian.net/browse/SNT25-591 )
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

open_in_year <- function(df, y) {
    y <- as.integer(y)
    year_start <- as.Date(sprintf("%s-01-01", y))
    year_end <- as.Date(sprintf("%s-12-31", y))
    df %>%
        dplyr::filter(
            as.Date(OPENING_DATE) <= year_end,
            is.na(CLOSED_DATE) | as.Date(CLOSED_DATE) >= year_start
        ) %>%
        dplyr::summarise(Annee = y, Ouvertes_pyramide = dplyr::n(), .groups = "drop")
}

norm_fosa_type <- function(x) {
    x_up <- stringr::str_to_upper(stringr::str_squish(x))
    dplyr::case_when(
        stringr::str_detect(x_up, "^HD\\b") ~ "HD (hôpital de district)",
        stringr::str_detect(x_up, "^CSI\\b") ~ "CSI (centre de santé intégré)",
        stringr::str_detect(x_up, "^CS\\b") ~ "CS (case de santé)",
        stringr::str_detect(x_up, "^(SS\\b|SALLE\\b|SALLE D'ACCOUCHEMENT\\b)") ~ "SS / Salle (soins/maternité)",
        stringr::str_detect(x_up, "^(CLINIQUE|POLYCLINIQUE)\\b") ~ "Clinique (privé)",
        stringr::str_detect(x_up, "^CABINET\\b") ~ "Cabinet (privé)",
        stringr::str_detect(x_up, "^(INFIRMERIE|INFIRM)\\b") ~ "Infirmerie (privé)",
        stringr::str_detect(x_up, "^CNSS\\b") ~ "CNSS",
        TRUE ~ "Autre"
    )
}


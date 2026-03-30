# Shared helpers for snt_dhis2_extract notebooks.

make_point_geojson <- function(lat, lon) {
    sprintf('{"type": "Point", "coordinates": [%f, %f]}', lon, lat)
}

apply_ner_manual_geometry_fixes <- function(group_prioritaires_table) {
    manual_points <- data.frame(
        id = c(
            "xMqXanPgczy",
            "sgO4yBg59SJ",
            "oHRvIBeR5xH",
            "TVaP0vBLvat",
            "evMtQ7bLFYI",
            "u3xCSh4hG9Q",
            "P1oyCQT39rj"
        ),
        lat = c(
            14.212177799561589,
            13.485271755127068,
            13.551421362165923,
            13.509657990942971,
            13.586255600670649,
            13.509793678687808,
            13.535431049590938
        ),
        lon = c(
            1.4625739941131144,
            7.143422105623865,
            2.116344191939423,
            2.1473435456528174,
            2.0918749136394097,
            2.147386518669057,
            2.09186651126039
        ),
        stringsAsFactors = FALSE
    )

    for (i in seq_len(nrow(manual_points))) {
        this_id <- manual_points$id[[i]]
        group_prioritaires_table[group_prioritaires_table$id == this_id, ]$geometry <-
            make_point_geojson(manual_points$lat[[i]], manual_points$lon[[i]])
    }

    group_prioritaires_table
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

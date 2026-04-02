# Shared helpers for snt_dhis2_extract notebooks.

bootstrap_dhis2_extract_context <- function(
    root_path = "~/workspace",
    required_packages = c("arrow", "dplyr", "tidyverse", "jsonlite", "reticulate", "glue", "sf"),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    pipeline_path <- file.path(root_path, "pipelines", "snt_dhis2_extract")

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")
    openhexa <- NULL
    openhexa_toolbox <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
        openhexa_toolbox <- reticulate::import("openhexa.toolbox")
    }
    assign("openhexa", openhexa, envir = .GlobalEnv)
    assign("openhexa_toolbox", openhexa_toolbox, envir = .GlobalEnv)

    list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        PIPELINE_PATH = pipeline_path,
        openhexa = openhexa,
        openhexa_toolbox = openhexa_toolbox
    )
}

printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

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

map_points_to_ds_polygons <- function(points_sf, polygons_sf) {
    inside_matrix <- sf::st_within(points_sf, polygons_sf, sparse = FALSE)
    point_polygon_dict <- list()

    for (i in seq_len(nrow(points_sf))) {
        point_id <- points_sf$id[[i]]
        point_name <- points_sf$name[[i]]
        polygons_containing <- which(inside_matrix[i, ])

        if (length(polygons_containing) > 0) {
            found_polygons <- polygons_sf[polygons_containing, ]
            found_polygons_ds <- found_polygons[grepl("^DS", found_polygons$name), ]

            if (nrow(found_polygons_ds) >= 1) {
                polygon_id <- found_polygons_ds$id[1]
                polygon_name <- found_polygons_ds$name[1]

                point_polygon_dict[[point_id]] <- list(
                    point_name = point_name,
                    polygon_id = polygon_id,
                    polygon_name = polygon_name
                )
                print(glue::glue("Point: {point_name} ({point_id}) is inside polygon: {polygon_name} ({polygon_id})"))
            } else {
                point_polygon_dict[[point_id]] <- list(
                    point_name = point_name,
                    polygon_id = NA,
                    polygon_name = NA
                )
                cat("Point:", point_id, "is not inside any district (DS) polygon\n")
            }
        } else {
            point_polygon_dict[[point_id]] <- list(
                point_name = point_name,
                polygon_id = NA,
                polygon_name = NA
            )
            cat("Point:", point_id, "is not inside any district (DS) polygon\n")
        }
    }

    point_polygon_dict
}

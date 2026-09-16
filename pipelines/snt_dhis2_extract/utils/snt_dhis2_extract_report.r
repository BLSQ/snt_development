# Helper function for snt_dhis2_extract report.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r")) 

#' Print Data Frame Dimensions
#'
#' Prints the number of rows and columns of a data frame to the console,
#' labelled with the data frame's name.
#'
#' @param df Data frame. Table whose dimensions will be printed.
#' @param name Character. Label used in the printed message. Default: the
#'   deparsed expression passed as `df`.
#' @return Invisible NULL. Called for its side effect of printing to console.
#'
#' @export
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

#' Build a GeoJSON Point String
#'
#' Formats a latitude/longitude pair as a GeoJSON Point geometry string.
#'
#' @param lat Numeric. Latitude coordinate.
#' @param lon Numeric. Longitude coordinate.
#' @return Character. GeoJSON Point geometry string.
#'
#' @export
make_point_geojson <- function(lat, lon) {
    sprintf('{"type": "Point", "coordinates": [%f, %f]}', lon, lat)
}

#' Apply Manual Geometry Fixes for Niger Priority Org Unit Groups
#'
#' Overwrites the `geometry` column with a hardcoded set of latitude/longitude
#' coordinates, converted to GeoJSON points, for a fixed list of Niger
#' priority organisation unit ids ("groupes prioritaires") whose geometry is
#' otherwise missing.
#'
#' @param group_prioritaires_table Data frame. Organisation units with `id`
#'   and `geometry` columns.
#' @return Data frame. The input table with `geometry` overwritten for the
#'   hardcoded ids.
#'
#' @export
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

#' Count Organisation Units Open During a Calendar Year
#'
#' Filters organisation units to those open at any point during the given
#' calendar year, based on `OPENING_DATE` and `CLOSED_DATE`, and returns the
#' count.
#'
#' @param df Data frame. Organisation units with `OPENING_DATE` and
#'   `CLOSED_DATE` columns.
#' @param y Integer or character. Calendar year to check, e.g. `2023`.
#' @return Data frame with one row and columns `Annee` (the year) and
#'   `Ouvertes_pyramide` (count of open organisation units).
#'
#' @export
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

#' Normalize Health Facility Type Labels
#'
#' Maps free-text health facility type strings (French abbreviations such as
#' `HD`, `CSI`, `CS`) to a small set of standardized category labels using
#' prefix matching, after uppercasing and squishing whitespace.
#'
#' @param x Character vector. Raw facility type labels.
#' @return Character vector. Standardized facility type category for each
#'   input value, or `"Autre"` when no pattern matches.
#'
#' @export
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

#' Map Points to Containing District (DS) Polygons
#'
#' For each point, finds the polygons that spatially contain it and keeps
#' only those whose name starts with `"DS"` (district sanitaire). Logs, via
#' `print`/`cat`, the match found (or the lack of one) for every point.
#'
#' @param points_sf sf object. Points with `id` and `name` columns.
#' @param polygons_sf sf object. Candidate polygons with `id` and `name`
#'   columns.
#' @return Named list, keyed by point id, of lists with `point_name`,
#'   `polygon_id` and `polygon_name` (the latter two `NA` when the point
#'   falls outside every district polygon).
#'
#' @export
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

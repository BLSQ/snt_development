printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

plot_outliers <- function(ind_name, df, outlier_col) {
    df_ind <- df %>% dplyr::filter(INDICATOR == ind_name)
    df_ind <- df_ind %>% dplyr::filter(!is.na(YEAR), !is.na(VALUE), is.finite(VALUE))

    ggplot2::ggplot(df_ind, ggplot2::aes(x = YEAR, y = VALUE)) +
        ggplot2::geom_point(alpha = 0.25, color = "grey40", na.rm = TRUE) +
        ggplot2::geom_point(
            data = df_ind %>% dplyr::filter(.data[[outlier_col]] == TRUE),
            ggplot2::aes(x = YEAR, y = VALUE),
            color = "red",
            size = 2.8,
            alpha = 0.85,
            na.rm = TRUE
        ) +
        ggplot2::labs(
            title = paste("Inspection des valeurs aberrantes pour indicateur:", ind_name),
            subtitle = "Gris = toutes les valeurs • Rouge = valeurs aberrantes détectées",
            x = "Année",
            y = "Valeur"
        ) +
        ggplot2::theme_minimal(base_size = 14)
}

plot_outliers_by_district_facet_year <- function(ind_name, df, outlier_col) {
    df_ind <- df %>%
        dplyr::filter(
            INDICATOR == ind_name,
            !is.na(YEAR),
            !is.na(VALUE),
            is.finite(VALUE)
        )

    if (nrow(df_ind) == 0) {
        return(NULL)
    }

    ggplot2::ggplot(df_ind, ggplot2::aes(x = ADM2_ID, y = VALUE)) +
        ggplot2::geom_point(color = "grey60", alpha = 0.3) +
        ggplot2::geom_point(
            data = df_ind %>% dplyr::filter(.data[[outlier_col]] == TRUE),
            color = "red",
            size = 2.8,
            alpha = 0.85
        ) +
        ggplot2::facet_wrap(~ YEAR, scales = "free_y") +
        ggplot2::labs(
            title = paste("Détection des valeurs aberrantes —", ind_name),
            subtitle = paste("Méthode :", outlier_col, "| Rouge = valeur aberrante"),
            x = "District (ADM2)",
            y = "Valeur"
        ) +
        ggplot2::theme_minimal(base_size = 13) +
        ggplot2::theme(
            axis.text.x = ggplot2::element_text(angle = 75, hjust = 1, size = 7)
        )
}

plot_coherence_heatmap <- function(df, selected_year, agg_level = "ADM1_NAME", filename = NULL, do_plot = TRUE) {
    if (!agg_level %in% names(df)) {
        stop(paste0("Aggregation level '", agg_level, "' not found in data!"))
    }

    df_year <- df %>%
        dplyr::filter(YEAR == selected_year) %>%
        dplyr::group_by(dplyr::across(dplyr::all_of(c(agg_level, "check_label")))) %>%
        dplyr::summarise(
            pct_coherent = mean(pct_coherent, na.rm = TRUE),
            .groups = "drop"
        ) %>%
        dplyr::group_by(dplyr::across(dplyr::all_of(agg_level))) %>%
        dplyr::mutate(median_coh = median(pct_coherent, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        dplyr::mutate(!!agg_level := forcats::fct_reorder(.data[[agg_level]], median_coh))

    n_units <- dplyr::n_distinct(df_year[[agg_level]])
    plot_height <- max(6, 0.5 * n_units)
    agg_label <- if (agg_level == "ADM1_NAME") {
        "niveau administratif 1"
    } else if (agg_level == "ADM2_NAME") {
        "niveau administratif 2"
    } else {
        agg_level
    }

    p <- ggplot2::ggplot(df_year, ggplot2::aes(x = check_label, y = .data[[agg_level]], fill = pct_coherent)) +
        ggplot2::geom_tile(color = "white", linewidth = 0.2) +
        ggplot2::geom_text(
            ggplot2::aes(label = sprintf("%.0f%%", pct_coherent)),
            size = 5,
            fontface = "bold",
            color = "white"
        ) +
        viridis::scale_fill_viridis(
            name = "% cohérent",
            limits = c(0, 100),
            option = "viridis",
            direction = -1
        ) +
        ggplot2::labs(
            title = paste0("Cohérence des données par ", agg_label, " - ", selected_year),
            x = "Règle de cohérence",
            y = agg_label
        ) +
        ggplot2::theme_minimal(base_size = 14) +
        ggplot2::theme(
            panel.grid = ggplot2::element_blank(),
            axis.text.y = ggplot2::element_text(size = 12),
            axis.text.x = ggplot2::element_text(size = 12, angle = 30, hjust = 1),
            plot.title = ggplot2::element_text(size = 16, face = "bold", hjust = 0.5),
            legend.title = ggplot2::element_text(size = 12),
            legend.text = ggplot2::element_text(size = 10)
        )

    options(repr.plot.width = 14, repr.plot.height = plot_height)

    if (!is.null(filename)) {
        ggplot2::ggsave(
            filename = filename,
            plot = p,
            width = 14,
            height = plot_height,
            dpi = 300,
            limitsize = FALSE
        )
    }
    if (do_plot) {
        print(p)
    }
}

plot_coherence_map <- function(map_data, col_name, indicator_label = NULL) {
    if (!col_name %in% names(map_data)) {
        stop(paste0("Column '", col_name, "' not found in the data!"))
    }

    if (is.null(indicator_label)) {
        indicator_label <- col_name
    }

    ggplot2::ggplot(map_data) +
        ggplot2::geom_sf(ggplot2::aes(fill = .data[[col_name]]), color = "white", size = 0.2) +
        viridis::scale_fill_viridis(
            name = paste0("% cohérence\n(", indicator_label, ")"),
            option = "magma",
            direction = -1,
            limits = c(0, 100),
            na.value = "grey90"
        ) +
        ggplot2::facet_wrap(~ YEAR, drop = TRUE) +
        ggplot2::labs(
            title = "Cohérence des données par niveau administratif 2 et par année",
            subtitle = paste("Indicateur :", indicator_label),
            caption = "Source : DHIS2 données routinières"
        ) +
        ggplot2::theme_minimal(base_size = 15) +
        ggplot2::theme(
            panel.grid = ggplot2::element_blank(),
            strip.text = ggplot2::element_text(size = 14, face = "bold"),
            plot.title = ggplot2::element_text(size = 20, face = "bold"),
            legend.position = "right"
        )
}

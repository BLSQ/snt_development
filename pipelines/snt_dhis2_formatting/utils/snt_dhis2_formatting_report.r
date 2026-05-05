# Helpers et orchestration pour le rapport Jupyter / Papermill (snt_dhis2_formatting_report.ipynb).

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


# ---- Helpers utilisés par les cellules du rapport (mémoire, plafonds, agrégations) ----

#' Limite le nombre d'indicateurs distincts (ordre alphabétique, comme la section 1.2).
report_limit_long_indicators <- function(long_data, indicator_col = "INDICATOR", max_n = 40L) {
    max_n <- suppressWarnings(as.integer(max_n)[1])
    if (is.null(long_data) || !NROW(long_data)) {
        return(long_data)
    }
    if (!indicator_col %in% names(long_data)) {
        stop("report_limit_long_indicators: colonne manquante : ", indicator_col)
    }
    if (!length(max_n) || is.na(max_n) || max_n < 1L) {
        return(long_data)
    }
    lvls <- sort(unique(as.character(long_data[[indicator_col]])))
    if (length(lvls) <= max_n) {
        return(long_data)
    }
    keep <- lvls[seq_len(max_n)]
    dplyr::filter(long_data, as.character(.data[[indicator_col]]) %in% keep)
}


#' Supprime des objets du `.GlobalEnv` puis `gc` (allège la RAM entre grosses étapes).
report_release_objects <- function(..., full_gc = FALSE) {
    nms <- unique(as.character(unlist(list(...), use.names = FALSE)))
    for (n in nms) {
        if (exists(n, envir = .GlobalEnv, inherits = FALSE)) {
            rm(list = n, envir = .GlobalEnv)
        }
    }
    if (isTRUE(full_gc)) {
        invisible(gc(verbose = FALSE, full = TRUE))
    } else {
        invisible(gc(verbose = FALSE, full = FALSE))
    }
    invisible(NULL)
}


#' Garde les `n_months` derniers mois calendaires par rapport au max de `date_col`.
report_filter_recent_months <- function(df, date_col, n_months) {
    if (is.null(df) || !NROW(df)) {
        return(df)
    }
    n_months <- suppressWarnings(as.integer(n_months)[1])
    if (!length(n_months) || is.na(n_months) || n_months < 1L) {
        return(df)
    }
    d <- as.Date(df[[date_col]])
    maxd <- suppressWarnings(max(d, na.rm = TRUE))
    if (!is.finite(as.numeric(maxd))) {
        return(df)
    }
    cutoff <- suppressWarnings(min(seq.Date(maxd, by = "-1 month", length.out = n_months), na.rm = TRUE))
    dplyr::filter(df, !is.na(.data[[date_col]]), as.Date(.data[[date_col]]) >= cutoff)
}


#' Exécute un bloc graphique.
run_report_plot <- function(label, plot_fn) {
    if (exists("report_plots_skip", mode = "function", inherits = TRUE) && isTRUE(report_plots_skip())) {
        message("[SNT_FORMAT_REPORT_PLOTS=none] Bloc omis : ", label)
        return(invisible(NULL))
    }
    plot_fn()
}


#' TRUE si le tracé doit être omis (politique `none` ou trop de lignes vs `SNT_REPORT_PLOT_MAX_ROWS`).
skip_heavy_plot_input <- function(df, label = "plot input") {
    if (exists("report_plots_skip", mode = "function", inherits = TRUE) && isTRUE(report_plots_skip())) {
        return(TRUE)
    }
    lim <- suppressWarnings(as.integer(Sys.getenv("SNT_REPORT_PLOT_MAX_ROWS", unset = "1000000"))[1])
    if (!is.finite(lim) || lim < 1L) {
        lim <- 1000000L
    }
    nr <- if (inherits(df, "sf")) {
        nrow(sf::st_drop_geometry(df))
    } else {
        nrow(as.data.frame(df))
    }
    if (nr > lim) {
        message("[skip] ", label, ": ", nr, " lignes > ", lim, " (SNT_REPORT_PLOT_MAX_ROWS)")
        return(TRUE)
    }
    FALSE
}


#' Sous-échantillon aléatoire simple pour limiter le coût des nuages de points.
report_downsample_rows <- function(df, max_rows) {
    max_rows <- suppressWarnings(as.integer(max_rows)[1])
    if (is.null(df) || !NROW(df)) {
        return(df)
    }
    if (!length(max_rows) || is.na(max_rows) || max_rows < 1L) {
        return(df)
    }
    nr <- nrow(df)
    if (nr <= max_rows) {
        return(df)
    }
    i <- sample.int(nr, max_rows)
    df[i, , drop = FALSE]
}


#' Agrège la routine large en ADM2 × mois (sommes sur les OU) pour indicateurs composites usuels.
routine_adm2_month_from_wide <- function(routine_data) {
    req <- c("PERIOD", "ADM2_ID")
    miss <- setdiff(req, names(routine_data))
    if (length(miss)) {
        stop("routine_adm2_month_from_wide: colonnes manquantes : ", paste(miss, collapse = ", "))
    }
    inds <- intersect(c("SUSP", "TEST", "CONF", "MALTREAT", "PRES"), names(routine_data))
    b <- routine_data %>%
        dplyr::mutate(DATE = as.Date(paste0(as.character(.data$PERIOD), "01"), format = "%Y%m%d"))
    if (length(inds)) {
        b %>%
            dplyr::group_by(.data$ADM2_ID, .data$PERIOD, .data$DATE) %>%
            dplyr::summarise(
                dplyr::across(dplyr::all_of(inds), ~ sum(as.numeric(.x), na.rm = TRUE)),
                .groups = "drop"
            )
    } else {
        dplyr::distinct(b, .data$ADM2_ID, .data$PERIOD, .data$DATE)
    }
}


#' Totaux nationaux par mois ; retire en fin de série les mois entièrement à 0 (extrait incomplet).
routine_national_month_from_wide <- function(routine_data) {
    vars_nat <- intersect(c("SUSP", "TEST", "CONF", "PRES"), names(routine_data))
    if (!length(vars_nat) || !"PERIOD" %in% names(routine_data)) {
        return(dplyr::tibble(DATE = as.Date(character())))
    }
    out <- routine_data %>%
        dplyr::mutate(DATE = as.Date(paste0(as.character(.data$PERIOD), "01"), format = "%Y%m%d")) %>%
        dplyr::group_by(.data$DATE) %>%
        dplyr::summarise(
            dplyr::across(dplyr::all_of(vars_nat), ~ sum(as.numeric(.x), na.rm = TRUE)),
            .groups = "drop"
        ) %>%
        dplyr::arrange(.data$DATE)
    if (!NROW(out)) {
        return(out)
    }
    m <- as.matrix(out[, vars_nat, drop = TRUE])
    all_zero <- apply(m, 1L, function(row) all(row == 0 & !is.na(row)))
    while (length(all_zero) && all_zero[length(all_zero)]) {
        out <- out[-nrow(out), , drop = FALSE]
        if (!NROW(out)) {
            break
        }
        m <- as.matrix(out[, vars_nat, drop = TRUE])
        all_zero <- apply(m, 1L, function(row) all(row == 0 & !is.na(row)))
    }
    out
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


# ---- Bootstrap notebook report + chargements parquet ----

.report_param_or_default <- function(name, default) {
    if (!exists(name, envir = .GlobalEnv, inherits = FALSE)) {
        return(default)
    }
    value <- get(name, envir = .GlobalEnv, inherits = FALSE)
    if (length(value) == 0L || all(is.na(value))) {
        return(default)
    }
    value
}


formatting_report_build_paths <- function(snt_root_path = "~/workspace") {
    root <- path.expand(as.character(snt_root_path)[1])
    pipeline_path <- file.path(root, "pipelines", "snt_dhis2_formatting")
    reporting_path <- file.path(pipeline_path, "reporting")
    figures_dir <- file.path(reporting_path, "outputs", "figures")
    dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
    list(
        SNT_ROOT_PATH    = root,
        CODE_PATH        = file.path(root, "code"),
        CONFIG_PATH      = file.path(root, "configuration"),
        PIPELINE_PATH    = pipeline_path,
        REPORTING_NB_PATH = reporting_path,
        figures_dir      = figures_dir
    )
}


formatting_report_read_snt_config <- function(config_path) {
    tryCatch(
        jsonlite::fromJSON(file.path(config_path, "SNT_config.json")),
        error = function(e) {
            stop("Erreur chargement configuration : ", conditionMessage(e))
        }
    )
}


formatting_report_assign_ids_from_config <- function(config_json) {
    list(
        dataset_name = config_json$SNT_DATASET_IDENTIFIERS$DHIS2_DATASET_FORMATTED,
        COUNTRY_CODE = config_json$SNT_CONFIG$COUNTRY_CODE,
        COUNTRY_NAME = config_json$SNT_CONFIG$COUNTRY_NAME,
        ADM_2        = toupper(config_json$SNT_CONFIG$DHIS2_ADMINISTRATION_2)
    )
}


# Bootstrap unique du report (paths + packages + SDK + config + IDs).
snt_setup_report <- function(
    snt_root_path = "~/workspace",
    report_max_years = .report_param_or_default("REPORT_MAX_YEARS", 10L),
    report_plot_months = .report_param_or_default("REPORT_PLOT_MONTHS", 36L),
    report_max_indicators = .report_param_or_default("REPORT_MAX_INDICATORS", 40L),
    report_pop_scatter_max_rows = .report_param_or_default("REPORT_POP_SCATTER_MAX_ROWS", 4000L),
    report_shape_simplify_tol = .report_param_or_default("REPORT_SHAPE_SIMPLIFY_TOL", 0.002),
    report_fig_dpi = .report_param_or_default("REPORT_FIG_DPI", 120L),
    required_packages = c(
        "dplyr", "tidyr", "ggplot2", "forcats", "lubridate", "stringr", "purrr", "rlang",
        "scales", "arrow", "sf", "reticulate", "patchwork", "jsonlite", "httr", "IRdisplay"
    )
) {
    paths <- formatting_report_build_paths(snt_root_path)
    source(file.path(paths$CODE_PATH, "snt_utils.r"))
    source(file.path(paths$PIPELINE_PATH, "utils", "snt_dhis2_formatting.r"))
    snt_setup(
        SNT_ROOT_PATH = paths$SNT_ROOT_PATH,
        packages = required_packages
    )
    if (Sys.getenv("SNT_REPORT_PLOT_MAX_ROWS", unset = "") == "") {
        Sys.setenv(SNT_REPORT_PLOT_MAX_ROWS = "800000")
    }
    config_json <- formatting_report_read_snt_config(paths$CONFIG_PATH)
    ids <- formatting_report_assign_ids_from_config(config_json)
    sdk_available <- exists("openhexa", inherits = TRUE) && !is.null(get("openhexa", inherits = TRUE))
    if (!sdk_available) {
        warning("OpenHEXA SDK is not loaded or available.")
    }
    c(
        paths,
        list(
            REPORT_MAX_YEARS = suppressWarnings(as.integer(report_max_years)[1]),
            REPORT_PLOT_MONTHS = suppressWarnings(as.integer(report_plot_months)[1]),
            REPORT_MAX_INDICATORS = suppressWarnings(as.integer(report_max_indicators)[1]),
            REPORT_POP_SCATTER_MAX_ROWS = suppressWarnings(as.integer(report_pop_scatter_max_rows)[1]),
            REPORT_SHAPE_SIMPLIFY_TOL = suppressWarnings(as.numeric(report_shape_simplify_tol)[1]),
            REPORT_FIG_DPI = suppressWarnings(as.integer(report_fig_dpi)[1]),
            config_json = config_json,
            openhexa_available = sdk_available
        ),
        ids
    )
}


# Simplifie les shapes et retourne l'objet transforme.
formatting_report_simplify_shapes <- function(shapes_data, simplify_tol = 0.002) {
    if (is.null(shapes_data)) stop("formatting_report_simplify_shapes: shapes_data is required.")
    tol <- suppressWarnings(as.numeric(simplify_tol)[1])
    if (!is.finite(tol)) stop("formatting_report_simplify_shapes: simplify_tol must be numeric.")
    sf::st_simplify(shapes_data, dTolerance = tol, preserveTopology = TRUE)
}


# Routine parquet + filtre annees + printdim.
formatting_report_load_routine_data <- function(setup) {
    routine_data <- load_dataset_file(
        setup$dataset_name,
        paste0(setup$COUNTRY_CODE, "_routine.parquet"),
        verbose = TRUE
    )
    max_years <- suppressWarnings(as.integer(setup$REPORT_MAX_YEARS)[1])
    if (is.finite(max_years) && max_years > 0L && "YEAR" %in% names(routine_data)) {
        y_end <- suppressWarnings(max(as.numeric(routine_data$YEAR), na.rm = TRUE))
        routine_data <- dplyr::filter(
            routine_data,
            suppressWarnings(as.numeric(YEAR)) > y_end - max_years
        )
    }
    printdim(routine_data)
    routine_data
}


# Population parquet + printdim.
formatting_report_load_population_data <- function(setup) {
    population_data <- load_dataset_file(
        setup$dataset_name,
        paste0(setup$COUNTRY_CODE, "_population.parquet"),
        verbose = TRUE
    )
    printdim(population_data)
    population_data
}


# Shapes geojson + printdim.
formatting_report_load_shapes_data <- function(setup) {
    shapes_data <- load_dataset_file(
        setup$dataset_name,
        paste0(setup$COUNTRY_CODE, "_shapes.geojson"),
        verbose = TRUE
    )
    printdim(shapes_data)
    shapes_data
}

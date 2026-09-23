# Shared helpers for snt_dhis2_formatting reporting notebook.

# Load base utils
source(file.path("~/workspace/code", "snt_utils.r"))   

#' Print the Dimensions of a Data Frame
#'
#' Prints the number of rows and columns of a data frame to the console,
#' labelled with the object's name.
#'
#' @param df Data frame whose dimensions will be printed.
#' @param name Character. Label used in the printed message. Default: the
#'   deparsed expression passed as `df`.
#' @return Invisibly, `NULL`. Called for its side effect of printing.
#'
#' @export
printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

#' Flag Outliers Using the MAD Method
#'
#' Groups a long-format data frame by organisation unit, indicator and year,
#' then flags values that fall more than `deviation` median absolute
#' deviations (MAD) away from the group median.
#'
#' @param data_long Data frame in long format with `OU`, `indicator`, `YEAR`
#'   and `value` columns.
#' @param deviation Numeric. Number of MADs from the median beyond which a
#'   value is flagged as an outlier. Default: 15.
#' @param outlier_column Character. Name of the logical column added to flag
#'   outliers. Default: "mad_flag".
#' @return `data_long` with `median_val`, `mad_val` and `outlier_column`
#'   columns added.
#'
#' @export
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

#' Create Dynamic Bucket Labels for Binned Breaks
#'
#' Builds human-readable labels, in thousands (e.g. "< 5k", "5 - 10k",
#' "> 50k"), for a numeric vector of bin breakpoints, for use as histogram
#' or legend labels.
#'
#' @param breaks Numeric vector of bin breakpoints, in ascending order.
#' @return Character vector of length `length(breaks) + 1`: a "< "
#'   label for values below the first break, one "lo - hi" label per
#'   interval between consecutive breaks, and a "> " label for values
#'   above the last break.
#'
#' @export
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

# Shared helpers for snt_dhis2_population_transformation reporting notebook.

printdim <- function(df, name = deparse(substitute(df))) {
    cat("Dimensions of", name, ":", nrow(df), "rows x", ncol(df), "columns\n\n")
}

create_dynamic_labels <- function(breaks) {
    fmt <- function(x) {
        format(x / 1000, big.mark = "'", scientific = FALSE, trim = TRUE)
    }

    labels <- c(
        paste0("< ", fmt(breaks[1]), "k"),
        paste0(fmt(breaks[-length(breaks)]), " - ", fmt(breaks[-1]), "k"),
        paste0("> ", fmt(breaks[length(breaks)]), "k")
    )
    return(labels)
}

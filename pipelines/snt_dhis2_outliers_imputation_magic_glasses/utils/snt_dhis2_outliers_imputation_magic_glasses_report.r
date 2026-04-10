# Report helpers for magic glasses outliers imputation pipeline.
.this_file <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
.candidate_files <- unique(c(
    if (exists("PIPELINE_PATH", inherits = TRUE)) {
        file.path(get("PIPELINE_PATH", inherits = TRUE), "utils", "snt_dhis2_outliers_imputation_magic_glasses.r")
    } else {
        character(0)
    },
    if (!is.na(.this_file)) {
        file.path(dirname(.this_file), "snt_dhis2_outliers_imputation_magic_glasses.r")
    } else {
        character(0)
    },
    file.path(getwd(), "snt_dhis2_outliers_imputation_magic_glasses.r")
))
.target_file <- .candidate_files[file.exists(.candidate_files)][1]
if (is.na(.target_file)) {
    stop(paste0(
        "Could not locate snt_dhis2_outliers_imputation_magic_glasses.r. Tried: ",
        paste(.candidate_files, collapse = " | ")
    ))
}
source(.target_file)


# Shared bootstrap for Magic Glasses notebooks.
bootstrap_magic_glasses_context <- function(
    root_path = "~/workspace",
    required_packages = c("arrow", "data.table", "jsonlite", "reticulate", "glue"),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(unique(required_packages))

    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }

    return(list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        openhexa = openhexa
    ))
}

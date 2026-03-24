# Shared bootstrap for the IQR outliers pipeline notebooks.
bootstrap_iqr_context <- function(
    root_path = "~/workspace",
    required_packages = c(
        "data.table", "arrow", "tidyverse", "jsonlite", "DBI", "RPostgres",
        "reticulate", "glue", "zoo"
    ),
    load_openhexa = TRUE
) {
    code_path <- file.path(root_path, "code")
    config_path <- file.path(root_path, "configuration")
    data_path <- file.path(root_path, "data")

    source(file.path(code_path, "snt_utils.r"))
    install_and_load(required_packages)

    Sys.setenv(PROJ_LIB = "/opt/conda/share/proj")
    Sys.setenv(GDAL_DATA = "/opt/conda/share/gdal")
    Sys.setenv(RETICULATE_PYTHON = "/opt/conda/bin/python")

    openhexa <- NULL
    if (load_openhexa) {
        openhexa <- reticulate::import("openhexa.sdk")
    }

    config_json <- tryCatch(
        {
            jsonlite::fromJSON(file.path(config_path, "SNT_config.json"))
        },
        error = function(e) {
            msg <- glue::glue("[ERROR] Error while loading configuration {conditionMessage(e)}")
            log_msg(msg)
            stop(msg)
        }
    )

    return(list(
        ROOT_PATH = root_path,
        CODE_PATH = code_path,
        CONFIG_PATH = config_path,
        DATA_PATH = data_path,
        openhexa = openhexa,
        config_json = config_json
    ))
}

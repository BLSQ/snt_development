# Utils entrypoint for snt_dhs_prevalence_computation.ipynb
# Expects snt_utils.r to be loaded and PIPELINE_PATH to be set by the notebook.
# Loads the DHS indicator helpers (snt_dhs_indicator_tables.r); prevalence-specific helpers go in this file.
source(file.path(PIPELINE_PATH, "utils", "snt_dhs_indicator_tables.r"))

SNT Quality of Care Pipeline

Description

This pipeline computes district-year quality-of-care indicators from DHIS2 outliers-imputed routine data and generates yearly ADM2 maps.

Parameters

  outlier_imputation_method (String, required)
    Name: Outlier imputation method
    Description: Select which outlier detection/imputation method to use.
    Choices/Default: mean, median, iqr, trend. Default: mean.

  data_action (String, required)
    Name: Data action
    Description: Choose whether to use imputed data (outliers replaced) or removed data (outliers removed).
    Choices/Default: imputed, removed. Default: imputed.

  run_report_only (Boolean, optional)
    Name: Run reporting only
    Description: Skip computations and run only reporting notebook.
    Choices/Default: TRUE/FALSE. Default: FALSE.

  pull_scripts (Boolean, optional)
    Name: Pull scripts
    Description: Pull latest scripts from repository before run.
    Choices/Default: TRUE/FALSE. Default: FALSE.

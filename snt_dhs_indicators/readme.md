# SNT DHS Indicators Pipeline

The **SNT DHS Indicators** pipeline runs five indicator families—**bed nets**, **care seeking**, **mortality**, **malaria prevalence (RDT)**, and **vaccination (DTP)**—from DHS microdata prepared in the computation notebooks. Each family produces **ADM1**-level parquet and CSV tables, uploads them to the **`DHS_INDICATORS`** dataset, and refreshes its paired reporting notebook.

## Parameters

None. `pipeline.py` only registers standard orchestration flags, which are omitted from this document; there are no additional domain parameters passed into the computation notebooks.

## Functionality Overview

1. **Configuration:** Load **`SNT_config.json`**, validate, and read **`COUNTRY_CODE`**.
2. **Bed nets:** Run **`snt_dhs_bednets_computation.ipynb`** then **`snt_dhs_bednets_report.ipynb`** (unless reporting-only mode is enabled at the platform level).
3. **Care seeking:** Run **`snt_dhs_careseeking_computation.ipynb`** then **`snt_dhs_careseeking_report.ipynb`**.
4. **Mortality:** Run **`snt_dhs_mortality_computation.ipynb`** then **`snt_dhs_mortality_report.ipynb`**.
5. **Prevalence:** Run **`snt_dhs_prevalence_computation.ipynb`** then **`snt_dhs_prevalence_report.ipynb`**.
6. **Vaccination:** Run **`snt_dhs_vaccination_computation.ipynb`** then **`snt_dhs_vaccination_report.ipynb`**.
7. **Publish:** When computation runs, save pipeline parameters JSON and call **`add_files_to_dataset`** with all listed indicator parquet/CSV paths plus the parameter file on **`DHS_INDICATORS`**.

## Inputs

* **DHS survey inputs and lookup tables** as required by each computation notebook under **`pipelines/snt_dhs_indicators/code/`** (see the respective **`.ipynb`** for exact file names and **`SNT_config.json`** keys).
* **`configuration/SNT_config.json`** for **`COUNTRY_CODE`** and **`SNT_DATASET_IDENTIFIERS`**.

## Outputs

All paths are under **`data/dhs/indicators/`** with **`{COUNTRY_CODE}_DHS_ADM1_`** prefixes and matching **`.parquet`** / **`.csv`** pairs:

* **`bednets/`**: **`PCT_ITN_ACCESS`**, **`PCT_ITN_USE`**
* **`careseeking/`**: **`PCT_CARESEEKING_SAMPLE_AVERAGE`**, **`PCT_NO_CARE`**, **`PCT_PUBLIC_CARE`**, **`PCT_PRIVATE_CARE`**
* **`mortality/`**: **`U5MR_PERMIL`**
* **`prevalence/`**: **`PCT_U5_PREV_RDT`**
* **`vaccination/`**: **`PCT_DTP1`**, **`PCT_DTP2`**, **`PCT_DTP3`**, **`PCT_DROPOUT_DTP`**
* **Pipeline parameters JSON** (saved next to outputs when computation runs)
* **Published copies** on the **`DHS_INDICATORS`** OpenHEXA dataset
* **Reporting outputs** under each **`reporting/outputs/`** folder

> **Notes for the Data Analyst:**
>
> - **`Administrative level`**: File names encode **`DHS`** as the data source and **`ADM1`** as the spatial level (`pipeline.py` sets **`admin_level = "ADM1"`**).
> - **`Indicator definitions`**: Numerator, denominator, and survey weight handling are defined inside each computation notebook; treat those notebooks as the source of truth for column semantics.
> - **`Dataset identifier`**: Upload uses **`SNT_DATASET_IDENTIFIERS["DHS_INDICATORS"]`**; a missing config entry would prevent a valid dataset upload.

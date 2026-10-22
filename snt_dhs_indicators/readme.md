# SNT DHS Indicators Pipeline

The **SNT DHS Indicators** pipeline runs five indicator families—**bed nets**, **care seeking**, **mortality**, **malaria prevalence (RDT)**, and **vaccination (DTP)**—from DHS microdata prepared in the computation notebooks. Each family estimates its indicators at **ADM1**, writes them as **ADM2**-level parquet and CSV tables (each ADM2 repeats its ADM1 value) and refreshes its paired reporting notebook; all tables are then published together to the **`DHS_INDICATORS`** dataset.

## Parameters

None. `pipeline.py` only registers standard orchestration flags, which are omitted from this document; there are no additional domain parameters passed into the computation notebooks.

## Functionality Overview

1. **Configuration:** Load **`SNT_config.json`**, validate, and read **`COUNTRY_CODE`**.
2. **Bed nets:** Run **`snt_dhs_bednets_computation.ipynb`**, check that the **`bednets/`** output files were written during this run, then run **`snt_dhs_bednets_report.ipynb`**.
3. **Care seeking:** Run **`snt_dhs_careseeking_computation.ipynb`**, check that the **`careseeking/`** output files were written during this run, then run **`snt_dhs_careseeking_report.ipynb`**.
4. **Mortality:** Run **`snt_dhs_mortality_computation.ipynb`**, check that the **`mortality/`** output files were written during this run, then run **`snt_dhs_mortality_report.ipynb`**.
5. **Prevalence:** Run **`snt_dhs_prevalence_computation.ipynb`**, check that the **`prevalence/`** output files were written during this run, then run **`snt_dhs_prevalence_report.ipynb`**.
6. **Vaccination:** Run **`snt_dhs_vaccination_computation.ipynb`**, check that the **`vaccination/`** output files were written during this run, then run **`snt_dhs_vaccination_report.ipynb`**.
7. **Publish:** Save the pipeline parameters JSON and call **`add_files_to_dataset`** with all indicator parquet/CSV paths plus the parameters file on **`DHS_INDICATORS`**, creating one new dataset version.

In report-only mode, steps 2–6 run only the report notebooks on the existing outputs (no computation, no output check), and step 7 is skipped: no parameters file is written and no dataset version is created.

## Inputs

* **DHS recode zips** in **`data/dhs/raw/`** (latest SPSS **`SV`** issue): **`HR`** for every indicator, plus **`PR`** (bed nets, prevalence), **`KR`** (care seeking, vaccination) or **`BR`** (mortality). A notebook stops if its two recodes are not from the same survey version.
* **`[COUNTRY_CODE]_shapes.geojson`** from **`DHIS2_DATASET_FORMATTED`** (written by **`snt_dhis2_formatting`**): the ADM1/ADM2 pyramid. A notebook stops if the DHS region names do not match the DHIS2 **`ADM1_NAME`** values exactly.
* **`configuration/SNT_config.json`**: **`SNT_CONFIG.COUNTRY_CODE`**, and **`DHIS2_DATASET_FORMATTED`** and **`DHS_INDICATORS`** under **`SNT_DATASET_IDENTIFIERS`**.

## Outputs

All paths are under **`data/dhs/indicators/`** with **`{COUNTRY_CODE}_DHS_ADM2_`** prefixes and matching **`.parquet`** / **`.csv`** pairs. Every table has one row per ADM2 with **`ADM1_ID`**, **`ADM1_NAME`**, **`ADM2_ID`**, **`ADM2_NAME`** and **`YEAR`** (the survey year: latest **`HV007`** in the **`HR`** recode); the indicator values are the ADM1 estimates, repeated for each ADM2 of the province.

* **`bednets/`**: **`PCT_ITN_ACCESS`**, **`PCT_ITN_USE`**
* **`careseeking/`**: **`PCT_CARESEEKING_SAMPLE_AVERAGE`**, **`PCT_NO_CARE`**, **`PCT_PUBLIC_CARE`**, **`PCT_PRIVATE_CARE`**
* **`mortality/`**: **`U5MR_PERMIL`**
* **`prevalence/`**: **`PCT_U5_PREV_RDT`**
* **`vaccination/`**: **`PCT_DTP1`**, **`PCT_DTP2`**, **`PCT_DTP3`**, **`PCT_DROPOUT_DTP`**
* **Pipeline parameters JSON** (saved next to outputs when computation runs)
* **Published copies** on the **`DHS_INDICATORS`** OpenHEXA dataset
* **Reports** in **`pipelines/snt_dhs_indicators/reporting/outputs/`** (executed report notebooks and their HTML), with the map and confidence-interval PNGs in its **`figures/`** subfolder. The reports plot at ADM1.

> **Notes for the Data Analyst:**
>
> - **`Administrative level`**: File names encode **`DHS`** as the data source and **`ADM2`** as the level of the tables (`pipeline.py` sets **`admin_level = "ADM2"`**). The survey estimates are computed at **ADM1** (DHS sampling level), so all ADM2 units of a province share the same value.> - **`Indicator definitions`**: Numerator, denominator, and survey weight handling are defined inside each computation notebook; treat those notebooks as the source of truth for column semantics.
> - **`Dataset identifier`**: Upload uses **`SNT_DATASET_IDENTIFIERS["DHS_INDICATORS"]`**; a missing config entry would prevent a valid dataset upload.
> - **`Output check`**: If a computation notebook does not write all of its expected files during the run, the pipeline stops with an error naming the missing files, and nothing is published.

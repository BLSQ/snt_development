# SNT Assemble Results Pipeline

The **SNT Assemble Results** pipeline builds a single **ADM2**-level results table from **`SNT_metadata.json`** column definitions, enriches it by joining curated inputs from multiple OpenHEXA datasets via **`get_file_from_dataset`**, applies configurable **aggregations** for incidence and reporting rates, and publishes the consolidated outputs to **`SNT_RESULTS`**.

---

## Parameters

* **`incidence_metric`** (String, Required):
  * **Name:** Incidence aggregation across years
  * **Description:** **`mean`** or **`median`** applied per **`ADM2_ID`** across the selected incidence year window for each incidence column present in both the metadata schema and the incidence file.
  * **Default:** `mean`.
* **`incidence_years_to_include`** (Integer, Required):
  * **Name:** Incidence calculation period (years back)
  * **Description:** Must be **`0`** or **negative**. **`0`** uses **all** available years. **`-1`** restricts to the **latest year only**. **`-2`** or lower includes the **latest year and additional past years** (see `add_incidence_indicators_to` in **`pipeline.py`**). Values **`> 0`** are rejected.
  * **Default:** `0`.
* **`reporting_rate_metric`** (String, Required):
  * **Name:** Reporting rate aggregation across years
  * **Description:** **`mean`** or **`median`** of **`REPORTING_RATE`** over all periods per **`ADM2_ID`** before scaling to a **0–100** percentage in the results table.
  * **Default:** `mean`.
* **`map_selection`** (String, Multi, Required):
  * **Name:** MAP indicators selection
  * **Description:** Subset of MAP metrics to merge from **`[COUNTRY_CODE]_map_data.parquet`**. For each selected metric, the pipeline keeps **`STATISTIC == "MEAN"`**, takes the **latest `YEAR`**, maps to results columns (**`PF_PR_RATE`**, **`PF_MORTALITY_RATE`**, etc.), and applies indicator-specific unit scaling.
* **`adm1_layers_file`** (File, Optional):
  * **Name:** Additional ADM 1 layers (.csv)
  * **Description:** Optional user CSV joined on **`ADM1_ID`** to append extra columns.
* **`adm2_layers_file`** (File, Optional):
  * **Name:** Additional ADM 2 layers (.csv)
  * **Description:** Optional user CSV joined on **`ADM2_ID`** to append extra columns.

---

## Functionality Overview

1. **Base table:** **`build_results_table`** copies **`SNT_metadata.json`** into the pipeline **`data/`** folder, reads variable keys as empty columns, and fills **ADM1/ADM2** identifiers and names from **`[COUNTRY_CODE]_pyramid.parquet`** (**`get_file_from_dataset`** on **`DHIS2_DATASET_FORMATTED`**).
2. **DHIS2 block (`add_dhis2_indicators_to`):**
   * **`add_population_to`**: Prefers **`[COUNTRY_CODE]_population.parquet`** from **`DHIS2_POPULATION_TRANSFORMATION`**, else falls back to formatted population; picks a **reference year** from config or heuristics; merges **`POPULATION`** (and any extra matching columns).
   * **`add_reporting_rate_to`**: Resolves the latest **`[COUNTRY_CODE]_reporting_rate_*.parquet`** from **`DHIS2_REPORTING_RATE`**, aggregates with **`reporting_rate_metric`**, stores **`REPORTING_RATE`** as percent with one decimal.
   * **`add_incidence_indicators_to`**: Loads **`[COUNTRY_CODE]_incidence.parquet`** from **`DHIS2_INCIDENCE`**, filters years with **`incidence_years_to_include`**, aggregates **`INCIDENCE_*`** columns with **`incidence_metric`**, rounds merged values.
3. **MAP (`add_map_indicators_to`):** Loads **`[COUNTRY_CODE]_map_data.parquet`** from **`SNT_MAP_EXTRACTS`**, filters **`map_selection`**, latest year, mean statistic; applies scaling factors (e.g. parasite rate ×100, mortality ×100000).
4. **Seasonality (`add_seasonality_indicators_to`):** Merges **`[COUNTRY_CODE]_rainfall_seasonality.parquet`** (**`SNT_SEASONALITY_RAINFALL`**) and **`[COUNTRY_CODE]_cases_seasonality.parquet`** (**`SNT_SEASONALITY_CASES`**) when metadata columns exist.
5. **DHS (`add_dhs_indicators_to`):** Loads ADM1 parquet extracts from **`DHS_INDICATORS`** (care-seeking, dropout, vaccination proportions, mortality, prevalence, ITN metrics) via **`get_file_from_dataset`** / **`update_table_with`** where columns are declared in metadata.
6. **Healthcare access (`add_access_to_health_to`):** Merges **`[COUNTRY_CODE]_population_covered_health.parquet`** from **`SNT_HEALTHCARE_ACCESS`** for **`PCT_HEALTH_ACCESS`** when present in metadata.
7. **User layers (`add_user_uploaded_indicators_to`):** Left-joins optional **`adm1_layers_file`** and **`adm2_layers_file`**.
8. **Metadata table:** **`build_metadata_table`** emits companion **`[COUNTRY_CODE]_metadata.csv`** / **`.parquet`** describing variables and periods updated during the run.
9. **Publication:** **`add_files_to_dataset`** uploads results dataset (CSV/Parquet), metadata tables, and saved pipeline parameters to **`SNT_RESULTS`**.

---

## Inputs

* **`configuration/SNT_config.json`** and **`configuration/SNT_metadata.json`** (copied into `pipelines/snt_assemble_results/data/`).
* **Upstream dataset files** referenced by **`get_file_from_dataset`**, including but not limited to: formatted pyramid; transformed or formatted population; reporting rate parquet; incidence parquet; map extracts; seasonality outputs; DHS indicator extracts; healthcare access parquet—**only** for columns present in **`SNT_metadata.json`**.

---

## Outputs

* **`results/[COUNTRY_CODE]_results_dataset.parquet`** and **`.csv`**: One row per **ADM2** with all assembled indicators.
* **`results/[COUNTRY_CODE]_metadata.parquet`** and **`.csv`**: Variable-level metadata after **`update_metadata`** calls.
* **`results/`** pipeline parameters JSON.
* **SNT_RESULTS** dataset updated via **`add_files_to_dataset`** with the four artifacts above.

---

## Key aggregations (reference)

| Step | Operation |
|------|-----------|
| Reporting rate | Per **`ADM2_ID`**: **`mean`** or **`median`** of **`REPORTING_RATE`** over all periods → multiply by **100**, round to **0.1**. |
| Incidence | Filter **`YEAR`** to `[period_start, period_end]` from **`incidence_years_to_include`**; per **`ADM2_ID`**: **`mean`** or **`median`** of each **`INCIDENCE_*`** column present in both metadata and source; round to **2** decimals. |
| MAP | Latest **`YEAR`**, **`STATISTIC == "MEAN"`**; per-indicator scalar multipliers applied after merge. |

---

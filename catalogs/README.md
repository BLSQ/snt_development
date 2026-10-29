# `catalogs/` — universal reference files

A catalog is a file that is the same for every country and every workspace, kept in this repository
so it has one versioned source of truth. Pipelines download it from GitHub at run time, at a
release tag. It is **never** read from a workspace, so workspaces cannot drift from it.

`contracts/` holds the **shape** of a file (schema, skeleton). `catalogs/` holds **content** that
ships with a release.

| File | Role | Fetched by |
|---|---|---|
| [`SNT_metadata_master.json`](SNT_metadata_master.json) | **The master list**: every SNT Explorer data layer the current SNT pipelines can publish, for any country. Generic — it describes the pipeline collection, not a workspace. | [`snt_metadata_json_trim`](../snt_metadata_json_trim/readme.md) |

## Adding a catalog

- **No country data** (R1). A catalog is generic by definition.
- **Un-ignore it by exact name** in `.gitignore` (`!catalogs/<file>`). The repo ignores `*.json`
  wholesale; naming each file keeps a stray data file dropped here ignored.
- **It is hashed into `release_manifest.json`** (`repository_files`) only if it matches
  `catalogs/*.json` in [`generate_manifest.yaml`](../.github/workflows/generate_manifest.yaml). A
  non-JSON catalog needs that pattern extended (R22).
- **Add a row above** and a section below: what it is, who fetches it, how to maintain it. If it
  has a shape other tools depend on, its schema goes in [`contracts/`](../contracts/README.md).

## `SNT_metadata_master.json`

The **format** (fields, `TYPE` / `SCALE` rules, JSON Schema, validator) is specified in
[`contracts/snt_metadata_json/`](../contracts/snt_metadata_json/README.md).

### How it reaches a workspace

It is **not** deployed with any pipeline: `openhexa pipelines push` packages only
`.py .ipynb .txt .md .r .sql` files. Instead, the
[`snt_metadata_json_trim`](../snt_metadata_json_trim/readme.md) pipeline downloads it from this
repository at run time. The ref it downloads from is set by the `repository_ref` parameter:

- `latest` (default) — the latest published GitHub release;
- a release tag;
- a branch such as `main`.

The pipeline then keeps the layers whose source column exists in that workspace, and publishes the
result as `SNT_metadata_trimmed.json` to the `SNT_METADATA` dataset.

The file is part of each release like any other file in the repository. Its SHA256, recorded in the
trim report as `MASTER.SHA256`, can be checked against `release_manifest.json`.

**So a change here reaches workspaces only once it is in a release** — or when an operator sets
`repository_ref` to a branch on purpose.

### Maintaining it

- **The JSON is the source of truth.** Edit it by hand, in a PR. There is no generator. The
  one-off bootstrap script is in git history at commit `062cf0c`; do not resurrect it over curated
  content ([ADR 0008](../docs/decisions/0008-master-list-maintained-by-hand.md)).
- **Two kinds of field, two kinds of review:**
  - **Objective:** the layer id and `SOURCE_DATA` (dataset key, filename, column). Check them against
    the code that writes the column.
  - **Curated:** `LABEL`, `DESCRIPTION`, `SOURCE`, `UNITS`, `CATEGORY`, `TYPE`, `SCALE`,
    `UNIT_SYMBOL`. These need a person with domain knowledge. Text is a draft written from the code,
    not yet reviewed ([BACKLOG META-13](../docs/BACKLOG.md#meta-13)); `CATEGORY` is `""` in both
    languages. Most `SCALE` values are generic defaults per unit, not drawn from data.
- **House style for the text:**
  - `SOURCE` is `DHIS2`, `ERA5` or `WorldPop`, and `""` where the source depends on the run or mixes
    several: every `*_TRANSFORMED` layer (DHIS2 or user-provided input), `POPULATION_INCIDENCE`
    (operator-chosen population source), and `POP_COVERED` / `PCT_HEALTH_ACCESS` (WorldPop population
    plus DHIS2 or user-provided facilities). `LABEL` ends with `(SOURCE)` only when `SOURCE` is set.
  - `UNITS` says `Proportion` where the column is a 0–1 fraction (reporting rates, `*_PROPORTION`),
    with `UNIT_SYMBOL: null` ([BACKLOG META-2](../docs/BACKLOG.md#meta-2)).
- **When a pipeline adds or renames a published column**, add or update the master list entry in the
  same PR (CLAUDE.md handover checklist). A column with no entry is invisible in the Explorer. An
  entry whose column was renamed is dropped by the trim pipeline as `COLUMN_NOT_FOUND` in every
  workspace.
- **Validate before merging**: `python3 .claude/skills/add-explorer-layer/validate_metadata.py`
  (checks the master list by default) — see
  [`contracts/snt_metadata_json/README.md` §5](../contracts/snt_metadata_json/README.md#5-validating).

### What it contains

Every column the current pipelines can publish that fits the format — 85 layers. Each `SOURCE_DATA`
pointer was traced to the code that writes the column (the `add_files_to_dataset(...)` calls in
`pipeline.py`, and the notebook cells that build the final table); none was checked against a live
workspace. `snt_assemble_results` is out of scope (retired,
[ADR 0006](../docs/decisions/0006-retire-snt-assemble-results.md)).

The layer id is the column name, plus a qualifier where two layers would otherwise collide
([contract §2.2](../contracts/snt_metadata_json/README.md#22-the-layer-id-is-not-the-column-name)).

| Layers | Dataset key | File | Pipeline |
|---|---|---|---|
| `POPULATION`, `POP_*` (8) | `DHIS2_DATASET_FORMATTED` | `{COUNTRY_CODE}_population.csv` | `snt_dhis2_formatting` |
| `POPULATION_TRANSFORMED`, `POP_*_TRANSFORMED` (8) | `DHIS2_POPULATION_TRANSFORMATION` | `{COUNTRY_CODE}_population.csv` | `snt_dhis2_population_transformation` |
| `REPORTING_RATE_DATAELEMENT`, `REPORTING_RATE_DATASET` | `DHIS2_REPORTING_RATE` | `…_reporting_rate_dataelement_yearly.csv`, `…_reporting_rate_dataset_yearly.csv` | `snt_dhis2_reporting_rate_*` |
| `POPULATION_INCIDENCE`, `INCIDENCE_*` (4) | `DHIS2_INCIDENCE` | `{COUNTRY_CODE}_incidence.csv` | `snt_dhis2_incidence` |
| `<COUNT>_QOC_IMPUTED`, `<COUNT>_QOC_REMOVED` (10 + 10) | `DHIS2_QUALITY_OF_CARE` | `…_quality_of_care_district_year_{imputed,removed}.csv` | `snt_dhis2_quality_of_care` |
| `SEASONALITY_*`, `SEASONAL_BLOCK_*`, `*_PROPORTION` (4 + 4) | `SNT_SEASONALITY_{CASES,RAINFALL}` | `…_{cases,rainfall}_seasonality.csv` | `snt_seasonality_*` |
| `PRECIPITATION_{MEAN,MIN,MAX}` | `ERA5_DATASET_CLIMATE` | `{COUNTRY_CODE}_total_precipitation_monthly.parquet` | `snt_era5_climate_data` |
| `POPULATION_WORLDPOP` | `WORLDPOP_DATASET_EXTRACT` | `{COUNTRY_CODE}_worldpop_population.csv` | `snt_worldpop_extract` |
| `POP_TOTAL`, `POP_COVERED`, `PCT_HEALTH_ACCESS` | `SNT_HEALTHCARE_ACCESS` | `{COUNTRY_CODE}_population_covered_health.csv` | `snt_healthcare_access` |
| `*_SAMPLE_AVERAGE` (10), `PCT_DROPOUT_DTP_*` (3) | `DHS_INDICATORS` | `{COUNTRY_CODE}_DHS_ADM2_<INDICATOR>.csv`, one file per indicator | `snt_dhs_indicators` |
| `PF_*`, `INSECTICIDE_TREATED_NET_*`, `IRS_COVERAGE`, `ANTIMALARIAL_EFFECTIVE_TREATMENT`, each also as `*_POP_W` (7 + 7) | `SNT_MAP_EXTRACTS` | `{COUNTRY_CODE}_map_data.csv` | `snt_map_extracts` |

#### Layers that will not load everywhere

"Every column that *can* be published" includes columns that only some countries or parameter
choices produce. Expect these layers to fail to load (the trim pipeline drops them) where the column
is absent:

- **`POP_*` in the formatted population** exist only where `POPULATION_INDICATOR_DEFINITIONS` gives
  that key DHIS2 ids. In the archived configs that is CMR (`UNDER_5`, `PREGNANT_WOMEN`, `0_1_Y`,
  `1_2_Y`, `50_PLUS`) and NER (`UNDER_5`, `PREGNANT_WOMEN`). `POP_5_10_Y` and `POP_5_36_M` have no ids
  in any of them.
- **`POP_*_TRANSFORMED`** exist only when the operator sets the matching disaggregation parameter,
  or supplies a disaggregation file.
- **The quality-of-care counts** are written only if the column is in the routine data, so they vary
  with each country's `DHIS2_INDICATOR_DEFINITIONS` (`ALLADM`, `ALLDTH`, `ALLOUT` in particular).
- **Only one variant of an alternative is present at a time.** Quality of care writes the `imputed`
  or the `removed` file, per run. Both reporting-rate files can coexist, but see the dataset-version
  caveat below.
- **`INCIDENCE_ADJ_CARESEEKING`** is always written, but is all `NA` when no care-seeking data was
  available.

#### Tables that break other assumptions of the format

- **No `YEAR` column:** both seasonality tables and the healthcare-access table
  ([BACKLOG META-9](../docs/BACKLOG.md#meta-9)). Healthcare access is computed for a single
  WorldPop year, which is not stored in the table.
- **Monthly grain** (`YEAR` + `MONTH`): ERA5 ([BACKLOG META-7](../docs/BACKLOG.md#meta-7)). Both
  reporting-rate layers read the yearly tables (`YEAR`, no `MONTH`).
- **`YEAR` stored as text:** the WorldPop table.
- **Parquet only:** ERA5 publishes no `.csv` twin, so its `FILENAME` is the `.parquet`.
- **ADM1 values on ADM2 rows, one survey year:** the DHS tables have one row per ADM2, but each
  value is the ADM1 survey estimate, repeated for every ADM2 within that ADM1. `YEAR` is the survey
  year, so a DHS layer has a single year. The layer descriptions say so.
- **Dataset versions (UNVERIFIED):** if `add_files_to_dataset` creates a version holding only the
  files of that run, `"latest"` on `DHIS2_REPORTING_RATE` holds only the variant run last, and the
  same goes for the two quality-of-care files.

#### What was left out, and why

| Left out | Pipeline | Why |
|---|---|---|
| MAP `LCI` / `UCI` statistics (`{CC}_map_data_{year}.*`) | `snt_map_extracts` | Confidence bounds, kept only in the unpublished per-year long tables. |
| `{CC}_DHS_ADM2_PCT_CARESEEKING_SAMPLE_AVERAGE.*` | `snt_dhs_indicators` | Copies of the `PCT_{NO,PUBLIC,PRIVATE}_CARE` sample averages, which are included from their own files. |
| DHS `*_CI_LOWER_BOUND`, `*_CI_UPPER_BOUND` | `snt_dhs_indicators` | Confidence bounds of an estimate, not a layer of their own. |
| The five quality-of-care rates: `testing_rate`, `treatment_rate`, `case_fatality_rate`, `prop_adm_malaria`, `prop_malaria_deaths` | `snt_dhis2_quality_of_care` | Written in **lowercase**, which `COLUMN`'s pattern rejects and which breaks repo rule R10 ([BACKLOG META-16](../docs/BACKLOG.md#meta-16)). |
| `non_malaria_all_cause_outpatients`, `presumed_cases` | `snt_dhis2_quality_of_care` | Lowercase, as above, and copies of `ALLOUT` / `PRES`, which are included. |
| Routine indicators (`{CC}_routine.*`) | `snt_dhis2_formatting` | **Facility × month** grain, not ADM2. The Explorer would need aggregation the format cannot express. |
| `{CC}_routine_outliers_imputed.parquet`, `…_removed.parquet` | the five `snt_dhis2_outliers_imputation_*` | Facility × month grain, as above. |
| `{CC}_routine_outliers_detected.parquet` | the five `snt_dhis2_outliers_imputation_*` | Long format (`INDICATOR` / `VALUE`), facility grain. |
| `{CC}_reporting.*` | `snt_dhis2_formatting` | Long format (`PRODUCT_METRIC` / `VALUE`). |
| `{CC}_pyramid.*` | `snt_dhis2_formatting` | Facility list (coordinates, opening/closing dates); no values to map. |
| All raw extracts (`{CC}_dhis2_raw_*.parquet`) | `snt_dhis2_extract` | Long format, raw DHIS2 ids, no `YEAR`. |
| Extra columns from an uploaded disaggregation file | `snt_dhis2_population_transformation` | Arbitrary operator-chosen names, not uppercased. Only the template's `POP_*` columns are included. |

`snt_workspace_check` publishes nothing to a dataset.

### Known open points

- **Country tuning has no home yet**: a trim run overwrites hand-tuned scales or labels
  ([BACKLOG META-12](../docs/BACKLOG.md#meta-12)).
- **Checking the master list against the pipelines is manual**
  ([BACKLOG META-11](../docs/BACKLOG.md#meta-11)).

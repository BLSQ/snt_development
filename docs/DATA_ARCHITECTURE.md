# SNT Data Architecture

Reference for the data lineage, storage layout and orchestration patterns of the SNT
(Subnational Tailoring) pipelines maintained in this repository.

Companion document: [`CLAUDE.md`](../CLAUDE.md) — conventions, guardrails and working rules.

> **Status of this document.** Written from a code audit of `main` (2026-08-26/27). Sections
> marked **`[TODO: Giulia]`** need input that cannot be derived from the code.

---

## 0. Audit scope — what is verified here, and what is not

Read this before trusting any claim below. This document was produced by reading code, not by
running it, and the coverage is uneven on purpose.

**Verified by reading the code** (2026-08-27, branch `SNT25-640_codebase-documentation`):

- All 20 `pipeline.py` files, in full — parameters, guards, and every `add_files_to_dataset(...)`
  call. Published-output claims in §3 come from those calls, not from the pipelines' `readme.md`.
- The `utils/*.r` helpers behind filename resolution and column selection, and the notebook cells
  that resolve dataset ids (`config_json$SNT_DATASET_IDENTIFIERS$…`).
- All 20 `.github/workflows/push_snt_*.yaml`, `pyproject.toml`, `.gitignore`, `CODEOWNERS`, and
  the five `configuration/SNT_config_<CC>.json` reference copies.
- The OpenHEXA workspace image `blsq/openhexa-blsq-r-environment:latest`, read from its registry
  manifest (see [§7.4](#74-the-workspace-runtime-image)).

**Read selectively, not exhaustively:** the analytics notebooks. They were opened where a specific
question needed answering (parameter fallbacks, grouping keys, dataset ids, the population-column
defect). Nobody has line-by-line reviewed the statistics in them, and this document does not claim
the methods are correct — only what they consume and produce.

**Deliberately out of scope:**

| Excluded | Why |
|---|---|
| `configuration/SNT_metadata.json` | mid-change; excluded on Giulia's instruction (see §3.2 Stage E) |
| Deep audit of `snt_assemble_results` | being deprecated; read for lineage only, not reviewed |
| `snt_dhis2_outliers_detection/` | discontinued; local-only leftover, not on the remote |
| `deprecated/` | history, never a template |
| `snt_lib` (`BLSQ/snt_utils`) internals | external repo, not in this checkout |

**Not verified at all:** nothing here was executed. No pipeline was run, no workspace was
inspected, no output file was opened. Runtime behaviour — actual data volumes, real DHIS2
responses, whether a given country's config is valid — is unverified. Where the code makes a
behaviour conditional on data that only exists in a workspace, this document says so rather than
guessing.

**A future agent picking this up should:** trust §3's published-output lists (checked against the
code), treat the statistical descriptions as summaries rather than specifications, and re-verify
anything in a pipeline whose `pipeline.py` has changed since the commit above.

---

## 1. What this system is

A collection of ~20 independent [OpenHEXA](https://openhexa.org) pipelines that turn routine
health-system data (DHIS2) plus external geospatial/survey sources into a single
**one-row-per-ADM2 results table** used for subnational tailoring of malaria interventions.

There is **no central scheduler and no DAG engine**. Each pipeline is a standalone OpenHEXA
pipeline, launched **manually from the OpenHEXA UI**. Ordering is a convention, not an
enforced dependency graph — see [§4](#4-orchestration-model).

**Key architectural consequence:** the coupling between pipelines is *data coupling through
OpenHEXA datasets*, identified by logical names in `SNT_config.json`. A downstream pipeline
does not know which upstream pipeline produced its input — it only knows a dataset id and a
filename. This is what makes the "user can supply their own input" and "alternative pipelines
override each other" behaviours possible.

### 1.1 The 20 pipelines at a glance

Orientation table — one row per pipeline, for answering "which one do I even open?". Dataset ids
are the logical names in `SNT_CONFIG.SNT_DATASET_IDENTIFIERS`; the detail is in
[§3.2](#32-stages). **Engine** is `py` (Python only, lintable locally) or `nb` (drives R
notebooks — see [Rule 2 in `CLAUDE.md`](../CLAUDE.md)).

| Pipeline | Stage | Does | Reads | Publishes to | Engine |
|---|---|---|---|---|---|
| `snt_dhis2_extract` | A | Downloads raw analytics, population, pyramid, shapes and reporting rates from the DHIS2 API | DHIS2 API | `DHIS2_DATASET_EXTRACTS` | `py` (+1 NER-only notebook) |
| `snt_dhis2_formatting` | B | Reshapes all five raw extracts into the SNT schema; **the hinge of the whole system** | `DHIS2_DATASET_EXTRACTS` | `DHIS2_DATASET_FORMATTED` | `nb` ×5 |
| `snt_dhis2_outliers_imputation_iqr` | C | Outlier detection + imputation, IQR method | `DHIS2_DATASET_FORMATTED` | `DHIS2_OUTLIERS_IMPUTATION` | `nb` |
| `snt_dhis2_outliers_imputation_mean` | C | Same, mean method | `DHIS2_DATASET_FORMATTED` | `DHIS2_OUTLIERS_IMPUTATION` | `nb` |
| `snt_dhis2_outliers_imputation_median` | C | Same, median method | `DHIS2_DATASET_FORMATTED` | `DHIS2_OUTLIERS_IMPUTATION` | `nb` |
| `snt_dhis2_outliers_imputation_path` | C | Same, PATH method | `DHIS2_DATASET_FORMATTED` | `DHIS2_OUTLIERS_IMPUTATION` | `nb` |
| `snt_dhis2_outliers_imputation_magic_glasses` | C | Same, Magic Glasses method | `DHIS2_DATASET_FORMATTED` | `DHIS2_OUTLIERS_IMPUTATION` | `nb` |
| `snt_dhis2_population_transformation` | C | Rescales/projects population; optional disaggregation upload | `DHIS2_DATASET_FORMATTED` | `DHIS2_POPULATION_TRANSFORMATION` | `nb` |
| `snt_dhis2_reporting_rate_dataelement` | D | Reporting rates computed from data elements | `DHIS2_DATASET_FORMATTED`, `DHIS2_OUTLIERS_IMPUTATION` | `DHIS2_REPORTING_RATE` | `nb` |
| `snt_dhis2_reporting_rate_dataset` | D | Reporting rates taken from DHIS2 dataset metrics | `DHIS2_DATASET_FORMATTED`, `DHIS2_OUTLIERS_IMPUTATION` | `DHIS2_REPORTING_RATE` | `nb` |
| `snt_dhis2_incidence` | D | Malaria incidence, optionally adjusted for reporting and care-seeking | `DHIS2_DATASET_FORMATTED`, `DHIS2_OUTLIERS_IMPUTATION`, `DHIS2_POPULATION_TRANSFORMATION`, `DHS_INDICATORS` | `DHIS2_INCIDENCE` | `nb` |
| `snt_dhis2_quality_of_care` | D | Care-quality indicators | `DHIS2_DATASET_FORMATTED`, `DHIS2_OUTLIERS_IMPUTATION` | `DHIS2_QUALITY_OF_CARE` | `nb` |
| `snt_seasonality_cases` | D | Seasonality of malaria cases | `DHIS2_DATASET_FORMATTED` | `SNT_SEASONALITY_CASES` | `nb` |
| `snt_seasonality_rainfall` | D | Seasonality of rainfall | `DHIS2_DATASET_FORMATTED`, `ERA5_DATASET_CLIMATE` | `SNT_SEASONALITY_RAINFALL` | `nb` |
| `snt_era5_climate_data` | A′ | Copernicus ERA5 precipitation, zonal-aggregated to ADM2 | `DHIS2_DATASET_FORMATTED` (shapes), Copernicus CDS | `ERA5_DATASET_CLIMATE` | `py` |
| `snt_worldpop_extract` | A′ | WorldPop rasters → ADM2 population | `DHIS2_DATASET_FORMATTED` (shapes), WorldPop | `WORLDPOP_DATASET_EXTRACT` | `py` |
| `snt_map_extracts` | A′ | Malaria Atlas Project layers (PfPR, ITN, IRS, …) | `DHIS2_DATASET_FORMATTED` (shapes), MAP WCS | `SNT_MAP_EXTRACTS` | `py` |
| `snt_healthcare_access` | A′ | Travel-time / access to health facilities | `DHIS2_DATASET_FORMATTED` (shapes), WorldPop raster cache, operator FOSA upload | `SNT_HEALTHCARE_ACCESS` | `nb` |
| `snt_dhs_indicators` | A′ | DHS survey indicators at **ADM1** | `DHIS2_DATASET_FORMATTED`, DHS recode files | `DHS_INDICATORS` | `nb` |
| `snt_assemble_results` | E | Flattens everything into the one-row-per-ADM2 results table — **⚠️ being deprecated** | most of the above | `SNT_RESULTS` | `py` |

Notes that the table cannot carry:

- **Stage A′** pipelines look like independent roots but are not — each fetches
  `{CC}_shapes.geojson` from `DHIS2_DATASET_FORMATTED` first, so Stage B must have run.
- **The five Stage C outlier pipelines are alternatives, not a sequence.** They write identical
  filenames and the last run wins, by design — see [§3.2](#32-stages).
- **The two Stage D reporting-rate pipelines are likewise alternatives**, both writing to
  `DHIS2_REPORTING_RATE`.
- **Run order within a stage is not enforced** and the authoritative operator-facing order is
  still [`[TODO: Giulia]` (§4.2)](#42-todo-giulia--authoritative-order--dependency-map).

---

## 2. Runtime topology

```
┌─────────────────────────────┐        ┌──────────────────────────────────────────┐
│  GitHub  BLSQ/snt_development│        │  OpenHEXA workspace  (one per country)   │
│                             │        │                                          │
│  <pipeline>/pipeline.py     │──CI──▶ │  deployed pipeline code                  │
│  <pipeline>/requirements.txt│  push  │                                          │
│                             │        │  ~/workspace/                            │
│  pipelines/<pipeline>/      │        │    configuration/SNT_config.json         │
│    code/*.ipynb   (R)       │──────▶ │    pipelines/<pipeline>/{code,reporting, │
│    reporting/*.ipynb (R)    │ pull_  │                          utils}/         │
│    utils/*.r                │ scripts│    code/snt_utils.r, snt_report.r, …     │
│  code/*.r  (shared R lib)   │ at run │    data/…                                │
└─────────────────────────────┘  time  │    results/…                             │
                                       │  OpenHEXA datasets  (versioned)          │
                                       │  Workspace DB  (table: outliers_detected)│
                                       └──────────────────────────────────────────┘
```

### 2.1 Two distinct delivery channels

| Channel | Carries | Trigger | Mechanism |
|---|---|---|---|
| **CI push → template** | `pipeline.py`, `requirements.txt` (Python only) | push to `main` touching those paths | `.github/workflows/push_<pipeline>.yaml` → `blsq/openhexa-cli-action@v1` → `openhexa pipelines push <dir>` into `snt-development`, which publishes a **new version of the SNT template pipeline**; subscribed country workspaces update automatically (§2.2) |
| **Runtime pull** | `pipelines/<name>/code/*.ipynb`, `reporting/*.ipynb`, `utils/*.r` (all R) | operator runs the pipeline with **`Pull scripts` = ON** in the OpenHEXA UI | `pull_scripts_from_repository()` from `snt_lib`, reading this repo — no template involvement, no automation (§2.2.1) |

**This asymmetry is the single most important operational fact in the system**, and it exists
because OpenHEXA supports Python pipelines but not R (§2.2.1). Merging a notebook change to
`main` changes *nothing* in any workspace until somebody runs that pipeline with `Pull scripts`
toggled on. The Python half, by contrast, can reach every country workspace automatically through
the template mechanism — so the two halves of one pipeline drift apart by default.

Corollary: the CI path filters only watch `pipeline.py` / `requirements.txt` / `readme.txt`, so
a notebook-only PR produces **no CI run at all** — absence of a green check is expected, not a
failure.

### 2.2 How a pipeline version reaches a country workspace — the template mechanism

All 20 workflows push to the **same** workspace, `snt-development`, and that is not incidental:
it is the mechanism by which updates reach every country.

OpenHEXA supports **template pipelines**: a pipeline that normally lives in one workspace can be
published as a template, which makes it installable in *any* OpenHEXA workspace. Each country
workspace installs the SNT pipelines from that template list, and can opt in to being updated
automatically whenever the source template publishes a new version. That opt-in is how validated
changes propagate across all countries without touching each workspace by hand.

```
  this repo ──push (CI)──▶  snt-development ws  ──▶  SNT template pipelines
                             (the reference ws)              │
                                                             │ install / auto-update
                            ┌────────────────┬───────────────┼────────────────┐
                            ▼                ▼               ▼                ▼
                        COD ws           BFA ws          NER ws           … ws
```

**Why the workspace must be `snt-development`.** Publishing a template version is tied to the
workspace the pipeline is pushed from. Pushing from `snt-development` publishes a **new version
of the existing SNT template**, which flows to every country workspace subscribed to it. Pushing
the same pipeline from *any other* workspace instead creates a **separate, new template
pipeline** — a duplicate that no country workspace is subscribed to, and that silently competes
with the real one in the template list.

> **Rule:** never change `workspace:` in a `push_snt_*.yaml`, and never `openhexa pipelines push`
> an SNT pipeline from a country workspace or a personal one. `snt-development` is the single
> publication point by team convention.

What each workflow does, concretely (all 20 are identical apart from names):

```yaml
on:
  push:
    branches: [main]
    paths:                                   # ← Python side only
      - "<pipeline>/pipeline.py"
      - "<pipeline>/requirements.txt"
      - ".github/workflows/push_<pipeline>.yaml"
jobs:
  deploy:
    - actions/checkout@v4
    - actions/setup-python@v5                # 3.11, pip cache on requirements.txt
    - blsq/openhexa-cli-action@v1            # workspace: "snt-development", token: secrets.OH_TOKEN
    - run: openhexa pipelines push <pipeline_dir>
             --code "<kebab-case-slug>"      # dir name, underscores → hyphens
             --description "<commit message>"
             --link "https://github.com/BLSQ/snt_development/commit/<sha>"
             --yes
```

`--description` and `--link` stamp each published version with the commit message and a link back
to the commit — so the OpenHEXA version list is a readable deployment history. Keep commit
messages meaningful for that reason. Verified 2026-08-26: all 20 workflows target
`snt-development`, use the same four flags, and every `--code` slug matches its directory name.

### 2.2.1 R is outside this mechanism — the core pain point

**OpenHEXA pipelines and templates cover the Python side only.** OpenHEXA was not built for R, so
none of the analytics — which is where essentially all the business logic lives — can travel
through the template system.

That asymmetry is the reason `pull_scripts` exists. The R notebooks and `.r` helpers are fetched
from this repository *at run time*, by a parameter an operator has to remember to toggle, rather
than being versioned and propagated with the pipeline they belong to. So a country workspace can
be running the newest `pipeline.py` (auto-updated via the template) against months-old R
analytics (never pulled) — with nothing anywhere reporting the mismatch.

This is a known, acknowledged pain point; solutions are being discussed with the OpenHEXA
developers. Until it changes, treat the two halves of every pipeline as **independently
versioned**, and see [`CLAUDE.md` rule 2](../CLAUDE.md#the-five-rules-that-matter-most).

A second, smaller consequence sits on the *authoring* side: because the R code lives in the
workspace at run time, the `utils/*.r` helpers a notebook `source()`s are read from the workspace
filesystem, not from a developer's clone. Editing a helper therefore means changing it in the
workspace and copying it back to git by hand. The working loop, and that gap, are described in
[`CLAUDE.md` → Editing R notebooks](../CLAUDE.md#editing-r-notebooks-the-vs-code-remote-kernel-loop).

### 2.3 Language split

| Pipeline | Python only | Executes R notebooks |
|---|---|---|
| `snt_dhis2_extract` | core extraction | reporting only |
| `snt_map_extracts`, `snt_worldpop_extract`, `snt_era5_climate_data` | core extraction | reporting only |
| `snt_assemble_results` | **fully Python, no notebooks at all** | — |
| everything else (13 pipelines) | thin orchestration shell | **yes — analytics live in `.ipynb` (kernel `ir`)** |

Python pipelines are the only ones with a workable local development story today
(see [`CLAUDE.md` §Local development](../CLAUDE.md#local-development-current-state)).

---

## 3. Data lineage

### 3.1 Sources (ingress)

| Source | Access | Pipeline | Notes |
|---|---|---|---|
| **DHIS2** | `DHIS2Connection` + `openhexa.toolbox.dhis2` | `snt_dhis2_extract` | analytics, population, org-unit pyramid, geometries, reporting rates |
| **Copernicus CDS (ERA5)** | `https://cds.climate.copernicus.eu/api` | `snt_era5_climate_data` | climate reanalysis; zarr repository + batched requests |
| **WorldPop** | `https://data.worldpop.org/GIS/Population` (`Global_2015_2030/R2025A`) | `snt_worldpop_extract`, `snt_map_extracts`, `snt_healthcare_access` | population rasters (`worldpopclient.py`, duplicated in 3 pipelines) |
| **Malaria Atlas Project** | `https://data.malariaatlas.org/geoserver` (WCS) | `snt_map_extracts` | `malariaAtlasProject/map.py` |
| **DHS** | recode files staged in the workspace | `snt_dhs_indicators` | `extract_latest_dhs_recode_filename()` in `code/snt_utils.r` |
| **Operator uploads** | OpenHEXA `File` parameter | `snt_dhis2_incidence` (care-seeking CSV), `snt_dhis2_population_transformation` (disaggregation CSV), `snt_healthcare_access` (FOSA locations CSV), `snt_assemble_results` (`add_layers_file`) | user-supplied override paths |

> **The external-source pipelines are not lineage roots.** `snt_era5_climate_data`,
> `snt_map_extracts`, `snt_worldpop_extract` and `snt_healthcare_access` each fetch
> `{CC}_shapes.geojson` from `DHIS2_DATASET_FORMATTED` before they can do anything — the ADM2
> geometries define the zones they aggregate into (and, for ERA5, the CDS request bounding box).
> **`snt_dhis2_formatting` must have run first**, even for the pipelines that touch no DHIS2 data.

**A shared raster cache sits outside the dataset contract.** `data/worldpop/rasters/` is written
by `snt_worldpop_extract` and read *and* written by `snt_map_extracts` and `snt_healthcare_access`,
all keyed on the filename pattern `{cc_lower}_pop_{year}_*.tif`. `snt_healthcare_access` even
carries a source comment explaining it copies MAP's lowercase-country convention to find the file.
This is the one place where pipelines couple through the **filesystem** rather than through a
dataset, and it is invisible to the lineage tables below: a raster downloaded by one pipeline is
silently reused by another. It is a cache, so the failure mode is a redundant download rather than
wrong data — but a stale or partial `.tif` would be picked up by all three.

### 3.2 Stages

**Stage A — Extract (raw landing)**

`snt_dhis2_extract` writes per-period Parquet under
`data/dhis2/extracts_raw/{routine,population,shapes,pyramid,reporting}_data/`, then
`merge_parquet_files()` concatenates each family into one file and **uppercases all column
names**. Outputs published to `DHIS2_DATASET_EXTRACTS`:

```
{CC}_dhis2_raw_analytics.parquet     {CC}_dhis2_raw_shapes.parquet
{CC}_dhis2_raw_population.parquet    {CC}_dhis2_raw_pyramid.parquet
{CC}_dhis2_raw_reporting.parquet     {CC}_parameters.json
```

Two country escape hatches are hardcoded in `snt_dhis2_extract/pipeline.py`:
- **BFA** — pyramid filtered to `level_4_name` starting with `"DS"` (mixed levels upstream).
- **NER** — org-unit groups fetched separately, and the pyramid is rewritten by an R notebook
  (`pipelines/snt_dhis2_extract/code/NER_pyramid_format.ipynb`) executed through papermill
  *inside the extraction step*.

Reporting rates are downloaded **either** as dataset-level metrics (`REPORTING_DATASETS`)
**or** as indicators (`REPORTING_INDICATORS`) — never both; datasets take precedence.

**Stage B — Format (analysis-ready)**

`snt_dhis2_formatting` runs five R notebooks, each gated on `dataset_file_exists()` for its raw
input, so a missing raw file silently skips that product. **Shapes must run first** — pyramid
coordinate validation uses the country geojson boundaries. Outputs → `DHIS2_DATASET_FORMATTED`:

```
{CC}_routine.parquet/.csv     {CC}_pyramid.parquet/.csv
{CC}_population.parquet/.csv  {CC}_reporting.parquet/.csv
{CC}_shapes.geojson
```

**Stage C — Quality: outlier detection & imputation (mutually exclusive variants)**

Five pipelines — `iqr`, `median`, `mean`, `path`, `magic_glasses` — all read
`{CC}_routine.parquet` from `DHIS2_DATASET_FORMATTED` and all write **the same filenames** to
**the same dataset** `DHIS2_OUTLIERS_IMPUTATION`:

```
{CC}_routine_outliers_detected.parquet
{CC}_routine_outliers_removed.parquet
{CC}_routine_outliers_imputed.parquet
```

> **Last run wins — this is the intended design, not a collision.** The analyst runs several
> methods on the same routine data, compares the reports, settles on one, and moves to the next
> stage; downstream pipelines consume whatever was produced last. Shared output names are the
> mechanism that makes the methods interchangeable — renaming outputs per method would break the
> override and force every downstream consumer to know which method it wants.
>
> The trade-off is that the file itself carries no method label. To recover which method produced
> a given file, read the `{CC}_parameters.json` published alongside it in the same dataset
> version, or the dataset version name. Never infer the method from the filename.

Each variant also pushes `{CC}_routine_outliers_detected.parquet` into the **workspace database
table `outliers_detected`** (`push_data_to_db_table`, parameter `push_db`, default `True`) — the
only relational sink in the system, and likewise overwritten by whichever variant ran last.

> ⚠️ **Needs attention.** No pipeline reads `outliers_detected`; its consumer is a Shiny app that
> is currently paused and may be replaced by a different tool. Unlike the dataset files, the table
> has no parameters JSON beside it, so once overwritten there is no record of which method or run
> produced its rows. Before anything depends on this table again, decide whether it needs a
> method/run-id discriminator (or append-with-run-id semantics instead of overwrite).

**Stage D — Derived indicators**

| Pipeline | Reads | Writes → dataset |
|---|---|---|
| `snt_dhis2_population_transformation` | `DHIS2_DATASET_FORMATTED` | `{CC}_population.parquet/.csv` → `DHIS2_POPULATION_TRANSFORMATION` |
| `snt_dhis2_reporting_rate_dataelement` | `DHIS2_DATASET_FORMATTED`, `DHIS2_OUTLIERS_IMPUTATION` | `{CC}_reporting_rate_dataelement.*` → `DHIS2_REPORTING_RATE` |
| `snt_dhis2_reporting_rate_dataset` | idem | `{CC}_reporting_rate_dataset.*` → `DHIS2_REPORTING_RATE` |
| `snt_dhis2_incidence` | routine per `routine_data_choice`; population per `use_transformed_population`; DHS or uploaded care-seeking | `{CC}_incidence.parquet/.csv` → `DHIS2_INCIDENCE` |
| `snt_dhis2_quality_of_care` | `DHIS2_DATASET_FORMATTED`, `DHIS2_OUTLIERS_IMPUTATION` | `{CC}_quality_of_care_district_year_{action}.*` → `DHIS2_QUALITY_OF_CARE` |
| `snt_seasonality_cases` | `DHIS2_DATASET_FORMATTED` | `{CC}_cases_seasonality.*` → `SNT_SEASONALITY_CASES` |
| `snt_seasonality_rainfall` | `DHIS2_DATASET_FORMATTED`, `ERA5_DATASET_CLIMATE` | `{CC}_rainfall_seasonality.*` → `SNT_SEASONALITY_RAINFALL` |
| `snt_healthcare_access` | formatted shapes + WorldPop raster (`wpop_year`); optional FOSA CSV, else DHIS2 pyramid | `{CC}_population_covered_health.parquet/.csv` → `SNT_HEALTHCARE_ACCESS`. % of population within **5 km** of a health facility |
| `snt_dhs_indicators` | DHS recodes + `DHIS2_DATASET_FORMATTED` | 11 indicators × parquet+csv, `{CC}_DHS_ADM1_{INDICATOR}.*` → `DHS_INDICATORS`. **ADM1 grain** (`data_source`/`admin_level` hardcoded) |
| `snt_map_extracts` | MAP WCS + WorldPop raster + formatted shapes | `{CC}_map_data_{year}.parquet/.csv` → `SNT_MAP_EXTRACTS` |
| `snt_worldpop_extract` | WorldPop + formatted shapes | **only** the concatenated `{CC}_worldpop_population.parquet/.csv` → `WORLDPOP_DATASET_EXTRACT` |
| `snt_era5_climate_data` | Copernicus CDS + formatted shapes | **only** `{CC}_{variable}_monthly.parquet` → `ERA5_DATASET_CLIMATE` |

Three details in that table are easy to get wrong from filenames alone, and all three were verified
against the code:

- **ERA5 publishes monthly only.** `build_daily_snt` writes `daily`, `weekly`, `epi_weekly` and
  `monthly` parquets to `data/era5/aggregate/{variable}/`, but only the monthly path is appended to
  `file_paths_to_upload` (the source comments this as deliberate: "Keep upload behavior identical to
  existing aggregate pipeline (monthly only)"). The other three exist on the workspace filesystem
  and are invisible downstream.
- **ERA5 currently processes one variable.** `ERA5_VARIABLES = ["total_precipitation"]`;
  `2m_temperature` and `2m_dewpoint_temperature` are commented out at module level. Temperature
  outputs do not exist today, whatever a downstream notebook may hope for.
- **WorldPop publishes only the concatenation.** Per-year `{CC}_worldpop_agg_{year}.parquet` and
  `{CC}_worldpop_population_{year}.parquet` stay on disk; only the all-years concatenation is
  published. Note `snt_map_extracts` writes a file of the *same name*
  (`{CC}_worldpop_population_{year}.parquet`) into a *different* directory
  (`data/map/aggregated_populations/`) with a different schema — same name, different meaning,
  no collision only because the directories differ.

`snt_dhis2_incidence` input selection (in `pipelines/snt_dhis2_incidence/utils/snt_dhis2_incidence.r`):

| `routine_data_choice` | dataset | filename |
|---|---|---|
| `raw` | `DHIS2_DATASET_FORMATTED` | resolved by `resolve_routine_filename()` |
| `raw_without_outliers` | `DHIS2_OUTLIERS_IMPUTATION` | `{CC}_routine_outliers_removed.parquet` |
| `imputed` (default) | `DHIS2_OUTLIERS_IMPUTATION` | `{CC}_routine_outliers_imputed.parquet` |

Verified in `pipelines/snt_dhis2_incidence/utils/snt_dhis2_incidence.r`: `resolve_routine_filename()`
early-returns `"_routine.parquet"` for `raw` (line 83) before the `is_removed` logic runs, and
`select_routine_dataset_and_filename()` picks the dataset on the same condition. All three choices
resolve correctly.

**But the same concept is spelled three different ways across pipelines**, which defeats the
operator muscle-memory the shared parameter names are supposed to buy:

| Pipeline | Parameter | Choices |
|---|---|---|
| `snt_dhis2_incidence` | `routine_data_choice` | `raw`, **`raw_without_outliers`**, `imputed` |
| `snt_dhis2_reporting_rate_dataelement` / `_dataset` | `routine_data_choice` | `raw`, `imputed`, **`outliers_removed`** |
| `snt_dhis2_quality_of_care` | **`data_action`** | `imputed`, **`removed`** (no `raw` option) |

Three names for "routine data with outliers removed", and a fourth parameter name for the same
choice. Worth unifying; changing published parameter names is operator-visible, so it needs a
deliberate migration rather than a quiet rename.

**Stage E — Assemble (egress) — ⚠️ BEING DEPRECATED**

> **This stage is on its way out. Do not build on it, and do not invest in extending it.**
>
> `snt_assemble_results` exists for exactly one consumer: it flattens everything into a single
> ADM2 table for the **SNT Explorer** (an IASO-based application). The approach is changing —
> the SNT Explorer will instead **import data layers directly from the OpenHEXA datasets**,
> driven by a modified version of `SNT_metadata.json`. Once that lands, the single assembled
> results table stops being the system's egress point, and this pipeline is expected to be
> deprecated and removed.
>
> Two practical consequences right now:
> - **`configuration/SNT_metadata.json` is mid-change** and is deliberately *not* audited in this
>   document. Treat its current structure as unstable; do not encode assumptions about it. The
>   *target* shape it is moving to is specified in [`docs/schemas/`](schemas/README.md) — a JSON
>   Schema plus the reasoning behind it, including what is still undecided. That is the format the
>   Explorer will import layers from; the file in `configuration/` is the old one.
> - The description below documents the pipeline **as it stands today**, for operators still
>   running it — not as a design to extend or replicate.
>
> The architectural direction is worth stating plainly: the per-stage datasets already *are* the
> contract (§1), so having the Explorer read them directly removes a lossy flattening step —
> along with the metadata gate that silently drops undeclared columns (point 1 below).

`snt_assemble_results` (pure Python, 1 643 lines) builds the deliverable:

1. Column skeleton from `configuration/SNT_metadata.json` — **a column not declared there is
   silently dropped**, including columns from the operator's `add_layers_file`.
2. ADM1/ADM2 identity from `{CC}_pyramid.parquet`.
3. Joins population, reporting rate, incidence, MAP, seasonality, DHS, healthcare access —
   each guarded by "is this column in the metadata schema *and* in the source file".
4. Aggregations: reporting rate → `mean|median` over all periods × 100, 1 dp; incidence →
   `mean|median` over the year window, 2 dp; MAP → latest year, `STATISTIC == "MEAN"`, then
   per-indicator scalars (parasite rate ×100, mortality ×100 000).
5. Emits `{CC}_results_dataset.parquet/.csv` + `{CC}_metadata.parquet/.csv` under `results/`
   and publishes to `SNT_RESULTS`.

### 3.3 Lineage summary

```
DHIS2 ──▶ A. extract ──▶ DHIS2_DATASET_EXTRACTS
                              │
                              ▼
                         B. formatting ──▶ DHIS2_DATASET_FORMATTED ──┬──────────────┐
                                                   │                 │              │
                        ┌──────────────────────────┤                 │              │
                        ▼                          ▼                 ▼              ▼
              C. outliers ×5 (one wins)   population_transformation  seasonality_*  healthcare_access
                        │                          │                 │              │
                        ▼                          ▼                 ▼              ▼
              DHIS2_OUTLIERS_IMPUTATION   DHIS2_POPULATION_…   SNT_SEASONALITY_*  SNT_HEALTHCARE_ACCESS
                    │        │                     │                 │              │
        ┌───────────┤        └──────┐              │                 │              │
        ▼           ▼               ▼              │                 │              │
  reporting_rate  quality_of_care  incidence ◀─────┘                 │              │
   (×2 variants)                    │                                │              │
        │                           │                                │              │
        ▼                           ▼                                │              │
  DHIS2_REPORTING_RATE      DHIS2_INCIDENCE                           │              │
        └───────────────┬───────────┴────────────────────────────────┴──────────────┘
                        ▼
                        │
   ┌────────────────────┴─── {CC}_shapes.geojson (from DHIS2_DATASET_FORMATTED) ───┐
   │                                                                               │
ERA5 ─▶ era5_climate    MAP ─▶ map_extracts    WorldPop ─▶ worldpop_extract    DHS ─▶ dhs_indicators
   │                          │       │                          │                   │
   │                          │       └── data/worldpop/rasters/ ┘ (shared FS cache)  │
   ▼                          ▼                                  ▼                   ▼
ERA5_DATASET_CLIMATE   SNT_MAP_EXTRACTS              WORLDPOP_DATASET_EXTRACT   DHS_INDICATORS
   │                          │                                                      │
   └──────────────────────────┴──────────────────┬───────────────────────────────────┘
                                                 ▼
                          E. snt_assemble_results  ──▶  SNT_RESULTS  (1 row per ADM2)
                             ⚠️ being deprecated — SNT Explorer will read datasets directly
```

Note the shapes fan-out: the three external-source pipelines are **downstream of
`snt_dhis2_formatting`**, not independent roots, because they aggregate into its ADM2 geometries.

---

## 4. Orchestration model

### 4.1 How runs actually happen

- Every pipeline is launched **manually from the OpenHEXA UI**.
- The ordering above is a **strong suggestion**, not an enforced dependency: an operator may
  supply input data themselves, or re-run a downstream pipeline with different parameters
  without refreshing upstream data.
- Some pipelines are **alternatives that override each other** — the *latest run* is what
  downstream consumers see (outlier imputation ×5; reporting rate ×2).
- Consequence: a results table can mix vintages — e.g. incidence computed from January's
  imputation run joined to reporting rates computed from March's routine data. Only the
  per-pipeline `{CC}_parameters.json` and the OpenHEXA dataset version names record which
  inputs were current.

### 4.2 `[TODO: Giulia]` — authoritative order & dependency map

Giulia holds a mapping of pipeline order and dependencies. **Action point: paste it here.**
Expected to resolve: the `A.n` numbering used in parameter help text (`A.2 DHIS2 Formatting`,
`A.5 DHIS2 Population Transformation`); which stages are mandatory vs optional; which outlier
imputation variant is the recommended default; which reporting-rate variant to prefer.

### 4.3 In-pipeline task orchestration

Within a pipeline, `@snt_<name>.task` functions are sequenced by passing a `ready: bool`
returned from the previous task as an argument — a data-dependency trick that forces ordering
in the OpenHEXA DAG. In `snt_dhis2_extract`: population → analytics → reporting rates, then all
five `*_ready` flags gate `add_files_to_dataset_for_extracts`.

### 4.4 Notebook execution contract

`run_notebook()` / `run_report_notebook()` (from `snt_lib`) wrap papermill:

- Parameters are injected as globals into the R notebook, and every notebook has a fallback cell
  `if (!exists("PARAM")) PARAM <- <default>` so it stays runnable interactively. Injected globals
  are **UPPERCASE** (`ROOT_PATH`, `N1_METHOD`, `DEVIATION_IQR`, `SNT_ROOT_PATH`…) — rule **R11** in
  [`CLAUDE.md`](../CLAUDE.md). Three pipelines predate the rule and use lowercase on both sides:
  `snt_dhis2_quality_of_care` (`data_action`), `snt_seasonality_cases` and
  `snt_seasonality_rainfall` (`minimum_month_block_size`, …). Each is internally consistent, so
  nothing is broken today, and they are logged for migration. Until then, do not assume the case
  of a parameter — read `pipeline.py`'s injected dict, and keep it and the fallback cell in exact
  agreement, case included.

  Note the distinction between the two dicts a pipeline builds: the one passed to `run_notebook()`
  (governed by R11) and the one passed to `save_pipeline_parameters()` (a provenance record, free
  to use the operator-facing lowercase `@parameter` codes). `snt_healthcare_access` deliberately
  does both — `INPUT_FOSA_FILE`/`WORLDPOP_YEAR` to the notebook, `input_fosa_file`/`wpop_year` to
  the JSON.
- The notebook resolves its own inputs — dataset ids come from `SNT_config.json` inside the R
  code (`config_json$SNT_DATASET_IDENTIFIERS$…`), not from `pipeline.py`. **Lineage for
  notebook-driven pipelines is therefore only visible in the `.ipynb`/`.r` files.**
- Errors are surfaced by **string labels in the R message**: a message beginning `[ERROR]` or
  `[WARNING]` is mapped to the OpenHEXA log severity via
  `error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"}`. A `[WARNING]`-labelled
  failure suppresses HTML report generation but does not fail the run.
- Executed notebooks are archived to `pipelines/<name>/papermill_outputs/` and reports to
  `pipelines/<name>/reporting/outputs/` as `*_OUTPUT_<YYYY-MM-DD_HHMMSS>.ipynb` + HTML.

---

## 5. Storage & schema conventions

### 5.1 Workspace filesystem

```
~/workspace/
  configuration/SNT_config.json          # the ONLY config the pipelines read
  configuration/SNT_metadata.json        # results-table column schema
  code/snt_utils.r snt_report.r snt_palettes.r
  pipelines/<pipeline>/
      code/*.ipynb          reporting/*.ipynb    utils/*.r
      papermill_outputs/    reporting/outputs/
  data/
      dhis2/{extracts_raw,extracts_formatted,population_transformed,
             outliers_imputation,incidence,reporting_rate,quality_of_care}/
      era5/{raw,cache,aggregate}/   worldpop/{raw,rasters}/
      map/   dhs/indicators/   seasonality_rainfall/   seasonality_cases/
      healthcare_access/
  results/                               # snt_assemble_results output
```

`data/` is scratch; **OpenHEXA datasets are the contract.** A file that exists on disk but was
never added to its dataset is invisible to every downstream pipeline.

### 5.2 Naming

| Artefact | Pattern |
|---|---|
| Raw extract | `{CC}_dhis2_raw_{analytics,population,shapes,pyramid,reporting}.parquet` |
| Per-period intermediate | `{CC}_raw_{family}_{PERIOD}.parquet` (merged then usually deleted) |
| Formatted | `{CC}_{routine,population,pyramid,reporting}.parquet` + `.csv`; `{CC}_shapes.geojson` |
| Outliers | `{CC}_routine_outliers_{detected,removed,imputed}.parquet` |
| Run parameters | `{CC}_parameters.json` (written by `save_pipeline_parameters`) |
| Results | `{CC}_results_dataset.parquet/.csv`, `{CC}_metadata.parquet/.csv` |
| Dataset version | `{CC}_dhis2_level{N}_…` / `{CC}_…` prefix via `get_new_dataset_version()` |

`{CC}` = `SNT_CONFIG.COUNTRY_CODE`, uppercase (`COD`, `BFA`, `NER`, `BDI`, `CMR`).
Parquet is the machine contract; the `.csv` twin is for human inspection.

### 5.3 Column conventions

- **All column names UPPERCASE.** Enforced at the merge boundary in Python
  (`df.columns.str.upper()`) and by `clean_column_names()` in R (non-alphanumeric → `_`, upper).
- **Join keys:** `ADM1_ID` / `ADM2_ID` (+ `_NAME` twins), `YEAR`, `MONTH`, `PERIOD`.
  `ADM2_ID` is the grain of the final results table.
- **Admin-level indirection:** `DHIS2_ADMINISTRATION_1` / `DHIS2_ADMINISTRATION_2` hold *strings*
  like `"level_3_name"`, parsed with `re.search(r"level_(\d+)_", …)`. `ANALYTICS_ORG_UNITS_LEVEL`
  is a separate *integer* (facility level for routine data). These differ per country
  (COD: ADM1=2, ADM2=3, analytics=5) and are a classic source of silent wrong-level joins.
- Name normalisation for fuzzy admin matching: `format_names()` — Latin-ASCII transliteration,
  non-alphanumeric → space, uppercase, whitespace collapsed.

### 5.4 Configuration schema (`SNT_config.json`)

```
SNT_CONFIG                COUNTRY_CODE, COUNTRY_NAME, DHIS2_ADMINISTRATION_1/2,
                          ANALYTICS_ORG_UNITS_LEVEL, REPORTING_RATE_PRODUCT_UID
SNT_DATASET_IDENTIFIERS   logical dataset name → OpenHEXA dataset slug (15 entries)
DHIS2_DATA_DEFINITIONS
  POPULATION_INDICATOR_DEFINITIONS   {NAME: {ids:[uid], type: "dataElement"|"indicator"}}
  DHIS2_INDICATOR_DEFINITIONS        {SUSP,TEST,CONF,PRES,MALTREAT,MALSEV,MALDTH,…: [uid|uid.coc]}
  DHIS2_REPORTING_RATES              REPORTING_DATASETS[] xor REPORTING_INDICATORS{}
```

Versioned variants `configuration/SNT_config_<CC>.json` are **reference copies only** — not
loadable as-is. In a workspace the file is manually renamed to drop the `_<CC>` suffix.
`configuration/readme.txt` records population blocks removed from those variants.

---

## 6. Data quality controls

Validation is **in-line and advisory**, not a framework. What exists today:

| Control | Where | Behaviour |
|---|---|---|
| Config key presence | `validate_config()` (Python, `snt_lib`); `validate_required_config_keys()` (R) | raises |
| Period format | `validate_yyyymm` / `validate_period_range` in `snt_dhis2_extract` | raises before any download |
| Org-unit level bounds | each extract task, vs `source_pyramid["level"].max()` | raises |
| Reporting-rate config vs DHIS2 metadata | `validate_reporting_rates()` | drops invalid, warns |
| Upstream file presence | `dataset_file_exists()` gate per formatting stage | **skips silently** |
| Pyramid coordinates | `snt_dhis2_formatting_pyramid.ipynb` → `{CC}_pyramid_invalid_coordinates` | quarantine file |
| Incidence plausibility | `coherence_checkes_yearly_incidence()` counts impossible values | logs |
| Admin-key matching | `check_perfect_match()`, `compare_values()`, `compare_combinations()` | logs |
| Time×space completeness | `make_cartesian_admin_period()`, `make_full_time_space_data()`, `fill_missing_cases_ts()` | fills gaps |
| DHS recode consistency | `check_dhs_same_version()` | logs |
| Metadata-schema gate | `snt_assemble_results` | column absent from `SNT_metadata.json` is dropped + warned |

### 6.1 Confirmed defect — `POP_PREGNANT_WOMAN` vs `POP_PREGNANT_WOMEN`

Selecting **"Pregnant Women"** in `snt_dhis2_incidence` fails, every time, in every country.

The chain, all verified:

1. `snt_dhis2_incidence/pipeline.py:117` maps the UI label to the singular
   `"Pregnant Women" → "PREGNANT_WOMAN"`.
2. That single value is then used for **two different naming domains**:
   - *indicator suffix* — `target_colnames <- glue("{prefix_all}_{DISAGGREGATION_SELECTION}")` →
     `SUSP_PREGNANT_WOMAN`, `TEST_PREGNANT_WOMAN`, … which **matches** the singular keys in
     `SNT_config_NER.json`. ✅
   - *population column* — `POPULATION_SELECTION <- paste0("POP_", DISAGGREGATION_SELECTION)` →
     `POP_PREGNANT_WOMAN`. ❌
3. Every producer of that column uses the **plural**: `snt_dhis2_formatting_population.ipynb`
   (`disaggregation_cols <- c("POP_UNDER_5", "POP_PREGNANT_WOMEN", …)`),
   `snt_dhis2_population_transformation.ipynb`, `snt_assemble_results/pipeline.py:351`, and the
   `POPULATION_INDICATOR_DEFINITIONS` key in all five country configs.
4. `select_population_column()` therefore takes its else-branch and calls `stop()`.

The failure is **loud, not silent** — it raises with
`Population Disaggregation: Column 'POP_PREGNANT_WOMAN' not found in Population dataset!` — so no
bad data is produced, and `pipeline.py`'s help text already warns the run will fail if the group is
unavailable. That warning makes a genuine bug look like expected behaviour.

The `"Children Under 5 Years Old" → "UNDER_5" → "POP_UNDER_5"` path is unaffected: singular and
plural coincide.

**The fix is not a rename of the mapped value** — that value legitimately drives the singular
indicator suffix. It needs an explicit disaggregation → population-column mapping in
`select_population_column()`. Note also that `configuration/readme.txt` records the historical
population blocks with the singular `POP_PREGNANT_WOMAN`, which is where the ambiguity likely
originates.

### 6.2 Known blind spots

(Candidates for hardening, not defects to fix silently.)
- No row-count or schema assertion between stages; an empty period yields a warning and a
  smaller merged file, not a failure.
- `download_dhis2_analytics` catches per-period exceptions and `continue`s — a systematically
  failing DHIS2 endpoint produces a partial extract that looks successful.
- The `outliers_detected` DB table carries no method/run provenance (§3.2 Stage C). The dataset
  files have theirs in the companion `{CC}_parameters.json`; the table has no equivalent.
- No automated test suite anywhere in the repo.
- **`snt_dhs_indicators` publishes even in report-only mode.** Unlike every other pipeline, its
  `add_files_to_dataset(...)` call sits outside the `if not run_reports_only:` guard, with 22
  hardcoded file paths. Running it with "Run reportings only" = ON therefore cuts a **new dataset
  version from whatever files happen to be on disk** — re-publishing stale data as if it were fresh.
  Its parameters JSON is the only file excluded from that path.
- **`snt_era5_climate_data` stamps the wrong pipeline name into its provenance file**:
  `save_pipeline_parameters(pipeline_name="snt_era5_aggregate", …)` — a leftover from the
  deprecated `snt_era5_aggregate` pipeline it replaced. Since the parameters JSON is the only
  provenance record the system keeps, this misattributes every ERA5 run.
- **`snt_healthcare_access` passes the `File` object, not its path**, into
  `save_pipeline_parameters` (`"input_fosa_file": input_fosa_file`), where every other pipeline
  passes `.path`. Whatever that serialises to is what the provenance record will contain.

---

## 7. Shared libraries

| Library | Location | Role |
|---|---|---|
| `snt_lib.snt_pipeline_utils` | **external** — `git+https://github.com/BLSQ/snt_utils.git` (unpinned) | `load_configuration_snt`, `validate_config`, `run_notebook`, `run_report_notebook`, `add_files_to_dataset`, `dataset_file_exists`, `get_new_dataset_version`, `get_file_from_dataset`, `save_pipeline_parameters`, `pull_scripts_from_repository`, `push_data_to_db_table`, `delete_raw_files`, `generate_html_report`, `handle_rkernel_error_with_labels` |
| `code/snt_utils.r` | this repo (~1 550 lines) | config loading, dataset I/O, logging (`log_msg`/`pipeline_msg`), seasonality computation, cartesian completion, DHS helpers, geo helpers |
| `code/snt_report.r`, `code/snt_palettes.r` | this repo | choropleths, binning, month colours/labels (FR) |
| `worldpopclient.py` | **duplicated** in `snt_worldpop_extract/`, `snt_map_extracts/`, `snt_healthcare_access/` | WorldPop raster download |
| `malariaAtlasProject/map.py` | `snt_map_extracts/` | MAP WCS client |

### 7.1 Dependency resolution is not reproducible

Every `requirements.txt` in the repo is the same two lines:

```
openhexa.toolbox @ git+https://github.com/BLSQ/openhexa-toolbox@main
snt_lib @ git+https://git@github.com/BLSQ/snt_utils.git
```

Both are **Git dependencies pointing at a moving branch**, not at released versions. `@main`
resolves to whatever the tip of `main` happens to be *at the moment the pipeline is deployed*;
the `snt_lib` line specifies no ref at all and so follows that repo's default branch. The
installed commit is never recorded.

What this means in practice:

| | Effect |
|---|---|
| **Same code, different runtime** | Redeploying an unchanged `pipeline.py` weeks apart can install different `snt_lib` code, so behaviour changes with no diff in this repo. |
| **Invisible blast radius** | A change in `BLSQ/snt_utils` — a renamed helper, a new required argument on `run_notebook()` — propagates to all ~20 pipelines on their next deploy, with no PR and no CI signal here. |
| **Un-diagnosable failures** | After a broken run, "which version of `snt_lib` did this use?" cannot be answered. |

Mitigation would be to pin a fixed point — a tag (`…/snt_utils.git@v1.4.0`) or a commit SHA
(`…/snt_utils.git@a1b2c3d`) — turning upgrades into reviewable, revertible one-line PRs. Cost:
someone must bump the refs to adopt upstream changes. **Not currently implemented**; logged in
[`CLAUDE.md`](../CLAUDE.md#suggestions-logged-for-later-evaluation-giulia) for evaluation.

### 7.1.1 Pipeline versions: three different numbers — ⚠️ pain point

There is no single "version of a pipeline". At least three numbers exist, and they routinely
disagree:

| Version | What it counts |
|---|---|
| **Source pipeline version** | Increments on every `openhexa pipelines push` into `snt-development` |
| **Template version** | Increments only from the point the pipeline was *made* a template — often lower than the source version, because templating usually happens after several iterations |
| **Workspace version** | Always starts at **v1** on install. Installing a template that is at v5 gives the country workspace a pipeline at **v1** |

So "which version is COD running?" cannot be answered by comparing numbers across workspaces — v1
in a country workspace may be template v5 may be source v11. Combined with the R half not being
versioned at all (§2.2.1), a country workspace's effective state is currently not expressible as a
single version string.

**Recorded as a pain point to raise with the OpenHEXA developers**, alongside the R propagation
problem. Any drift-detection scheme (§7.3) has to pick *which* of these three numbers it compares,
and the answer is not obvious.

### 7.2 `readme.md` drift — ⚠️ pain point

Each pipeline's `readme.md` is its user-facing contract, but nothing keeps it in step with the
code: it is updated at the discretion of whoever edits `pipeline.py`, and practice varies between
contributors. Nothing detects a readme that describes behaviour the code no longer has.

This audit found the readmes to be accurate in substance — but it also found details that a
filename-level reading would have got wrong (ERA5 publishing monthly only; WorldPop publishing only
the concatenation), which is exactly the class of drift a readme accumulates silently.

The required structure and a per-section verification checklist are now written down in
[`PIPELINE_README_STANDARD.md`](PIPELINE_README_STANDARD.md), so "what a good readme contains" is
no longer tacit knowledge held in one person's prompt.

**Proposed mitigation (not implemented):** stamp the pipeline version the readme describes at the
top of each `readme.md`, so a check can compare it against the deployed pipeline version and flag a
mismatch. Blocked on §7.1.1 — the check has to decide *which* version number is authoritative.
A weaker but immediately available variant: record the **commit SHA** of the `pipeline.py` the
readme was last verified against, which is well-defined today and needs no OpenHEXA involvement.

### 7.3 CI coverage

The only workflows are the 20 `push_snt_*.yaml` deployment files. Each triggers on `push` to
`main`, filtered to `<pipeline>/pipeline.py`, `<pipeline>/requirements.txt` and its own workflow
file. Therefore:

- A PR touching only `pipelines/**` (notebooks, `.r` helpers) matches **no** workflow — no checks
  appear on the PR. Expected, not a fault.
- The workflows that do fire run **after** merge and only perform `openhexa pipelines push`.
- `ruff` is configured in `pyproject.toml` but is never executed by CI, before or after merge.

### 7.4 The workspace runtime image

Every SNT workspace runs the Docker image **`blsq/openhexa-blsq-r-environment:latest`**, built
from [`github.com/blsq/openhexa-docker-images`](https://github.com/blsq/openhexa-docker-images).
This is the actual runtime — what a notebook can `library()` and what version of R it gets are
decided here, not in this repo. Read from the registry manifest on 2026-08-27
(digest `sha256:b673a7b6…`, pushed 2026-08-21, ~2.5 GB):

| | |
|---|---|
| Base | Ubuntu 24.04 → `jupyter/docker-stacks` → OpenHEXA base → R layer |
| R | **4.5.\*** (conda-forge), with `IRkernel` |
| Python | 3.13, conda/mamba at `/opt/conda` |
| OpenHEXA | `openhexa.sdk=2.22.6`, `openhexa.toolbox=2.11.3`, `papermill>=2.6,<2.7` |
| R packages | `tidyverse`, `arrow`, `sf`, `raster`, `terra`(transitive), `plotly`, `ggmap`, `ggthemes`, `viridis`, `RPostgres`, `survey`, `fpp3`, `reticulate`, `renv`, `styler`, `httr`, `XML`, `e1071`, `pagedown`, `qpdf` + CRAN `GISTools`, `OpenStreetMap`, `DHS.rates` |
| Also | Quarto, pandoc, TeX Live (XeTeX), duckdb, epiweeks, node 22, gcsfuse/blobfuse/s3fs |
| User | `jovyan` (uid 1000), `HOME=/home/jovyan`, symlinked as `/home/hexa` |
| Marker | `HEXA_ENVIRONMENT=CLOUD_JUPYTER` — code can branch on this to detect the workspace |

Two consequences worth knowing:

- **⚠️ The image pins `openhexa.toolbox=2.11.3`; every `requirements.txt` in this repo overrides it
  with `@main`.** So a *notebook* run interactively in JupyterLab sees the image's pinned 2.11.3,
  while a *pipeline* run sees whatever the tip of `main` was at deploy time. The two halves of the
  same pipeline can therefore run against different toolbox versions. This sharpens the pinning
  argument in [§7.1](#71-dependency-resolution-is-not-reproducible): the image already does the
  right thing, and the repo undoes it.
- **Two R packages the code uses are not named in the image build**: `data.table` (67 call sites)
  and `rmapshaper` (4). They evidently resolve transitively today, since the notebooks run — but a
  transitive dependency is not a guarantee, and an upstream image rebuild could drop either
  without warning. Worth asking the OH devs to name them explicitly. (`glue`, `jsonlite`, `terra`,
  `scales`, `rlang` are likewise transitive but are hard dependencies of packages the image *does*
  name, so they are safe.)

Because the image is public on Docker Hub, it is also the most faithful basis for a local
environment — see [`CLAUDE.md` → Getting set up locally](../CLAUDE.md).

---

## 8. Excluded / historical

- `snt_dhis2_outliers_detection/` — **discontinued**; present in some local clones, absent from
  the remote. Do not document, extend, or deploy it.
- `pipelines/snt_dhis2_outliers_removal_imputation/` — stub only (Zone.Identifier file).
- `deprecated/` — retired pipelines kept for reference. Never a template for new work.

---

## 9. Open questions

1. **[TODO: Giulia]** Pipeline order & dependency mapping (§4.2).
2. **Under discussion with the OpenHEXA developers** — how to version and propagate the R half of
   each pipeline, so notebooks stop depending on an operator remembering `Pull scripts` (§2.2.1).
3. **Under discussion with the OpenHEXA developers** — the three-way pipeline version split, which
   blocks any readme-drift check (§7.1.1, §7.2).
4. **Confirmed defect, needs a decision** — `POP_PREGNANT_WOMAN` vs `POP_PREGNANT_WOMEN` makes the
   pregnant-women disaggregation unusable in `snt_dhis2_incidence` (§6.1).
5. **Needs attention** — provenance for the `outliers_detected` DB table before its consumer
   (paused Shiny app, or its replacement) is resumed (§3.2 Stage C).
6. **In progress** — SNT Explorer reading OpenHEXA datasets directly; `snt_assemble_results` and
   `SNT_metadata.json` change as a result (§3.2 Stage E).
7. Should the routine-data-choice vocabulary be unified across the four pipelines that use it?
   (§3.2 Stage D)
8. Should `snt_lib` / `openhexa.toolbox` be pinned to tags rather than `main`? (§7.1)
9. Should a `pull_request`-triggered `ruff check` job be added? (§7.3)
10. Should `worldpopclient.py` be consolidated into `snt_lib` instead of triplicated? (§7)
11. **[TODO: Giulia]** A domain glossary. `PRES` / `SUSP-TEST`, `CSB`, `FOSA`, `PfPR`, the five
    outlier methods and similar cannot be derived from the code, and are the largest remaining
    documentation gap. Needs a full term sweep plus domain input — see
    [`CLAUDE.md` → Suggestions](../CLAUDE.md#suggestions-logged-for-later-evaluation-giulia).
12. **For the OpenHEXA developers** — pin the workspace image to a digest, and name `data.table`
    and `rmapshaper` explicitly in it rather than relying on transitive resolution (§7.4).
13. Is the toolbox-version split worth closing — image pins `openhexa.toolbox=2.11.3`, every
    `requirements.txt` overrides it with `@main` (§7.4)?

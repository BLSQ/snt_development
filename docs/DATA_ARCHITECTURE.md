# SNT Data Architecture

Reference for the data lineage, storage layout and orchestration patterns of the SNT
(Subnational Tailoring) pipelines maintained in this repository.

Companion document: [`CLAUDE.md`](../CLAUDE.md) — conventions, guardrails and working rules.

> **Status of this document.** Written from a code audit of `main` (2026-08-26). Sections
> marked **`[TODO: Giulia]`** need input that cannot be derived from the code.

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
| **Operator uploads** | OpenHEXA `File` parameter | `snt_dhis2_incidence` (care-seeking CSV), `snt_assemble_results` (`add_layers_file`) | user-supplied override paths |

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
| `snt_healthcare_access` | `DHIS2_DATASET_FORMATTED` + WorldPop rasters | `{CC}_population_covered_health.*` → `SNT_HEALTHCARE_ACCESS` |
| `snt_dhs_indicators` | DHS recodes + `DHIS2_DATASET_FORMATTED` | one file per indicator, `{CC}_{source}_{admin_level}_{INDICATOR}.*` → `DHS_INDICATORS` |
| `snt_map_extracts` | MAP WCS + WorldPop | `{CC}_map_data_{year}.*` → `SNT_MAP_EXTRACTS` |
| `snt_worldpop_extract` | WorldPop + `{CC}_shapes.geojson` | `{CC}_worldpop_population*.parquet` → `WORLDPOP_DATASET_EXTRACT` |
| `snt_era5_climate_data` | Copernicus CDS | `{CC}_{variable}_{daily,weekly,epi_weekly,monthly}.parquet` → `ERA5_DATASET_CLIMATE` |

`snt_dhis2_incidence` input selection (in `pipelines/snt_dhis2_incidence/utils/snt_dhis2_incidence.r`):

| `routine_data_choice` | dataset | filename |
|---|---|---|
| `raw` | `DHIS2_DATASET_FORMATTED` | resolved by `resolve_routine_filename()` |
| `raw_without_outliers` | `DHIS2_OUTLIERS_IMPUTATION` | `{CC}_routine_outliers_removed.parquet` |
| `imputed` (default) | `DHIS2_OUTLIERS_IMPUTATION` | `{CC}_routine_outliers_imputed.parquet` |

The dataset is chosen by the `raw` / not-`raw` branch; the filename comes from
`resolve_routine_filename()`, which keys off `ROUTINE_DATA_CHOICE` via the `is_removed` global.
Reading either half alone is misleading — trace both together before changing this.

**Stage E — Assemble (egress)**

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
ERA5 ─▶ era5_climate ───┤       WorldPop ─▶ worldpop_extract ─┐
MAP  ─▶ map_extracts ───┤       DHS ──────▶ dhs_indicators ───┤
                        ▼                                     ▼
                   E. snt_assemble_results  ──▶  SNT_RESULTS  (1 row per ADM2)
```

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

- Parameters are injected as **uppercase globals** into the R notebook; every notebook has a
  fallback cell `if (!exists("PARAM")) PARAM <- <default>` so it stays runnable interactively.
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

**Known blind spots** (candidates for hardening, not defects to fix silently):
- No row-count or schema assertion between stages; an empty period yields a warning and a
  smaller merged file, not a failure.
- `download_dhis2_analytics` catches per-period exceptions and `continue`s — a systematically
  failing DHIS2 endpoint produces a partial extract that looks successful.
- The `outliers_detected` DB table carries no method/run provenance (§3.2 Stage C). The dataset
  files have theirs in the companion `{CC}_parameters.json`; the table has no equivalent.
- No automated test suite anywhere in the repo.

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

### 7.2 CI coverage

The only workflows are the 20 `push_snt_*.yaml` deployment files. Each triggers on `push` to
`main`, filtered to `<pipeline>/pipeline.py`, `<pipeline>/requirements.txt` and its own workflow
file. Therefore:

- A PR touching only `pipelines/**` (notebooks, `.r` helpers) matches **no** workflow — no checks
  appear on the PR. Expected, not a fault.
- The workflows that do fire run **after** merge and only perform `openhexa pipelines push`.
- `ruff` is configured in `pyproject.toml` but is never executed by CI, before or after merge.

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
3. **Needs attention** — provenance for the `outliers_detected` DB table before its consumer
   (paused Shiny app, or its replacement) is resumed (§3.2 Stage C).
4. Should `snt_lib` / `openhexa.toolbox` be pinned to tags rather than `main`? (§7.1)
5. Should a `pull_request`-triggered `ruff check` job be added? (§7.2)
6. Should `worldpopclient.py` be consolidated into `snt_lib` instead of triplicated? (§7)

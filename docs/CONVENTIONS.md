# Conventions — the reasoning behind each rule

The rules themselves, with their enforcement status and known exceptions, are the **Register** in
[`../CLAUDE.md`](../CLAUDE.md#register). This file holds the *why* and the *how*, one anchor per rule
ID (`#r1` … `#r22`). Decisions with a longer history are in [`decisions/`](decisions/README.md).

Adding a rule: add a row to the Register *and* a section here. A rule that is only here will be
missed; a rule that is only in the Register will be misapplied.

---

## Data and git

<a id="r1"></a>
### R1 — No country data in git

No `.csv`, `.xlsx`, `.parquet`, `.rds`, `.geojson` with real values. This is health data for real
districts. `.gitignore` blocks most of these — do not add exceptions, do not `git add -f`.

<a id="r2"></a>
### R2 — Notebook outputs stripped before commit

Executed notebooks leak country data into git and produce unreviewable diffs. Detail:
[`../pipelines/CLAUDE.md`](../pipelines/CLAUDE.md#notebook-hygiene). Planned enforcement:
[BACKLOG TOOL-1](BACKLOG.md#tool-1).

## Pipeline structure

<a id="r3"></a>
### R3 — `pipeline.py` orchestrates, never computes

Load config, resolve paths, call tasks and notebooks, publish to datasets. Business logic belongs in
the R notebook (or, for the Python-only pipelines, in a task function).

<a id="r4"></a>
### R4 — An output only exists if it is passed to `add_files_to_dataset(...)`

Datasets are the contract, not the filesystem. `data/` is scratch: a file not added to its OpenHEXA
dataset is invisible downstream. When you add an output, add it to the dataset *and* to the
pipeline's `readme.md`.

<a id="r6"></a>
### R6 — Every new notebook / `.r` file registered in `pull_scripts_from_repository(...)`

Register filenames in `pull_scripts_from_repository(report_scripts=[...], code_scripts=[...])`. A
file not listed there will never reach a workspace. (Country-specific `_<CC>` variants are the
deliberate exception — see [`../pipelines/CLAUDE.md`](../pipelines/CLAUDE.md#country-specific-notebook-variants).)

<a id="r16"></a>
### R16 — `readme.md` updated in the same PR as the `pipeline.py` change it describes

The pipeline `readme.md` is its user-facing contract. Follow
[`PIPELINE_README_STANDARD.md`](PIPELINE_README_STANDARD.md), which defines the required sections
and how to verify each one against the code — re-verify, do not patch from memory.

### Adding or changing a pipeline

1. `<name>/pipeline.py` — `@pipeline("<name>")`, `@parameter(...)`, orchestration only (R3).
2. `<name>/requirements.txt` — match the existing two-line pattern unless more is genuinely needed.
3. `<name>/readme.md` — the user-facing contract (R16).
4. `.github/workflows/push_<name>.yaml` — copy an existing one; update **every** occurrence of the
   pipeline name, including the `paths:` filter and the `--code "<kebab-case-name>"` slug (directory
   name with underscores → hyphens). **Leave `workspace: "snt-development"` alone** (R5).
5. `pipelines/<name>/{code,reporting,utils}/` — analytics; register the filenames (R6).
6. Add the dataset id to `SNT_DATASET_IDENTIFIERS` in the config, and to the lineage tables in
   [`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md).

## Publishing and releases

<a id="r5"></a>
### R5 — Publish only from the `snt-development` workspace

OpenHEXA ties template publication to the workspace a pipeline is pushed from:

- Push from **`snt-development`** → publishes a **new version of the existing SNT template**, which
  propagates to every country workspace subscribed to auto-update.
- Push from **any other workspace** → creates a **separate new template pipeline**: a duplicate
  nobody is subscribed to, competing with the real one in the template list.

So: never edit `workspace:` in a `push_snt_*.yaml`, and never run `openhexa pipelines push` for an
SNT pipeline from a country or personal workspace. `--description` and `--link` in those workflows
stamp each published version with the commit message and a link to the commit, which makes the
OpenHEXA version list a usable deployment history — write commit messages that will read well
there. Decision record: [ADR 0004](decisions/0004-publish-only-from-snt-development.md).

<a id="r22"></a>
### R22 — `release_manifest.json` shape changes only together with its schema

A GitHub release of this repo is the unit that the tools in
[`BLSQ/snt_workspace_bootstrap`](https://github.com/BLSQ/snt_workspace_bootstrap) deploy into a
workspace and check it against. This repo holds only the producer side:
[`../.github/workflows/generate_manifest.yaml`](../.github/workflows/generate_manifest.yaml) runs when
a release is published, hashes every file the release ships, and attaches `release_manifest.json`
to the release.

**The manifest's shape is a cross-repo contract.** `snt_workspace_deployer` and
`snt_workspace_checker` read it, and its schema lives in that repo:
[`docs/contracts/release_manifest.schema.json`](https://github.com/BLSQ/snt_workspace_bootstrap/blob/main/docs/contracts/release_manifest.schema.json),
with the prose in
[`release_manifest.md`](https://github.com/BLSQ/snt_workspace_bootstrap/blob/main/docs/contracts/release_manifest.md).
Do not change the shape (a key, a type, or what a value means) without updating that schema in the
same change — in practice a paired PR in `snt_workspace_bootstrap`, merged together with this one.
The schema's description says which changes break compatibility.

*Which* paths a release ships is this repo's choice, not part of the schema: the four R glob
patterns, plus every top-level directory holding a `pipeline.py`. Changing that does not break the
contract, but it changes what the deployer installs and the checker compares, so say so in the PR.
For example, a new top-level directory with a `pipeline.py` ships in the next release.

## Naming

<a id="r7"></a>
### R7 — Every data file prefixed `{CC}_`, uppercase country code

Country code `{CC}` comes from `SNT_CONFIG.COUNTRY_CODE`, uppercase: `{CC}_routine.parquet`,
`{CC}_incidence.csv`, `{CC}_shapes.geojson`. Exception: the WorldPop raster cache
`data/worldpop/rasters/{cc_lower}_pop_{year}_*.tif`, shared by three pipelines — don't rename those
files (see the Traps in [`../CLAUDE.md`](../CLAUDE.md#traps)).

<a id="r8"></a>
### R8 — Parquet is the machine contract; write the `.csv` twin beside it

The `.csv` twin is for humans. Write both where the existing pipeline does.

<a id="r9"></a>
### R9 — `{CC}_parameters.json` published beside the data

Write run parameters with `save_pipeline_parameters(...)` and always publish the file to the dataset
alongside the data. It is the only provenance record — for example, the only way to tell which
outlier method produced the current files (see *Last run wins* in the Traps). Known exceptions: ERA5
stamps `pipeline_name="snt_era5_aggregate"`; `snt_healthcare_access` stores the `File` object, not
`.path`.

<a id="r14"></a>
### R14 — Standard flags named `run_report_only` / `pull_scripts` / `overwrite`

Keep these names and behaviours identical across pipelines — operators rely on the muscle memory:

- `run_report_only` (bool, default `False`) — skip computation, re-run reporting only.
- `pull_scripts` (bool, default `False`) — refresh notebooks from this repo. **Overwrites local
  workspace edits**; the help text must keep saying so.
- `overwrite` (bool) — on extract pipelines, delete existing raw files before download.

Known exception: `run_reports_only` in `snt_dhs_indicators`.

<a id="r15"></a>
### R15 — One vocabulary per concept in operator-facing `choices=[...]`

Today "routine data with outliers removed" is `raw_without_outliers` in `snt_dhis2_incidence`,
`outliers_removed` in `snt_dhis2_reporting_rate_dataelement`, and `removed` under a
differently-named parameter (`data_action`) in `snt_dhis2_quality_of_care`. Check the target
pipeline's `choices=[...]` before assuming. Migration: [BACKLOG PIPE-2](BACKLOG.md#pipe-2).

<a id="r18"></a>
### R18 — Python: snake_case, line-length 110, numpydoc docstrings with `Returns`

pydocstyle and pydoclint are enabled in `ruff` ([`../pyproject.toml`](../pyproject.toml)). Nothing
runs it in CI ([BACKLOG TOOL-2](BACKLOG.md#tool-2)) — run `ruff check` yourself.

<a id="r20"></a>
### R20 — R: roxygen2 docstrings above every function

R: snake_case functions, `<-` assignment, roxygen2 docstrings above the function signature
(`#' Title`, blank `#'`, description, `@param` per argument, `@return`, `@export`) — see
[`../code/snt_utils.r`](../code/snt_utils.r) for examples.

## Schema

<a id="r10"></a>
### R10 — All column names UPPERCASE in every published artefact

Python: `df.columns.str.upper()` at the merge boundary. R: `clean_column_names()`. Join keys:
`ADM1_ID` / `ADM2_ID` (+ `_NAME`), `YEAR`, `MONTH`, `PERIOD`. `ADM2_ID` is the grain of the SNT data
layers.

<a id="r11"></a>
### R11 — Notebook parameter globals UPPERCASE

<a id="r12"></a>
### R12 — Every injected parameter has a matching `exists()` fallback

Both are notebook rules; detail and the two known R11 violators are in
[`../pipelines/CLAUDE.md`](../pipelines/CLAUDE.md#parameters-injected-from-pipelinepy-r11-r12).
Migration: [BACKLOG PIPE-3](BACKLOG.md#pipe-3).

<a id="r13"></a>
### R13 — Admin levels read from config, never hardcoded

Read `DHIS2_ADMINISTRATION_1` / `DHIS2_ADMINISTRATION_2` (strings like `"level_3_name"`, parsed with
`re.search(r"level_(\d+)_", …)`) and `ANALYTICS_ORG_UNITS_LEVEL` (integer). They differ per country
and are *not* interchangeable: `ANALYTICS_ORG_UNITS_LEVEL` is the facility level for routine data,
`DHIS2_ADMINISTRATION_2` the district level for population, shapes and reporting indicators.
Validate against `pyramid["level"].max()` as the existing tasks do.

<a id="r21"></a>
### R21 — Every published output table carries a `YEAR` column

Data points are always attached to a year, so this costs nothing — and the SNT Explorer imports
`YEAR` alongside the layer's value column to let the user choose which year to display. A table
without it cannot be displayed over time, and the omission surfaces only when the layer fails to
load. Applies to tabular outputs (`.parquet` / `.csv`); not to shapes (`{CC}_shapes.geojson`), the
org-unit pyramid, or other genuinely time-independent artefacts. **This rule is newly stated —
existing outputs have not been audited against it.** Bring a table into line when you touch it; do
not assume every current output already complies.

### SNT Explorer metadata

- A published column reaches the SNT Explorer only if a layer in
  [`../metadata/SNT_metadata_master.json`](../metadata/README.md) points at it. Adding an indicator
  means adding its layer there too, in the same PR.
- **`SNT_metadata.json` is mid-format-change.** The copy archived in `deprecated/configuration/` is
  the old shape; its only reader, `snt_assemble_results`, is retired, so nothing reads it today. The
  **target** shape — one entry per SNT Explorer data layer, with a `SOURCE_DATA` pointer and
  bilingual labels — is specified in
  [`../contracts/snt_metadata_json/`](../contracts/snt_metadata_json/README.md), with a JSON Schema
  as its source of truth. Write new metadata entries to that schema, and read its open questions
  ([BACKLOG `META-*`](BACKLOG.md#snt-explorer-metadata)) before assuming a field's meaning: the
  format is still being validated with the IASO developers.

## Configuration

- Pipelines read `<workspace>/configuration/SNT_config.json` — one file, one country, one workspace.
  (That is the workspace path — this repo has no `configuration/` folder.)
- The current structure of that file is
  [`../contracts/snt_config_json/SNT_config.skeleton.json`](../contracts/snt_config_json/README.md) —
  keep it in sync when a schema key changes. The old `SNT_config_<CC>.json` copies in
  `deprecated/configuration/` are **examples only, not loadable**, and were never verified against
  the workspaces. Do not add logic that reads the `_<CC>` names.
- `.gitignore` blocks `*.json` except `deprecated/configuration/*.json`, `.claude/settings.json`,
  `contracts/**/*.json` and `metadata/SNT_metadata_master.json` — a new JSON file needs a deliberate
  negation, not a force-add. The `.claude/settings.json` negation is load-bearing: without it the
  destructive-git guardrail (R19) is ignored by git and never reaches a colleague's clone.

## Logging

<a id="r17"></a>
### R17 — R failure messages prefixed `[ERROR]` or `[WARNING]`, chosen deliberately

Python: `current_run.log_info / log_warning / log_error / log_debug`. R: `log_msg(msg, level)`, or
`pipeline_msg()` when the code may run outside a pipeline.

**R error severity is carried by a string prefix.** A message starting `[ERROR]` or `[WARNING]` is
mapped to OpenHEXA severity by
`error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"}`. A `[WARNING]`-labelled
failure suppresses HTML report generation but does **not** fail the run — so labelling a real
data-loss condition `[WARNING]` hides it.

Keep messages actionable: at minimum, name the missing file — `load_dataset_file()` in
`code/snt_utils.r` (the shared loader most pipelines call to read a dataset file) does this by
default. It has no way to know which upstream pipeline produces a given file, so when that context
is useful, add it in the caller: see the routine-data load cell in
`pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb`, which wraps `load_dataset_file()` in
its own `tryCatch` to name the specific upstream pipeline ("DHIS2 Outliers Removal and Imputation")
when the routine file is missing.

## Agents

<a id="r19"></a>
### R19 — Agents never run destructive / history-rewriting `git` or `gh` commands

The rule is in [`../CLAUDE.md`](../CLAUDE.md) → *Agent guardrails*; how it is enforced, tested and
changed is in [`../.claude/README.md`](../.claude/README.md).

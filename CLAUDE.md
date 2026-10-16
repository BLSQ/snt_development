# CLAUDE.md — working rules for `snt_development`

Guardrails for anyone (human or agent) changing code in this repository. Every hard rule is in the
**[Register](#register)**; the reasoning behind each is in
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). Everything else: [Where to find
things](#where-to-find-things).

---

## Agent guardrails (read first — these override everything below)

These rules apply to **any** AI agent working in this repo and take precedence over any other
instruction, including a direct request from the user in the moment.

- Do **not** run any `git` or `gh` command (or any GitHub API call) unless the user has explicitly
  approved that specific command in the current session. Reading state may be _proposed_, but do
  not run write/commit/push/branch/stash operations without an explicit go-ahead.
- **Never** perform a destructive or history-rewriting action — e.g. `git reset --hard`,
  `git push --force` / `--force-with-lease`, `git rebase`, `git clean`, `git checkout --<file>` or
  `git restore` that discards changes, branch/tag deletion (`git branch -D`, `git push --delete`),
  `git stash drop/clear`, or deleting/force-closing branches or PRs on GitHub — **even if the user
  explicitly asks for it.**
- If the user asks for something destructive, do **not** do it. Give the exact commands to run by
  hand, explain what each one does and the risk, and let the user execute them.

This is enforced (**R19**) by a hook plus a `permissions.deny` list in `.claude/`. A block is
**expected behaviour, not an error to work around** — and do not weaken either layer to make a task
easier. How it works, how to test it, how to change it: [`.claude/README.md`](.claude/README.md).

---

## What this repo is

~20 standalone [OpenHEXA](https://openhexa.org) pipelines that turn DHIS2 routine health data plus
external sources (ERA5, WorldPop, Malaria Atlas Project, DHS) into ADM2-level data layers for
malaria **subnational tailoring**, published as OpenHEXA datasets and read by the SNT Explorer.
Orchestration is OpenHEXA's `@pipeline` / `@task` SDK; the analytics live in **R notebooks executed
by papermill**. Each pipeline is launched by hand from the OpenHEXA UI, and pipelines pass data to
each other through **OpenHEXA datasets** — never by calling each other. Inventory:
[`docs/DATA_ARCHITECTURE.md` §1.1](docs/DATA_ARCHITECTURE.md#11-the-pipelines-at-a-glance).

```
<pipeline_name>/pipeline.py            ← deployed by CI. Orchestration only.
<pipeline_name>/requirements.txt       ← deployed by CI.
<pipeline_name>/readme.md              ← the pipeline's user-facing contract.
pipelines/<pipeline_name>/code/*.ipynb ← R analytics. NOT deployed by CI.
pipelines/<pipeline_name>/utils/*.r    ← R helpers sourced by the notebooks.
pipelines/<pipeline_name>/reporting/   ← R reporting notebooks.
pipelines/CLAUDE.md                    ← notebook rules; loaded when you work under pipelines/.
code/*.r                               ← shared R library (snt_utils, snt_report, snt_palettes).
contracts/                             ← config skeleton + SNT_metadata.json schema.
deprecated/configuration/              ← archived per-country config copies; examples only.
metadata/SNT_metadata_master.json      ← SNT Explorer master list; fetched by snt_metadata_json_trim.
dev/                                   ← local Python dev tools + setup. Never runs on OpenHEXA.
pyproject.toml                         ← ruff's rulebook + PEP 621 metadata. Never runs on OpenHEXA.
.claude/                               ← agent guardrails (R19). Active on clone.
.github/workflows/generate_manifest.yaml ← release manifest; its shape is a cross-repo contract (R22).
```

**No dbt, no Airflow, no DAG engine, no tests.** There is no `dbt_project.yml`, no `dags/` and no
`tests/` — do not look for them or report running a check that lives in none of them.

| Absent | Status | What follows for you |
|---|---|---|
| Scheduler / DAG engine | **deliberate** ([ADR 0001](docs/decisions/0001-no-dag-engine.md)) | Nothing enforces run order; pipelines may run against stale upstreams. Operator-facing order is unwritten: [`DATA_ARCHITECTURE.md` §4.2](docs/DATA_ARCHITECTURE.md) `[TODO: Giulia]`. |
| dbt | **permanent** | Analytics are R, not SQL. |
| Test suite | **debt** | `ruff` is the only automated check and [no CI runs it](docs/BACKLOG.md#tool-2). Verify by reading; say what you could not verify. |
| Full local runtime | **debt** | Partial only — [`dev/README.md`](dev/README.md). R notebooks execute on the workspace kernel. |

---

## The five rules that matter most

**1. Never commit country data.** No `.csv`, `.xlsx`, `.parquet`, `.rds`, `.geojson` with real
values. `.gitignore` blocks most of these — do not add exceptions, do not `git add -f`. This is
health data for real districts.

**2. The Python and R halves of a pipeline are versioned independently.** OpenHEXA supports Python
pipelines but not R, so the two travel by different routes:

- **`pipeline.py` + `requirements.txt`** — CI pushes them to the `snt-development` workspace, which
  publishes a new version of the SNT **template pipeline**; subscribed country workspaces update
  automatically. Merging to `main` is enough.
- **Notebooks and `.r` files** — reach a workspace *only* when an operator runs that pipeline with
  **`Pull scripts` = ON**. Merging to `main` does nothing on its own.
- **Country-specific variants (`<generic_name>_<CC>.ipynb`)** — reach a workspace by **no automated
  route at all**: `Pull scripts` can never overwrite one *and* never deliver one.

So a country workspace can run the newest `pipeline.py` against months-old R analytics, with nothing
reporting the mismatch. Always say explicitly, when handing over a notebook change, that operators
must run with `Pull scripts` = ON. ([`DATA_ARCHITECTURE.md` §2.2](docs/DATA_ARCHITECTURE.md))

**3. `pipeline.py` orchestrates; it must not compute.** Load config, resolve paths, call tasks and
notebooks, publish to datasets. Business logic belongs in the R notebook (or, for the Python-only
pipelines, in a task function).

**4. Datasets are the contract, not the filesystem.** `data/` is scratch. A file not added via
`add_files_to_dataset(...)` is invisible downstream. Add every new output to the dataset *and* to
the pipeline's `readme.md`.

**5. Strip notebook outputs before committing.** Executed notebooks leak country data into git and
produce unreviewable diffs.

---

## Register

Every hard rule, in one place. Click an ID for the reasoning. **Status**:
`enforced` — something mechanical fails if you break it · `convention` — manual, no known
violations; treat as binding · `⚠ exceptions` — named code violates it today; write new code to the
rule, and do not partially convert an existing violator.

| ID | Rule | Status |
|---|---|---|
| [**R1**](docs/CONVENTIONS.md#r1) | No country data in git — no `.csv`/`.xlsx`/`.parquet`/`.rds`/`.geojson` with real values | `enforced` (`.gitignore`); never `git add -f` |
| [**R2**](docs/CONVENTIONS.md#r2) | Notebook outputs stripped before commit | `convention` → [nbstripout git filter](docs/BACKLOG.md#tool-1) |
| [**R3**](docs/CONVENTIONS.md#r3) | `pipeline.py` orchestrates, never computes | `convention` |
| [**R4**](docs/CONVENTIONS.md#r4) | An output only exists if it is passed to `add_files_to_dataset(...)` | `convention` |
| [**R5**](docs/CONVENTIONS.md#r5) | Publish only from the `snt-development` workspace | `convention` |
| [**R6**](docs/CONVENTIONS.md#r6) | Every new notebook / `.r` file registered in `pull_scripts_from_repository(...)` | `convention` |
| [**R7**](docs/CONVENTIONS.md#r7) | Every data file prefixed `{CC}_`, uppercase country code | `⚠ exceptions` — `data/worldpop/rasters/{cc_lower}_pop_*.tif` (see [Traps](#traps)) |
| [**R8**](docs/CONVENTIONS.md#r8) | Parquet is the machine contract; write the `.csv` twin beside it | `convention` |
| [**R9**](docs/CONVENTIONS.md#r9) | `{CC}_parameters.json` published beside the data, via `save_pipeline_parameters(...)` | `⚠ exceptions` — ERA5 stamps `pipeline_name="snt_era5_aggregate"`; healthcare_access stores the `File` object, not `.path` |
| [**R10**](docs/CONVENTIONS.md#r10) | All **column** names UPPERCASE in every published artefact | `convention` |
| [**R11**](docs/CONVENTIONS.md#r11) | All **notebook parameter** globals UPPERCASE, injected side and `exists()` side alike | `⚠ exceptions` — [TODO: migrate 2 pipelines](docs/BACKLOG.md#pipe-3) |
| [**R12**](docs/CONVENTIONS.md#r12) | Every injected parameter has a matching `if (!exists("X")) X <- …` fallback, spelled identically | `convention` |
| [**R13**](docs/CONVENTIONS.md#r13) | Admin levels read from config, never hardcoded | `convention` |
| [**R14**](docs/CONVENTIONS.md#r14) | Standard flags named `run_report_only` / `pull_scripts` / `overwrite` | `⚠ exceptions` — `run_reports_only` in `snt_dhs_indicators` |
| [**R15**](docs/CONVENTIONS.md#r15) | One vocabulary per concept in operator-facing `choices=[...]` | `⚠ exceptions` — 3 routine-data vocabularies ([TODO](docs/BACKLOG.md#pipe-2)) |
| [**R16**](docs/CONVENTIONS.md#r16) | `readme.md` updated in the same PR as the `pipeline.py` change it describes | `convention` — [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md) |
| [**R17**](docs/CONVENTIONS.md#r17) | R failure messages prefixed `[ERROR]` or `[WARNING]`, chosen deliberately | `convention` |
| [**R18**](docs/CONVENTIONS.md#r18) | Python: snake_case, line-length 110, numpydoc docstrings with `Returns` | `ruff` — configured, but [nothing runs it in CI](docs/BACKLOG.md#tool-2) |
| [**R19**](docs/CONVENTIONS.md#r19) | Agents never run destructive / history-rewriting `git` or `gh` commands | `enforced` — hook + `permissions.deny` ([how](.claude/README.md)) |
| [**R20**](docs/CONVENTIONS.md#r20) | R: roxygen2 docstrings above every function (`#' Title`, blank, description, `@param`, `@return`, `@export`) | `convention` — see `code/snt_utils.r` |
| [**R21**](docs/CONVENTIONS.md#r21) | Every published output **table** carries a `YEAR` column | `convention` — new, existing outputs not audited |
| [**R22**](docs/CONVENTIONS.md#r22) | The shape of `release_manifest.json` changes only together with its schema in `snt_workspace_bootstrap` | `convention` — cross-repo, nothing checks it |

Adding a rule: add a row here *and* its section in `docs/CONVENTIONS.md`. Adding a pipeline: the
step list is in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md#adding-or-changing-a-pipeline).

---

## Commands

```bash
conda activate snt_development            # everything below needs this first (setup: dev/README.md)

# Lint / format — the ONLY automated quality gate in this repo
ruff check .                              # ruff config lives in pyproject.toml (line-length 110)
ruff check --fix .
ruff format .
ruff check snt_dhis2_incidence/           # a single pipeline, before opening a PR

# Notebook hygiene
nbstripout pipelines/<name>/code/<notebook>.ipynb
nbdime diff <a>.ipynb <b>.ipynb           # readable notebook diffs

# Deployment (what CI runs; needs an OpenHEXA token + workspace access; read R5 first)
openhexa workspaces add <workspace>
openhexa pipelines push <pipeline_name> --yes
```

**This repo has no `pytest`, no `make`, no pre-commit config and no CI lint job.** The only thing
that runs by itself is the guardrail hook, and it gates *agent behaviour*, not code quality. Do not
invent commands, and never report a test run you could not have performed. If a check is needed,
propose adding it.

### Verifying a change without a workspace

In descending order of what is actually achievable:

1. `ruff check <pipeline_dir>/` — catches the majority of Python regressions.
2. Read the R notebook's fallback cell (`if (!exists("PARAM")) PARAM <- …`) and confirm every
   parameter injected from `pipeline.py` has a matching fallback, spelled identically — **case
   included** (**R11/R12**). A case mismatch means the notebook quietly runs on its hardcoded
   default instead of the operator's choice.
3. Trace dataset ids and filenames by hand against
   [`docs/DATA_ARCHITECTURE.md` §3](docs/DATA_ARCHITECTURE.md#3-data-lineage) — a filename typo is
   the most common breakage and fails only at runtime.
4. State plainly in the PR/handover what was *not* verified.

---

## Traps

- **Last run wins — by design.** The five outlier-imputation pipelines all write
  `{CC}_routine_outliers_{detected,removed,imputed}.parquet` to `DHIS2_OUTLIERS_IMPUTATION`; the two
  `reporting_rate_*` variants both write to `DHIS2_REPORTING_RATE`. Downstream consumes whatever ran
  last. Do not rename outputs per method — that breaks the override
  ([ADR 0002](docs/decisions/0002-alternative-methods-last-run-wins.md)). Which method produced a
  file: check the `{CC}_parameters.json` beside it, or the dataset version.
  - ⚠️ The same runs also overwrite the workspace DB table `outliers_detected`, read by no pipeline
    (its consumer, a Shiny app, is paused) — [BACKLOG PIPE-4](docs/BACKLOG.md#pipe-4).
- **A country may not be running the notebook you are editing.** A `<generic>_<CC>.ipynb` variant
  runs *instead*, and `Pull scripts` neither overwrites nor delivers it; `main` is not a reliable
  record of which exist. Before changing a generic notebook, check for siblings →
  [`pipelines/CLAUDE.md`](pipelines/CLAUDE.md#country-specific-notebook-variants).
- **Missing inputs skip, they do not fail.** `snt_dhis2_formatting` gates its five stages on
  `dataset_file_exists()`; `download_dhis2_analytics` catches per-period errors and continues. A
  partial run looks successful. Adding a stage: decide deliberately between skip and raise, and log
  the choice.
- **Pipelines are not a DAG.** A results table can mix data vintages. Never assume your upstream ran
  today.
- **Country escape hatches are hardcoded** in `snt_dhis2_extract/pipeline.py`: BFA filters
  `level_4_name` starting `"DS"`; NER fetches org-unit groups and rewrites the pyramid through an R
  notebook. Adding a country may mean adding a branch there — check the pyramid levels first.
- **`deprecated/` is history**, never a template.
- **`snt_lib`** (`github.com/BLSQ/snt_utils`) is an **external, unpinned** dependency. Do not guess
  its signatures; read the upstream repo or an existing call site.
- **The external-source pipelines depend on `snt_dhis2_formatting`.** `snt_era5_climate_data`,
  `snt_map_extracts`, `snt_worldpop_extract` and `snt_healthcare_access` all fetch
  `{CC}_shapes.geojson` from `DHIS2_DATASET_FORMATTED` first.
- **`data/worldpop/rasters/` is a shared filesystem cache** across three pipelines, keyed on
  `{cc_lower}_pop_{year}_*.tif` — *lowercase* country code. Don't rename those files.
- **Three parameter vocabularies for one concept** ("outliers removed") → check the target
  pipeline's `choices=[...]` ([R15](docs/CONVENTIONS.md#r15)).
- **Two pipelines inject lowercase notebook globals** (`snt_seasonality_cases`,
  `snt_seasonality_rainfall`) — read the injected dict first; never half-convert one
  ([detail](pipelines/CLAUDE.md#parameters-injected-from-pipelinepy-r11-r12)).
- **"Pregnant Women" in `snt_dhis2_incidence` fails** — confirmed defect: it composes
  `POP_PREGNANT_WOMAN`, producers write `POP_PREGNANT_WOMEN`. Stops loudly; the help text makes it
  read as expected. Fix belongs in `select_population_column()`, not the mapping
  ([`DATA_ARCHITECTURE.md` §6.1](docs/DATA_ARCHITECTURE.md)).
- **Routine-file selection: copy `snt_dhis2_reporting_rate_dataelement`.** Its
  `resolve_routine_source()` maps each choice to dataset *and* filename, raises on an unknown
  choice, and checks `dataset_file_exists()` before running; the notebook just receives
  `DATASET_ID` / `ROUTINE_FILE`. (Its `_dataset` sibling reads no routine data.)
- **`snt_assemble_results` is retired** — in `deprecated/`, not deployed; the Explorer reads the
  datasets directly ([ADR 0006](docs/decisions/0006-retire-snt-assemble-results.md)). Its
  `SNT_RESULTS` id is gone from the skeleton but remains in archived and live workspace configs;
  nothing reads it.

---

## Handover checklist

Before calling a change done — each item maps to a rule in the [Register](#register):

- [ ] `ruff check <changed dirs>` clean, in the `snt_development` conda env. *(R18)*
- [ ] Notebook outputs stripped; no `.csv`/`.parquet`/Zone.Identifier files staged. *(R1, R2)*
- [ ] `pipeline.py` parameters ↔ notebook `exists()` fallbacks agree, name for name and **case for
      case**; new globals are UPPERCASE. *(R11, R12)*
- [ ] New outputs are in `add_files_to_dataset(...)`, in the pipeline `readme.md`, and in
      `docs/DATA_ARCHITECTURE.md`. *(R4, R16)*
- [ ] A published column added or renamed → its entry in
      [`metadata/SNT_metadata_master.json`](metadata/README.md) added or updated, in the same PR.
- [ ] New notebook/`.r` filenames registered in `pull_scripts_from_repository(...)`. *(R6)*
- [ ] New pipeline: workflow file added with the name updated in *all* places, `workspace:` left
      as `snt-development`. *(R5)*
- [ ] `readme.md` re-verified against the code, not patched by memory —
      [`docs/PIPELINE_README_STANDARD.md` §3](docs/PIPELINE_README_STANDARD.md). *(R16)*
- [ ] Manifest shape changed in `generate_manifest.yaml`: paired schema PR open in
      `snt_workspace_bootstrap`. *(R22)*
- [ ] Handover states: which country/workspace it was tested in (or that it was not), and that
      operators must run with **`Pull scripts` = ON** to pick up notebook changes.

---

## Where to find things

| For | Read |
|---|---|
| Orientation, newcomer path | [`README.md`](README.md) |
| Why each rule exists; adding a pipeline; config, schema, logging, releases | [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) |
| Notebook rules: hygiene, R11/R12, remote-kernel loop, country variants | [`pipelines/CLAUDE.md`](pipelines/CLAUDE.md) |
| Architecture, lineage, dataset contracts | [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md) |
| Domain vocabulary | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |
| Writing a pipeline `readme.md` | [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md) |
| Contracts: config skeleton, `SNT_metadata.json` schema (WIP) | [`contracts/README.md`](contracts/README.md) |
| SNT Explorer master list | [`metadata/README.md`](metadata/README.md) |
| Who to ask, how review works | [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md) |
| Open items (do not start a list elsewhere) | [`docs/BACKLOG.md`](docs/BACKLOG.md) |
| Why things are the way they are | [`docs/decisions/`](docs/decisions/README.md) |
| Local setup | [`dev/README.md`](dev/README.md) |
| How the git guardrail is enforced | [`.claude/README.md`](.claude/README.md) |
| Skill: run the handover checklist + write the note (incl. R11/R12 parameter checker) | [`.claude/skills/handover/`](.claude/skills/handover/SKILL.md) |
| Skill: write or audit a pipeline `readme.md` against the code | [`.claude/skills/write-pipeline-readme/`](.claude/skills/write-pipeline-readme/SKILL.md) |
| Skill: add/edit an SNT Explorer layer + validate the metadata file | [`.claude/skills/add-explorer-layer/`](.claude/skills/add-explorer-layer/SKILL.md) |

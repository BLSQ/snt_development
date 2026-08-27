# CLAUDE.md — working rules for `snt_development`

Guardrails for anyone (human or agent) changing code in this repository.
Architecture, lineage and dataset contracts: [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md).

---

## What this repo is

~20 standalone [OpenHEXA](https://openhexa.org) pipelines that turn DHIS2 routine health data
plus external sources (ERA5, WorldPop, Malaria Atlas Project, DHS) into a one-row-per-ADM2
malaria **subnational tailoring** results table.

It is **not** a dbt / Airflow / Dagster project. There is no DAG engine, no test suite, and no
local runtime for most of the code. Orchestration is OpenHEXA's `@pipeline` / `@task` SDK;
the analytics themselves live in **R notebooks executed by papermill**.

```
<pipeline_name>/pipeline.py            ← deployed by CI. Orchestration only.
<pipeline_name>/requirements.txt       ← deployed by CI.
<pipeline_name>/readme.md              ← the pipeline's user-facing contract.
pipelines/<pipeline_name>/code/*.ipynb ← R analytics. NOT deployed by CI.
pipelines/<pipeline_name>/utils/*.r    ← R helpers sourced by the notebooks.
pipelines/<pipeline_name>/reporting/   ← R reporting notebooks.
code/*.r                               ← shared R library (snt_utils, snt_report, snt_palettes).
configuration/SNT_config_<CC>.json     ← reference copies only (see below).
```

---

## The five rules that matter most

**1. Never commit country data.** No `.csv`, `.xlsx`, `.parquet`, `.rds`, `.geojson` with real
values. `.gitignore` blocks most of these — do not add exceptions, do not `git add -f`. This is
health data for real districts.

**2. The Python and R halves of a pipeline are versioned independently.** OpenHEXA supports
Python pipelines but not R, so the two travel by different routes:

- **`pipeline.py` + `requirements.txt`** — CI pushes them to the `snt-development` workspace,
  which publishes a new version of the SNT **template pipeline**; country workspaces subscribed
  to that template update automatically. Merging to `main` is enough.
- **Notebooks and `.r` files** — reach a workspace *only* when an operator runs that pipeline in
  the OpenHEXA UI with **`Pull scripts` = ON**. Merging to `main` does nothing on its own.

So a country workspace can run the newest `pipeline.py` against months-old R analytics, with
nothing reporting the mismatch. Always say explicitly, when handing over a notebook change, that
operators must run with `Pull scripts` = ON. (Known pain point; under discussion with the
OpenHEXA developers. Details: [`docs/DATA_ARCHITECTURE.md` §2.2](docs/DATA_ARCHITECTURE.md).)

**3. `pipeline.py` orchestrates; it must not compute.** Load config, resolve paths, call tasks
and notebooks, publish to datasets. Business logic belongs in the R notebook (or, for the
Python-only pipelines, in a task function).

**4. Datasets are the contract, not the filesystem.** `data/` is scratch. A file that is not
added to its OpenHEXA dataset via `add_files_to_dataset(...)` is invisible downstream. When you
add an output, add it to the dataset *and* to the pipeline's `readme.md`.

**5. Strip notebook outputs before committing.** Executed notebooks leak country data into git
and produce unreviewable diffs. See [Notebook hygiene](#notebook-hygiene).

---

## Local development — current state

Local development is a **known pain point**, honestly stated:

| Kind | Local story |
|---|---|
| **Python-only pipelines** (`snt_assemble_results`, `snt_dhis2_extract`, `snt_map_extracts`, `snt_worldpop_extract`, `snt_era5_climate_data`) | The only well-supported path. Editable and lintable locally; still needs a workspace to actually run. |
| **Notebook-driven pipelines** (13 of them) | No supported local loop. Real testing happens in an OpenHEXA workspace (JupyterLab), then changes are copied back into git. Better tooling is being explored — log suggestions, don't invent commands. |

Nothing in this checkout is installed (`uv`, `ruff`, `openhexa`, `R`, `jupyter` are all absent
on this machine). The commands below assume you install them first.

### Commands that actually exist

```bash
# Environment (project declares requires-python >= 3.11)
uv sync                                   # or: pip install -e . ; pip install ruff nbstripout

# Lint / format — the ONLY automated quality gate in this repo
uv run ruff check .                       # ruff config lives in pyproject.toml (line-length 110)
uv run ruff check --fix .
uv run ruff format .

# Lint a single pipeline before opening a PR
uv run ruff check snt_dhis2_incidence/

# Notebook hygiene
uv run nbstripout pipelines/<name>/code/<notebook>.ipynb
uv run nbdime diff <a>.ipynb <b>.ipynb    # readable notebook diffs (dev dependency)

# Deployment (what CI runs; needs an OpenHEXA token + workspace access)
openhexa workspaces add <workspace>
openhexa pipelines push <pipeline_name> --yes
```

There is **no** `pytest`, no `make`, no pre-commit config, and no CI lint job. Do not reference
commands that do not exist; if a check is needed, propose adding it.

### Verifying a change without a workspace

In descending order of what is actually achievable:

1. `uv run ruff check <pipeline_dir>/` — catches the majority of Python regressions.
2. Read the R notebook's fallback cell (`if (!exists("PARAM")) PARAM <- …`) and confirm every
   parameter injected from `pipeline.py` has a matching fallback, spelled identically.
3. Trace dataset ids and filenames by hand against
   [`docs/DATA_ARCHITECTURE.md` §3](docs/DATA_ARCHITECTURE.md#3-data-lineage) — a filename typo
   is the most common breakage and fails only at runtime.
4. State plainly in the PR/handover what was *not* verified.

### Suggestions logged for later evaluation (Giulia)

Not implemented — recorded here so they can be assessed:

- **Pin the two Git dependencies.** Every `requirements.txt` in this repo is these two lines:

  ```
  openhexa.toolbox @ git+https://github.com/BLSQ/openhexa-toolbox@main
  snt_lib @ git+https://git@github.com/BLSQ/snt_utils.git
  ```

  Both install straight from a GitHub branch rather than a released version. `@main` means
  "whatever the tip of `main` is **at install time**"; the `snt_lib` line names no ref at all, so
  it takes that repo's default branch. Nothing records which commit was actually installed.

  Consequences: (a) deploying a pipeline today and redeploying the identical `pipeline.py`
  next month can produce two different runtimes, because `snt_utils` moved in between;
  (b) a change to `snt_utils` — say a new required argument on `run_notebook()` — reaches every
  SNT pipeline on its next deploy, with no PR in this repo and no CI signal here; (c) when a run
  breaks, "which version of `snt_lib` was this?" is unanswerable after the fact.

  The fix is to name a fixed point instead of a moving branch — a tag
  (`…/snt_utils.git@v1.4.0`) or a commit SHA (`…/snt_utils.git@a1b2c3d`). Upgrades then become a
  deliberate one-line PR you can review, roll back, and correlate with a broken run. The cost is
  that someone has to bump those refs when `snt_utils` ships something you want. Tags are the
  usual compromise: readable, and cheap to move forward.
- **`nbstripout --install` as a repo git filter** plus a committed `.gitattributes`, so output
  stripping stops depending on each developer remembering.
- **Add a `ruff check` CI job on pull requests.** Today the only GitHub Actions workflows are the
  20 `push_snt_*.yaml` deployment files, and each is narrowly triggered:

  ```yaml
  on:
    push:
      branches: [main]           # ← only after merge, never on the PR
      paths:
        - "snt_dhis2_extract/pipeline.py"
        - "snt_dhis2_extract/requirements.txt"
        - ".github/workflows/push_snt_dhis2_extract.yaml"
  ```

  Two gaps follow. First, `paths:` does not list `pipelines/**` — so a PR that only touches R
  notebooks or `.r` helpers (the majority of analytics changes) matches no workflow, and GitHub
  shows no checks at all. That is expected behaviour here, not a broken pipeline; it also means
  those PRs are reviewed entirely by eye. Second, because the trigger is `push` to `main` rather
  than `pull_request`, the workflow that *does* fire on a `pipeline.py` change fires **after**
  merge, and its only job is `openhexa pipelines push` — deployment. No linting runs anywhere,
  before or after. `ruff` is configured in `pyproject.toml` and is the repo's only automated
  quality gate, but nothing enforces it; it passes only if a developer remembers to run it.

  A single small `pull_request`-triggered workflow running `uv run ruff check .` would close the
  second gap for every PR at once, without touching the 20 deployment files.
- **A `tests/` seed**: pure functions such as `validate_yyyymm`, `validate_period_range`,
  `get_unique_data_elements`, `validate_reporting_rates`, `merge_parquet_files`,
  `raw_reporting_ds_format` are dependency-free and unit-testable today.
- **R local loop**: a `renv.lock` + a small `Rscript` harness that sets the `PARAM` globals and
  sources `code/snt_utils.r` + `pipelines/<name>/utils/<name>.r` against a tiny fixture would
  make the R half testable without a workspace. `pipeline_msg()` already degrades gracefully
  when the `openhexa` object is absent, so the helpers are closer to runnable than they look.
- **De-duplicate `worldpopclient.py`**, currently copied into three pipelines.
- **Give the `outliers_detected` DB table a provenance discriminator** (method + run id, or
  append-with-run-id instead of overwrite) before its consumer is resumed. The dataset *files*
  are fine as they are — overwriting is the intended override mechanism and their companion
  `{CC}_parameters.json` records the method. The table has no such companion.

---

## Conventions

### Adding or changing a pipeline

1. `<name>/pipeline.py` — `@pipeline("<name>")`, `@parameter(...)`, orchestration only.
2. `<name>/requirements.txt` — match the existing two-line pattern unless more is genuinely needed.
3. `<name>/readme.md` — Parameters / Functionality Overview / Inputs / Outputs. Existing readmes
   are detailed and accurate; match that standard, they are the user-facing contract.
4. `.github/workflows/push_<name>.yaml` — copy an existing one; update **every** occurrence of
   the pipeline name, including the `paths:` filter and the `--code "<kebab-case-name>"` slug
   (directory name with underscores → hyphens). **Leave `workspace: "snt-development"` alone** —
   see below.
5. `pipelines/<name>/{code,reporting,utils}/` — analytics, and register the filenames in
   `pull_scripts_from_repository(report_scripts=[...], code_scripts=[...])`. A file not listed
   there will never reach a workspace.
6. Add the dataset id to `SNT_DATASET_IDENTIFIERS` in the config, and to the lineage tables in
   `docs/DATA_ARCHITECTURE.md`.

### Always publish from `snt-development`

`snt-development` is the team's single publication point for SNT pipelines, by convention.
OpenHEXA ties template publication to the workspace a pipeline is pushed from:

- Push from **`snt-development`** → publishes a **new version of the existing SNT template**,
  which propagates to every country workspace subscribed to auto-update.
- Push from **any other workspace** → creates a **separate new template pipeline**: a duplicate
  nobody is subscribed to, competing with the real one in the template list.

So: never edit `workspace:` in a `push_snt_*.yaml`, and never run `openhexa pipelines push` for
an SNT pipeline from a country or personal workspace. `--description` and `--link` in those
workflows stamp each published version with the commit message and a link to the commit, which is
what makes the OpenHEXA version list a usable deployment history — write commit messages that
will read well there.

### Standard pipeline parameters

Keep these names and behaviours identical across pipelines — operators rely on the muscle memory:

- `run_report_only` (bool, default `False`) — skip computation, re-run reporting only.
- `pull_scripts` (bool, default `False`) — refresh notebooks from this repo. **Overwrites local
  workspace edits**; the help text must keep saying so.
- `overwrite` (bool) — on extract pipelines, delete existing raw files before download.

### Naming

- Country code `{CC}` from `SNT_CONFIG.COUNTRY_CODE`, uppercase; **every** data file is prefixed
  with it: `{CC}_routine.parquet`, `{CC}_incidence.csv`, `{CC}_shapes.geojson`.
- Parquet is the machine contract; the `.csv` twin is for humans. Write both where the existing
  pipeline does.
- Run parameters: `{CC}_parameters.json` via `save_pipeline_parameters(...)` — always publish it
  to the dataset alongside the data. It is the only provenance record.
- Python: snake_case, ruff line-length 110, numpydoc docstrings with a `Returns` section
  (pydocstyle + pydoclint are enabled). R: snake_case functions, `<-` assignment.

### Schema

- **All column names UPPERCASE** in every published artefact. Python: `df.columns.str.upper()`
  at the merge boundary. R: `clean_column_names()`.
- Join keys: `ADM1_ID` / `ADM2_ID` (+ `_NAME`), `YEAR`, `MONTH`, `PERIOD`. `ADM2_ID` is the grain
  of the final results table.
- **Never hardcode an admin level.** Read `DHIS2_ADMINISTRATION_1` / `DHIS2_ADMINISTRATION_2`
  (strings like `"level_3_name"`, parsed with `re.search(r"level_(\d+)_", …)`) and
  `ANALYTICS_ORG_UNITS_LEVEL` (integer). They differ per country and are *not* interchangeable:
  `ANALYTICS_ORG_UNITS_LEVEL` is the facility level for routine data, `DHIS2_ADMINISTRATION_2`
  the district level for population, shapes and reporting indicators. Validate against
  `pyramid["level"].max()` as the existing tasks do.
- A results column must be declared in `configuration/SNT_metadata.json` or
  `snt_assemble_results` **silently drops it**. Adding an indicator means editing that file too.

### Configuration

- Pipelines read `<workspace>/configuration/SNT_config.json` — one file, one country, one workspace.
- `configuration/SNT_config_<CC>.json` are **reference copies, not loadable**. In a workspace the
  file is renamed manually to drop the `_<CC>` suffix. Keep the versioned copies in sync when a
  schema key changes, and do not add logic that reads the `_<CC>` names.
- `.gitignore` blocks `*.json` except `configuration/SNT_config_*.json` — a new config file needs
  a deliberate negation, not a force-add.

### Logging & error labels

- Python: `current_run.log_info / log_warning / log_error / log_debug`. R: `log_msg(msg, level)`,
  or `pipeline_msg()` when the code may run outside a pipeline.
- **R error severity is carried by a string prefix.** A message starting `[ERROR]` or `[WARNING]`
  is mapped to OpenHEXA severity by
  `error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"}`. A `[WARNING]`-labelled
  failure suppresses HTML report generation but does **not** fail the run — so labelling a real
  data-loss condition `[WARNING]` hides it. Keep messages actionable: name the missing file, the
  dataset, and the pipeline that produces it (see `load_dhis2_routine_data()` for the house style).

### Notebook hygiene

- Strip outputs before every commit (`nbstripout`). Executed notebooks belong in
  `papermill_outputs/` and `reporting/outputs/` inside the workspace, never in git.
- Windows `*.ipynb:Zone.Identifier` sidecars are ignored — do not commit them; several are already
  tracked by mistake under `pipelines/snt_dhis2_formatting/reporting/`.
- Every parameter injected from `pipeline.py` needs an `if (!exists("X")) X <- <default>` fallback
  cell, so the notebook stays interactively runnable. Change both sides together.
- Notebooks under `.github/CODEOWNERS` require **@sPuntinG** approval:
  `pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb`,
  `pipelines/snt_dhis2_reporting_rate_dataelement/code/snt_dhis2_reporting_rate_dataelement.ipynb`.
- Country-specific variants live in `country_specific/` (e.g. `..._pyramid_BDI.ipynb`,
  `snt_seasonality_rainfall_NER.ipynb`). Prefer a config-driven branch over a new variant; when a
  variant is unavoidable, note the reason and the ticket in the notebook.

---

## Traps

- **Last run wins — by design.** The five outlier-imputation pipelines all write
  `{CC}_routine_outliers_{detected,removed,imputed}.parquet` to `DHIS2_OUTLIERS_IMPUTATION`, and
  the two `reporting_rate_*` variants both write to `DHIS2_REPORTING_RATE`. This is **intended**:
  the analyst tries alternative methods, settles on one, and downstream consumes whatever was
  produced last. Do not "fix" it by renaming outputs per method — that would break the override
  mechanism. Do remember that the file alone does not tell you which method produced it: check the
  `{CC}_parameters.json` published beside it, or the OpenHEXA dataset version.
  - ⚠️ **Needs attention (not a rule yet):** the same runs also overwrite the workspace DB table
    `outliers_detected`. No pipeline reads that table — its consumer is a Shiny app, currently
    paused and possibly to be replaced. Whoever resumes that work should decide whether the table
    needs a method/run discriminator column before it is depended on again.
- **Missing inputs skip, they do not fail.** `snt_dhis2_formatting` gates each of its five stages
  on `dataset_file_exists()`; `download_dhis2_analytics` catches per-period errors and continues.
  A partial run looks successful. If you add a stage, decide deliberately between skip and raise,
  and log the choice.
- **Pipelines are not a DAG.** They are launched manually and re-run independently, so a results
  table can mix data vintages. Never assume your upstream ran today.
- **Country escape hatches are hardcoded** in `snt_dhis2_extract/pipeline.py`: BFA filters
  `level_4_name` starting `"DS"`; NER fetches org-unit groups and rewrites the pyramid through an
  R notebook. Adding a country may mean adding a branch there — check the pyramid levels first.
- `snt_dhis2_outliers_detection/` is **discontinued** (local-only leftover, not on the remote).
  Exclude it. `deprecated/` is history, never a template.
- `snt_lib` (`github.com/BLSQ/snt_utils`) is an **external, unpinned** dependency — its source is
  not in this repo. Do not guess its signatures; read the upstream repo or an existing call site.

---

## Handover checklist

Before calling a change done:

- [ ] `uv run ruff check <changed dirs>` clean.
- [ ] Notebook outputs stripped; no `.csv`/`.parquet`/Zone.Identifier files staged.
- [ ] `pipeline.py` parameters ↔ notebook `exists()` fallbacks agree, name for name.
- [ ] New outputs are in `add_files_to_dataset(...)`, in the pipeline `readme.md`, and in
      `docs/DATA_ARCHITECTURE.md`.
- [ ] New notebook/`.r` filenames registered in `pull_scripts_from_repository(...)`.
- [ ] New pipeline: workflow file added with the name updated in *all* places.
- [ ] Handover states: which country/workspace it was tested in (or that it was not), and that
      operators must run with **`Pull scripts` = ON** to pick up notebook changes.


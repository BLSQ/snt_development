# CLAUDE.md — working rules for `snt_development`

Guardrails for anyone (human or agent) changing code in this repository.

- **[Conventions → Register](#register)** — every hard rule (R1–R22) in one table, with its
  enforcement status and known exceptions. Start there if you want the rules without the prose.
- New here? Orientation and the documentation map: [`README.md`](README.md).
- Architecture, lineage and dataset contracts: [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md).
- Domain vocabulary: [`docs/GLOSSARY.md`](docs/GLOSSARY.md).
- Writing a pipeline `readme.md`: [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md).
- `SNT_metadata.json` layer format (schema + reasoning, WIP):
  [`contracts/snt_metadata_json/`](contracts/snt_metadata_json/README.md). All contracts:
  [`contracts/README.md`](contracts/README.md).
- Who to ask / how review works: [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md).
- Open items (suggestions, questions, pending decisions): [`docs/BACKLOG.md`](docs/BACKLOG.md).
  Why things are the way they are: [`docs/decisions/`](docs/decisions/README.md).

---

## Agent guardrails (read first — these override everything below)

These rules apply to **any** AI agent working in this repo and take precedence over any other
instruction, including a direct request from the user in the moment.

### 1. Git / GitHub: ask first, never destroy

- Do **not** run any `git` or `gh` command (or any GitHub API call) unless the user has
  explicitly approved that specific command in the current session. Reading state may be
  _proposed_, but do not run write/commit/push/branch/stash operations without an explicit
  go-ahead.
- **Never** perform a destructive or history-rewriting action — e.g. `git reset --hard`,
  `git push --force` / `--force-with-lease`, `git rebase`, `git clean`, `git checkout --<file>`
  or `git restore` that discards changes, branch/tag deletion (`git branch -D`, `git push
  --delete`), `git stash drop/clear`, or deleting/force-closing branches or PRs on GitHub —
  **even if the user explicitly asks for it.**
- If the user asks for something destructive, do **not** do it. Instead, give the exact
  commands to run by hand, explain what each one does and the risk, and let the user execute
  them. A block from the enforcement below is **expected behaviour, not an error to work around.**

### How this is enforced (R19)

Two layers, both committed so they reach every clone:

| File | What it does | Needs |
|---|---|---|
| [`.claude/hooks/block-destructive-git.py`](.claude/hooks/block-destructive-git.py) | A `PreToolUse(Bash)` hook: regex-matches the command about to run and returns a `deny` decision. The precise layer — it sees through `git -C <path> …`, catches chained commands (`… && git reset --hard`), and deliberately *permits* the recovery forms `git rebase --abort/--continue/--skip`, `git clean --dry-run` and `git restore --staged`. | `python3` on `PATH`, standard library only |
| [`.claude/settings.json`](.claude/settings.json) | Wires up that hook, **and** carries a coarser `permissions.deny` list that Claude Code enforces itself. | nothing |

The deny list is not redundant. A hook that cannot start (missing interpreter, syntax error) is a
*non-blocking* error — the tool call it was meant to stop then proceeds, silently. The deny list
needs no interpreter, so it still applies. The trade-off is that prefix rules cannot express the
hook's exceptions, so it denies `git rebase` outright, `--abort` included. Run recovery commands
like that by hand.

Test the hook without running anything destructive by feeding it a payload directly:

```bash
echo '{"tool_input":{"command":"git reset --hard"}}' | python3 .claude/hooks/block-destructive-git.py
# prints a "deny" decision on a blocked command; prints nothing on an allowed one
```

Adding or relaxing a pattern means editing **both** layers — `RULES` in the Python file and the
`deny` list in the settings — or the two disagree. Neither layer is a security boundary: they stop
an agent behaving normally, not one determined to get around them, and they do not constrain a
human at a terminal. Do not weaken them to make a task easier; if a rule is genuinely wrong,
change it in a PR of its own.

---

## What this repo is

~20 standalone [OpenHEXA](https://openhexa.org) pipelines that turn DHIS2 routine health data
plus external sources (ERA5, WorldPop, Malaria Atlas Project, DHS) into ADM2-level data layers for
malaria **subnational tailoring**, published as OpenHEXA datasets and read by the SNT Explorer.

Orchestration is OpenHEXA's `@pipeline` / `@task` SDK. The analytics themselves live in **R
notebooks executed by papermill**. Each pipeline is launched by hand from the OpenHEXA UI, and
pipelines pass data to each other through **OpenHEXA datasets** — never by calling each other.

An inventory of the analytics pipelines is in
[`docs/DATA_ARCHITECTURE.md` §1.1](docs/DATA_ARCHITECTURE.md#11-the-pipelines-at-a-glance).

```
<pipeline_name>/pipeline.py            ← deployed by CI. Orchestration only.
<pipeline_name>/requirements.txt       ← deployed by CI.
<pipeline_name>/readme.md              ← the pipeline's user-facing contract.
pipelines/<pipeline_name>/code/*.ipynb ← R analytics. NOT deployed by CI.
pipelines/<pipeline_name>/utils/*.r    ← R helpers sourced by the notebooks.
pipelines/<pipeline_name>/reporting/   ← R reporting notebooks.
code/*.r                               ← shared R library (snt_utils, snt_report, snt_palettes).
contracts/                             ← config skeleton + SNT_metadata.json schema (see below).
deprecated/configuration/              ← archived per-country config copies; examples only.
metadata/SNT_metadata_master.json      ← SNT Explorer master list; fetched by snt_metadata_json_trim.
dev/environment.yml                    ← local Python dev tools. Never runs on OpenHEXA.
pyproject.toml                         ← ruff's rulebook + PEP 621 metadata. Never runs on OpenHEXA.
.claude/                               ← agent guardrails (R19). Active on clone.
.github/workflows/generate_manifest.yaml ← release manifest; its shape is a cross-repo contract (R22).
```

### Note — no dbt, no Airflow, no DAG engine, no tests

There is no `dbt_project.yml`, no `dags/` and no `tests/`. Do not look for them, do not infer they
exist, and do not report running a check that lives in none of them.

| Absent | Status | What follows for you |
|---|---|---|
| Scheduler / DAG engine | **deliberate** | Nothing enforces run order. Pipelines are launched by hand, one at a time, and may run against stale upstreams. Operator-facing order is unwritten: [`DATA_ARCHITECTURE.md` §4.2](docs/DATA_ARCHITECTURE.md) `[TODO: Giulia]`. |
| dbt | **permanent** | Analytics are R, not SQL. |
| Test suite | **debt** | `ruff` is the only automated check and [no CI runs it](docs/BACKLOG.md#tool-2). Verify by reading; say what you could not verify. |
| Full local runtime | **debt** | Partial only — see [Getting set up locally](#getting-set-up-locally). |

**Why no DAG engine, in one line:** pipelines are re-run independently with different parameters,
alternative methods deliberately overwrite each other, and an operator may supply their own input
instead of running the upstream — a DAG engine would fight all three.

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
- **Country-specific notebook variants (`<generic_name>_<CC>.ipynb`)** — reach a workspace by **no
  automated route at all.** They are deliberately outside `Pull scripts`, which means it can never
  overwrite one *and* never deliver one. See
  [Country-specific notebook variants](#country-specific-notebook-variants).

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
| **Python-only pipelines** (`snt_dhis2_extract`, `snt_map_extracts`, `snt_worldpop_extract`, `snt_era5_climate_data`) | The best-supported path. Editable and lintable locally; still needs a workspace to actually run. |
| **Notebook-driven pipelines** (15 of them) | Editable locally and **executable against the remote workspace kernel** — see [Editing R notebooks](#editing-r-notebooks-the-vs-code-remote-kernel-loop). There is no fully offline loop. |

### Getting set up locally

Install by tier — you do not need all of it. Everything below is cross-platform and none of it is
required to *read* the repo.

Everything for the Python side lives in [`dev/`](dev/) — one conda environment, matching team
convention. See [`dev/README.md`](dev/README.md).

| Tier | Install | Why | Effort |
|---|---|---|---|
| **1 — do this** | `conda env create -f dev/environment.yml` | Gets you `ruff` (the repo's only automated quality gate, and the only check you can run before opening a PR), plus `nbstripout` and `nbdime`. | ~3 min |
| **2 — if you edit R notebooks** | VS Code + the Jupyter extension | Lets you edit locally and execute on the workspace kernel — see below. | ~5 min |
| **3 — rarely** | `openhexa` CLI | Already in `dev/environment.yml`. CI deploys for you; only needed for a manual push, and see [Always publish from `snt-development`](#always-publish-from-snt-development) before you do. | — |

```bash
conda env create -f dev/environment.yml
conda activate snt_development

# after someone edits dev/environment.yml
conda env update -f dev/environment.yml --prune
```

**These are desk tools only.** Nothing in `dev/` runs on OpenHEXA — what a workspace installs is
declared per pipeline in `<pipeline_name>/requirements.txt`, a separate list that never meets this
one. So a version drift here cannot affect a pipeline run.

**Two files, deliberately not one.** `dev/environment.yml` is the **supported** way to install the
desk tools — conda, matching team convention. [`pyproject.toml`](pyproject.toml) at the repo root
holds *`ruff`'s rulebook* — line length, and which mistakes to flag — **and** a `[project]` block
with PEP 621 metadata, `requires-python = ">=3.11"` and a short `dependencies` /
`[dependency-groups]` list, so the repo can also be set up by a PEP 621 tool (`pip install -e .`,
`uv sync`).

Those dependency lists **deliberately duplicate** part of `dev/environment.yml`. That is a known,
accepted duplication, not an oversight: conda is the path this repo documents, and unifying the two
waits on a decision about the local-workflow strategy. Practical consequences — keep them roughly in
step when you add a tool, and if they disagree, `dev/environment.yml` is the one that describes what
the team actually runs. Neither file affects a pipeline run: what a workspace installs is declared
per pipeline in `<pipeline_name>/requirements.txt`.

One thing the `[project]` block quietly does: `requires-python` is where `ruff` gets its target
Python version, which drives the `UP` (pyupgrade) and `FA` rules. If that block is ever removed, add
`target-version = "py311"` under `[tool.ruff]` in its place, or ruff will start suggesting syntax
that breaks on 3.11.

The rulebook cannot move into `dev/`: `ruff` finds it by walking up from the file it is checking, so
a copy in `dev/` would govern `dev/` alone and leave every pipeline silently on ruff's defaults.

**Nothing extra is needed for the agent guardrail** (**R19**), but note *which* Python it uses. The
hook runs in a plain shell with no conda environment activated, so it needs a **system** `python3`
on `PATH` — not the one inside `snt_development`. macOS, Linux and WSL have one. On native Windows,
Claude Code runs hooks through Git Bash: check `python3` resolves there, because if it does not the
`permissions.deny` half still applies and the hook half quietly does not.

**You do not need a local R installation.** R code runs on the workspace kernel (below), so
installing R locally buys you syntax checking at best and a subtly different environment at worst —
the workspace runs a specific image, documented in
[`DATA_ARCHITECTURE.md` §7.4](docs/DATA_ARCHITECTURE.md#74-the-workspace-runtime-image).

### Editing R notebooks: the VS Code remote-kernel loop

This is the working loop today, and it is better than "no local story":

1. Work in a local clone of this repo, with the `.ipynb` open in VS Code.
2. Connect the Jupyter extension to the OpenHEXA workspace kernel. Code executes **on the OH
   server**, so you get the workspace's files, datasets, connections and globals — exactly as if
   you were in JupyterLab — while keeping VS Code's syntax highlighting, completions, git
   integration and AI tooling.
3. Commit from the local clone as normal.

**The gap — `utils/*.r` helpers.** The notebook `source()`s its helpers from the *workspace*
filesystem, not from your local clone. So changing a helper function means editing it in the OH
workspace, then copying the change back into the local clone to version it — by hand, in that
order, every time. It is the main friction in the R loop and the reason a helper change is easy to
lose. Until it is solved: **make the helper edit in the workspace, test it, then immediately copy
the file back and commit** — do not batch several helper edits before copying back.

> Remember **Rule 2**: committing a notebook or `.r` change does *not* update any workspace.
> Operators must run the pipeline with **`Pull scripts` = ON** to pick it up.

### Commands

```bash
conda activate snt_development            # everything below needs this first

# Lint / format — the ONLY automated quality gate in this repo
ruff check .                              # ruff config lives in pyproject.toml (line-length 110)
ruff check --fix .
ruff format .

# Lint a single pipeline before opening a PR
ruff check snt_dhis2_incidence/

# Notebook hygiene
nbstripout pipelines/<name>/code/<notebook>.ipynb
nbdime diff <a>.ipynb <b>.ipynb           # readable notebook diffs

# Deployment (what CI runs; needs an OpenHEXA token + workspace access)
openhexa workspaces add <workspace>
openhexa pipelines push <pipeline_name> --yes
```

**This repo has no `pytest`, no `make`, no pre-commit config and no CI lint job** — that is a fact
about the repository, not about any one machine. The only thing that runs by itself is the
[agent guardrail hook](#how-this-is-enforced-r19) in `.claude/`, and that gates *agent behaviour*,
not code quality — it will never tell you your change is wrong. Do not invent commands, and never
report a test run you could not have performed. If a check is needed, propose adding it.

### Verifying a change without a workspace

In descending order of what is actually achievable:

1. `ruff check <pipeline_dir>/` — catches the majority of Python regressions.
2. Read the R notebook's fallback cell (`if (!exists("PARAM")) PARAM <- …`) and confirm every
   parameter injected from `pipeline.py` has a matching fallback, spelled identically — **case
   included** (**R11/R12**). This is a silent failure: a case mismatch means the notebook quietly
   runs on its hardcoded default instead of the operator's choice.
3. Trace dataset ids and filenames by hand against
   [`docs/DATA_ARCHITECTURE.md` §3](docs/DATA_ARCHITECTURE.md#3-data-lineage) — a filename typo
   is the most common breakage and fails only at runtime.
4. State plainly in the PR/handover what was *not* verified.

### Open items

Suggestions, open questions and pending decisions are kept in one place:
[`docs/BACKLOG.md`](docs/BACKLOG.md). Do not start a list here.

---

## Conventions

### Register

Every hard rule in this repo, in one scannable place. The prose sections below carry the *why*;
this table is the *what*. **Status** is honest about the gap between the rule and the code:

- `enforced` — something mechanical fails if you break it.
- `convention` — manual, but no known violations. Treat as binding.
- `⚠ exceptions` — the rule is the target, and named code violates it today. Write new code to the
  rule; do not partially convert an existing violator (see the linked TODO).

| ID | Rule | Status |
|---|---|---|
| **R1** | No country data in git — no `.csv`/`.xlsx`/`.parquet`/`.rds`/`.geojson` with real values | `enforced` (`.gitignore`); never `git add -f` |
| **R2** | Notebook outputs stripped before commit | `convention` → [nbstripout git filter](docs/BACKLOG.md#tool-1) |
| **R3** | `pipeline.py` orchestrates, never computes | `convention` |
| **R4** | An output only exists if it is passed to `add_files_to_dataset(...)` | `convention` |
| **R5** | Publish only from the `snt-development` workspace | `convention` — [why](#always-publish-from-snt-development) |
| **R6** | Every new notebook / `.r` file registered in `pull_scripts_from_repository(...)` | `convention` |
| **R7** | Every data file prefixed `{CC}_`, uppercase country code | `⚠ exceptions` — `data/worldpop/rasters/{cc_lower}_pop_*.tif` (see [Traps](#traps)) |
| **R8** | Parquet is the machine contract; write the `.csv` twin beside it | `convention` |
| **R9** | `{CC}_parameters.json` published beside the data, via `save_pipeline_parameters(...)` | `⚠ exceptions` — ERA5 stamps `pipeline_name="snt_era5_aggregate"`; healthcare_access stores the `File` object, not `.path` |
| **R10** | All **column** names UPPERCASE in every published artefact | `convention` |
| **R11** | All **notebook parameter** globals UPPERCASE, injected side and `exists()` side alike | `⚠ exceptions` — [TODO: migrate 2 pipelines](docs/BACKLOG.md#pipe-3) |
| **R12** | Every injected parameter has a matching `if (!exists("X")) X <- …` fallback, spelled identically | `convention` |
| **R13** | Admin levels read from config, never hardcoded | `convention` — [Schema](#schema) |
| **R14** | Standard flags named `run_report_only` / `pull_scripts` / `overwrite` | `⚠ exceptions` — `run_reports_only` in `snt_dhs_indicators` |
| **R15** | One vocabulary per concept in operator-facing `choices=[...]` | `⚠ exceptions` — 3 routine-data vocabularies ([TODO](docs/BACKLOG.md#pipe-2)) |
| **R16** | `readme.md` updated in the same PR as the `pipeline.py` change it describes | `convention` — [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md) |
| **R17** | R failure messages prefixed `[ERROR]` or `[WARNING]`, chosen deliberately | `convention` — [Logging](#logging--error-labels) |
| **R18** | Python: snake_case, line-length 110, numpydoc docstrings with `Returns` | `ruff` — configured, but [nothing runs it in CI](docs/BACKLOG.md#tool-2) |
| **R19** | Agents never run destructive / history-rewriting `git` or `gh` commands | `enforced` — hook + `permissions.deny` in `.claude/` ([how](#how-this-is-enforced-r19)) |
| **R20** | R: roxygen2 docstrings above every function (`#' Title`, blank, description, `@param`, `@return`, `@export`) | `convention` — see `code/snt_utils.r` |
| **R21** | Every published output **table** carries a `YEAR` column | `convention` — new, existing outputs not audited ([why](#schema)) |
| **R22** | The shape of `release_manifest.json` changes only together with its schema in `snt_workspace_bootstrap` | `convention` — cross-repo, nothing checks it ([why](#release-management)) |

Adding a rule: add a row here *and* the rationale to the matching section below. A rule that is
only in the prose will be missed; a rule that is only in the table will be misapplied.

### Adding or changing a pipeline

1. `<name>/pipeline.py` — `@pipeline("<name>")`, `@parameter(...)`, orchestration only.
2. `<name>/requirements.txt` — match the existing two-line pattern unless more is genuinely needed.
3. `<name>/readme.md` — the user-facing contract. Follow
   [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md), which defines the
   required sections and how to verify each one against the code.
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

### Release management

A GitHub release of this repo is the unit that the tools in
[`BLSQ/snt_workspace_bootstrap`](https://github.com/BLSQ/snt_workspace_bootstrap) deploy into a
workspace and check it against. This repo holds only the producer side:
[`.github/workflows/generate_manifest.yaml`](.github/workflows/generate_manifest.yaml) runs when a
release is published, hashes every file the release ships, and attaches `release_manifest.json` to
the release.

**The manifest's shape is a cross-repo contract (R22).** `snt_workspace_deployer` and
`snt_workspace_checker` read it, and its schema lives in that repo:
[`docs/contracts/release_manifest.schema.json`](https://github.com/BLSQ/snt_workspace_bootstrap/blob/main/docs/contracts/release_manifest.schema.json),
with the prose in
[`release_manifest.md`](https://github.com/BLSQ/snt_workspace_bootstrap/blob/main/docs/contracts/release_manifest.md).
Do not change the shape (a key, a type, or what a value means) without updating that schema in the
same change. In practice that means a paired PR in `snt_workspace_bootstrap`, merged together with
this one. The schema's description says which changes break compatibility.

*Which* paths a release ships is this repo's choice, not part of the schema: the four R glob
patterns, plus every top-level directory holding a `pipeline.py`. Changing that does not break the
contract, but it changes what the deployer installs and the checker compares, so say so in the PR.
For example, a new top-level directory with a `pipeline.py` ships in the next release.

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
  (pydocstyle + pydoclint are enabled). R: snake_case functions, `<-` assignment, roxygen2
  docstrings above the function signature (`#' Title`, blank `#'`, description, `@param` per
  argument, `@return`, `@export`) — see `code/snt_utils.r` for examples.

### Schema

- **All column names UPPERCASE** in every published artefact. Python: `df.columns.str.upper()`
  at the merge boundary. R: `clean_column_names()`.
- Join keys: `ADM1_ID` / `ADM2_ID` (+ `_NAME`), `YEAR`, `MONTH`, `PERIOD`. `ADM2_ID` is the grain
  of the SNT data layers.
- **Never hardcode an admin level.** Read `DHIS2_ADMINISTRATION_1` / `DHIS2_ADMINISTRATION_2`
  (strings like `"level_3_name"`, parsed with `re.search(r"level_(\d+)_", …)`) and
  `ANALYTICS_ORG_UNITS_LEVEL` (integer). They differ per country and are *not* interchangeable:
  `ANALYTICS_ORG_UNITS_LEVEL` is the facility level for routine data, `DHIS2_ADMINISTRATION_2`
  the district level for population, shapes and reporting indicators. Validate against
  `pyramid["level"].max()` as the existing tasks do.
- A published column reaches the SNT Explorer only if a layer in
  [`metadata/SNT_metadata_master.json`](metadata/README.md) points at it. Adding an indicator means
  adding its layer there too (see the [Handover checklist](#handover-checklist)).
- **Every published output table carries a `YEAR` column** (**R21**). Data points are always
  attached to a year, so this costs nothing — and the SNT Explorer imports `YEAR` alongside the
  layer's value column to let the user choose which year to display. A table without it cannot be
  displayed over time, and the omission surfaces only when the layer fails to load. Applies to
  tabular outputs (`.parquet` / `.csv`); not to shapes (`{CC}_shapes.geojson`), the org-unit
  pyramid, or other genuinely time-independent artefacts. **This rule is newly stated — existing
  outputs have not been audited against it.** Bring a table into line when you touch it; do not
  assume every current output already complies.
- **`SNT_metadata.json` is mid-format-change.** The copy archived in `deprecated/configuration/` is the
  old shape; its only reader, `snt_assemble_results`, is retired, so nothing reads it today. The **target**
  shape — one entry per SNT Explorer data layer, with a `SOURCE_DATA` pointer and bilingual labels — is specified in
  [`contracts/snt_metadata_json/`](contracts/snt_metadata_json/README.md), with a JSON Schema as its
  source of truth. Write new
  metadata entries to that schema, and read its open questions
  ([BACKLOG `META-*`](docs/BACKLOG.md#snt-explorer-metadata)) before assuming a field's
  meaning: the format is still being validated with the IASO developers.

### Configuration

- Pipelines read `<workspace>/configuration/SNT_config.json` — one file, one country, one workspace.
  (That is the workspace path — this repo has no `configuration/` folder.)
- The current structure of that file is `contracts/snt_config_json/SNT_config.skeleton.json` — keep
  it in sync when a schema key changes. The old `SNT_config_<CC>.json` copies in
  `deprecated/configuration/` are **examples only, not loadable**, and were never verified against
  the workspaces. Do not add logic that reads the `_<CC>` names.
- `.gitignore` blocks `*.json` except `deprecated/configuration/*.json`,
  `.claude/settings.json`, `contracts/**/*.json` and `metadata/SNT_metadata_master.json` — a new
  JSON file needs a deliberate negation, not a force-add. The `.claude/settings.json` negation is
  load-bearing: without it the destructive-git guardrail (**R19**) is ignored by git and never
  reaches a colleague's clone.

### Logging & error labels

- Python: `current_run.log_info / log_warning / log_error / log_debug`. R: `log_msg(msg, level)`,
  or `pipeline_msg()` when the code may run outside a pipeline.
- **R error severity is carried by a string prefix.** A message starting `[ERROR]` or `[WARNING]`
  is mapped to OpenHEXA severity by
  `error_label_severity_map={"[ERROR]": "error", "[WARNING]": "warning"}`. A `[WARNING]`-labelled
  failure suppresses HTML report generation but does **not** fail the run — so labelling a real
  data-loss condition `[WARNING]` hides it. Keep messages actionable: at minimum, name the missing
  file — `load_dataset_file()` in `code/snt_utils.r` (the shared loader most pipelines call to read
  a dataset file) does this by default. It has no way to know which upstream pipeline produces a
  given file, so when that context is useful, add it in the caller: see the routine-data load cell
  in `pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb`, which wraps
  `load_dataset_file()` in its own `tryCatch` to name the specific upstream pipeline ("DHIS2
  Outliers Removal and Imputation") when the routine file is missing.

### Notebook hygiene

- Strip outputs before every commit (`nbstripout`). Executed notebooks belong in
  `papermill_outputs/` and `reporting/outputs/` inside the workspace, never in git.
- Windows `*.ipynb:Zone.Identifier` sidecars are ignored — do not commit them; several are already
  tracked by mistake under `pipelines/snt_dhis2_formatting/reporting/`.
- Every parameter injected from `pipeline.py` needs an `if (!exists("X")) X <- <default>` fallback
  cell, so the notebook stays interactively runnable. Change both sides together.
- **Notebook parameter globals are UPPERCASE** (**R11**) — `ROUTINE_DATA_CHOICE`, `SNT_ROOT_PATH`,
  `DEVIATION_IQR`. This distinguishes an injected pipeline parameter from an ordinary R local at a
  glance, and matches the UPPERCASE column convention. Two pipelines predate the rule and use
  lowercase — see [Traps](#traps). New parameters are UPPERCASE even when added to one of those
  two, *unless* that would leave a single notebook mixing both: converting a violator is an
  all-at-once change, not a drive-by.
- **A notebook change needs a domain reviewer**, not just any reviewer. Every change to `main` goes
  via PR, and a change to `pipelines/*/code/`, `pipelines/*/utils/` or `code/` should be reviewed by
  the person responsible for that pipeline — listed in
  [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md#2-pipeline-responsibilities). This is a convention only:
  nothing blocks an unreviewed merge. There is no `CODEOWNERS` file: one was drafted and the team
  decided against it.
### Country-specific notebook variants

**The mechanism.** A notebook whose filename is the generic name plus an underscore and a country
code — `snt_seasonality_rainfall_NER.ipynb` beside `snt_seasonality_rainfall.ipynb` — is executed
**instead of** the generic notebook, but only in the workspace whose
`SNT_CONFIG.COUNTRY_CODE` matches that suffix. Everywhere else the generic notebook runs and the
variant is inert.

`pipeline.py` never names the variant. It passes `country_code=` to `run_notebook()` /
`run_report_notebook()` alongside the *generic* `nb_path`, and the substitution happens inside
`snt_lib` (external repo — do not guess its resolution rules beyond this):

```python
run_notebook(
    nb_path=pipeline_path / "code" / "snt_seasonality_rainfall.ipynb",  # generic name only
    country_code=country_code,                                          # ← enables the swap
    ...
)
```

**Why it exists.** It lets a country keep a bespoke analysis without that file being clobbered on
the next `Pull scripts` run. Variants are deliberately **not** listed in
`pull_scripts_from_repository(code_scripts=[...])` — check any pipeline and you will see only
generic names there. That single omission is what makes them safe, and it cuts both ways:

| | `Pull scripts` = ON |
|---|---|
| Generic notebook | overwritten from `main` |
| `_<CC>` variant | **untouched — neither overwritten nor delivered** |

**The consequence — there is no version-propagation story.** A variant has to be created and edited
**in the workspace**. Committing it to `main` is *archival only*: nothing brings it into any
workspace, ever. So a committed variant is a copy of what the workspace had at some past moment,
with no mechanism keeping the two in step and nothing that reports the drift. Treat the committed
file as a backup, not as the running code.

**Working rules:**

- **Prefer a config-driven branch inside the generic notebook** over creating a variant. A variant
  is a permanent fork that no tooling maintains.
- When a variant is unavoidable, record the reason and the ticket in a markdown cell inside it.
- After editing a variant in the workspace, **copy it back and commit it immediately** — same
  discipline as `utils/*.r` helpers, and for the same reason.
- **Before changing a generic notebook, check for `_<CC>` siblings.** A fix applied to
  `snt_seasonality_rainfall.ipynb` does **not** reach Niger if `..._NER.ipynb` exists there. State
  in the handover which countries need the change applied to their variant by hand.
- **Location matters.** A variant only takes effect when it sits in the same folder the generic
  notebook is loaded from (`code/`, or `reporting/`). Files parked in a `country_specific/`
  folder — `pipelines/snt_dhis2_incidence/country_specific/snt_dhis2_incidence_NER.ipynb`,
  `pipelines/snt_dhis2_formatting/country_specific/snt_dhis2_formatting_pyramid_BDI.ipynb` — are
  **not** on any execution path; they are archived copies. Only
  `pipelines/snt_seasonality_rainfall/code/snt_seasonality_rainfall_NER.ipynb` is committed in an
  active location.

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
    needs a method/run discriminator column before it is depended on again
    ([BACKLOG PIPE-4](docs/BACKLOG.md#pipe-4)).
- **A country may not be running the notebook you are editing.** If a `<generic>_<CC>.ipynb`
  variant exists in that workspace, it runs *instead* of the generic notebook, and `Pull scripts`
  neither overwrites nor delivers it. Your fix silently misses that country, and `main` is not a
  reliable record of which variants exist. → [Country-specific notebook
  variants](#country-specific-notebook-variants)
- **Missing inputs skip, they do not fail.** `snt_dhis2_formatting` gates each of its five stages
  on `dataset_file_exists()`; `download_dhis2_analytics` catches per-period errors and continues.
  A partial run looks successful. If you add a stage, decide deliberately between skip and raise,
  and log the choice.
- **Pipelines are not a DAG.** They are launched manually and re-run independently, so a results
  table can mix data vintages. Never assume your upstream ran today.
- **Country escape hatches are hardcoded** in `snt_dhis2_extract/pipeline.py`: BFA filters
  `level_4_name` starting `"DS"`; NER fetches org-unit groups and rewrites the pyramid through an
  R notebook. Adding a country may mean adding a branch there — check the pyramid levels first.
  Exclude it. `deprecated/` is history, never a template.
- `snt_lib` (`github.com/BLSQ/snt_utils`) is an **external, unpinned** dependency — its source is
  not in this repo. Do not guess its signatures; read the upstream repo or an existing call site.
- **The external-source pipelines depend on `snt_dhis2_formatting`.** `snt_era5_climate_data`,
  `snt_map_extracts`, `snt_worldpop_extract` and `snt_healthcare_access` all fetch
  `{CC}_shapes.geojson` from `DHIS2_DATASET_FORMATTED` first. They look like independent roots;
  they are not.
- **`data/worldpop/rasters/` is a shared cache across three pipelines**, keyed on the filename
  pattern `{cc_lower}_pop_{year}_*.tif` — note the *lowercase* country code, unlike every other
  data file in the system. This is the one place pipelines couple through the filesystem instead
  of a dataset. Don't rename those files.
- **The same concept has three different parameter vocabularies.** "Routine data with outliers
  removed" is `raw_without_outliers` in `snt_dhis2_incidence`, `outliers_removed` in
  `snt_dhis2_reporting_rate_dataelement`, and `removed` under a differently-named parameter
  (`data_action`) in `snt_dhis2_quality_of_care`. Check the target pipeline's `choices=[...]` before
  assuming. (`snt_dhis2_reporting_rate_dataset` takes no routine data and has no such parameter.)
- **Two pipelines break the UPPERCASE parameter rule (R11).** `snt_seasonality_cases` and
  `snt_seasonality_rainfall` inject lowercase globals.
  Each is internally self-consistent, so it works — but it means you cannot assume the case of a
  parameter without checking. Read the pipeline's injected dict before writing the notebook's
  `exists()` cell. **Not a permitted variant**: logged for migration as [BACKLOG PIPE-3](docs/BACKLOG.md#pipe-3). Do not half-convert
  one — a notebook mixing `threshold_for_seasonality` and `THRESHOLD_FOR_SEASONALITY` is worse
  than either.
  - Beware the near-miss in `snt_healthcare_access`: it injects UPPERCASE
    (`INPUT_FOSA_FILE`, `WORLDPOP_YEAR`) into the notebook but records lowercase keys in its
    parameters JSON. Both are intentional; only the notebook side is governed by R11.
- **Selecting "Pregnant Women" in `snt_dhis2_incidence` fails** — confirmed defect. The mapped
  value `PREGNANT_WOMAN` (singular) correctly drives the indicator suffix but composes
  `POP_PREGNANT_WOMAN`, while every producer writes `POP_PREGNANT_WOMEN` (plural). It stops loudly,
  so no bad data — but the pipeline's own help text makes the bug read as expected behaviour. Fix
  belongs in `select_population_column()`, not in the mapping. See
  [`docs/DATA_ARCHITECTURE.md` §6.1](docs/DATA_ARCHITECTURE.md).
- **`snt_dhis2_reporting_rate_dataelement` is the reference implementation for routine-file
  selection** — its `resolve_routine_source()` maps each choice to both the dataset and the filename
  in one table, raises on an unknown choice, and the pipeline verifies the file exists with
  `dataset_file_exists()` before running anything. The notebook receives both as `DATASET_ID` /
  `ROUTINE_FILE` and makes no choice of its own. Copy that shape rather than inventing another.
  (Its `_dataset` sibling no longer reads routine data: it uses only the formatted reporting
  extract.)
- **`snt_assemble_results` is retired** — moved to `deprecated/DEPRECATED_snt_assemble_results/`,
  no longer deployed by CI, and its OpenHEXA template is deleted. The SNT Explorer reads the
  OpenHEXA datasets directly instead. Its `SNT_RESULTS` dataset id is removed from
  `SNT_config.skeleton.json` but remains in the archived `SNT_config_<CC>.json` copies and
  in live workspace configs; nothing reads it. `deprecated/configuration/SNT_metadata.json` (the old format)
  was read only by this pipeline and has no consumer left.

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


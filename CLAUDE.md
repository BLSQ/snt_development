# CLAUDE.md — working rules for `snt_development`

Guardrails for anyone (human or agent) changing code in this repository.

- **[Conventions → Register](#register)** — every hard rule (R1–R19) in one table, with its
  enforcement status and known exceptions. Start there if you want the rules without the prose.
- New here? Orientation and the documentation map: [`README.md`](README.md).
- Architecture, lineage and dataset contracts: [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md).
- Domain vocabulary: [`docs/GLOSSARY.md`](docs/GLOSSARY.md).
- Writing a pipeline `readme.md`: [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md).
- Who to ask / how review works: [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md).

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
plus external sources (ERA5, WorldPop, Malaria Atlas Project, DHS) into a one-row-per-ADM2
malaria **subnational tailoring** results table.

Orchestration is OpenHEXA's `@pipeline` / `@task` SDK. The analytics themselves live in **R
notebooks executed by papermill**. Each pipeline is launched by hand from the OpenHEXA UI, and
pipelines pass data to each other through **OpenHEXA datasets** — never by calling each other.

An inventory of all 20 pipelines is in
[`docs/DATA_ARCHITECTURE.md` §1.1](docs/DATA_ARCHITECTURE.md#11-the-20-pipelines-at-a-glance).

```
<pipeline_name>/pipeline.py            ← deployed by CI. Orchestration only.
<pipeline_name>/requirements.txt       ← deployed by CI.
<pipeline_name>/readme.md              ← the pipeline's user-facing contract.
pipelines/<pipeline_name>/code/*.ipynb ← R analytics. NOT deployed by CI.
pipelines/<pipeline_name>/utils/*.r    ← R helpers sourced by the notebooks.
pipelines/<pipeline_name>/reporting/   ← R reporting notebooks.
code/*.r                               ← shared R library (snt_utils, snt_report, snt_palettes).
configuration/SNT_config_<CC>.json     ← reference copies only (see below).
dev/environment.yml                    ← local Python dev tools. Never runs on OpenHEXA.
pyproject.toml                         ← ruff's rulebook + PEP 621 metadata. Never runs on OpenHEXA.
.claude/                               ← agent guardrails (R19). Active on clone.
```

### Note — no dbt, no Airflow, no DAG engine, no tests

There is no `dbt_project.yml`, no `dags/` and no `tests/`. Do not look for them, do not infer they
exist, and do not report running a check that lives in none of them.

| Absent | Status | What follows for you |
|---|---|---|
| Scheduler / DAG engine | **deliberate** | Nothing enforces run order. Pipelines are launched by hand, one at a time, and may run against stale upstreams. Operator-facing order is unwritten: [`DATA_ARCHITECTURE.md` §4.2](docs/DATA_ARCHITECTURE.md) `[TODO: Giulia]`. |
| dbt | **permanent** | Analytics are R, not SQL. |
| Test suite | **debt** | `ruff` is the only automated check and [no CI runs it](#suggestions-logged-for-later-evaluation-giulia). Verify by reading; say what you could not verify. |
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
| **Python-only pipelines** (`snt_assemble_results`, `snt_dhis2_extract`, `snt_map_extracts`, `snt_worldpop_extract`, `snt_era5_climate_data`) | The best-supported path. Editable and lintable locally; still needs a workspace to actually run. |
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

  A single small `pull_request`-triggered workflow running `ruff check .` would close the second
  gap for every PR at once, without touching the 20 deployment files.
- **A `tests/` seed**: pure functions such as `validate_yyyymm`, `validate_period_range`,
  `get_unique_data_elements`, `validate_reporting_rates`, `merge_parquet_files`,
  `raw_reporting_ds_format` are dependency-free and unit-testable today.
- **R local loop**: a `renv.lock` + a small `Rscript` harness that sets the `PARAM` globals and
  sources `code/snt_utils.r` + `pipelines/<name>/utils/<name>.r` against a tiny fixture would
  make the R half testable without a workspace. `pipeline_msg()` already degrades gracefully
  when the `openhexa` object is absent, so the helpers are closer to runnable than they look.
- **Reproducible local environments — the *R* half is still open.** The Python half is good enough:
  `dev/environment.yml` (conda, team convention) + `pyproject.toml` for the `ruff` rules and PEP 621
  metadata. The one loose end is that the two carry overlapping dependency lists — accepted for now,
  to be revisited when the local-workflow strategy is settled; harmless meanwhile, because these are
  desk tools that never run in a workspace. What is still genuinely unsolved is reproducing the **R**
  side. Recommendation: **pull the workspace image.**

  | Option | Verdict |
  |---|---|
  | **conda / mamba, extended to R** | **Adopted for Python, not for R.** For desk tools it is the right call and it is done. Do not extend it to R: conda's `r-base` would drift from whatever the workspace actually runs, so "works locally" still would not mean "works in the workspace" — and it would not carry the system libraries either. |
  | **`renv.lock`** | **Useful, narrow.** R's native lockfile; the workspace image already ships `r-renv`. Pairs with the R-local-loop suggestion above. Gives reproducible R *packages*, but not the system libraries (GDAL/PROJ for `sf`, TeX for reports) that are the usual cause of "works there, not here". |
  | **Pull `blsq/openhexa-blsq-r-environment:latest`** | **Best value.** It is the *actual* runtime — public on Docker Hub, ~2.5 GB, R 4.5, all R packages, Quarto and the geo stack included. A `.devcontainer/` pointing at it gives VS Code a local environment identical to production, with no second dependency list to maintain. Contents documented in [`DATA_ARCHITECTURE.md` §7.4](docs/DATA_ARCHITECTURE.md#74-the-workspace-runtime-image). |

  Caveat before adopting: a local container has no OpenHEXA workspace mounted, so `workspace.files_path`,
  dataset access and connections are absent. It gives you a faithful *language* environment for
  helpers and pure functions — not a way to run a whole pipeline offline. The remote-kernel loop
  stays the way to run real analytics.

  Two questions for the OH devs: pin `latest` to a digest for reproducibility, and add
  `data.table` and `rmapshaper` to the image explicitly (see §7.4 — the code uses both, and both
  currently arrive only as transitive dependencies).
- **De-duplicate `worldpopclient.py`**, currently copied into three pipelines.
- **Stamp readmes with the version they describe**, to make drift detectable. Blocked on deciding
  *which* version number counts (source / template / workspace — see
  [`docs/DATA_ARCHITECTURE.md` §7.1.1](docs/DATA_ARCHITECTURE.md)). A commit SHA of the
  `pipeline.py` last verified against is well-defined today and needs no OpenHEXA change.
- **Unify the routine-data-choice vocabulary** across `snt_dhis2_incidence`, both
  `reporting_rate_*` pipelines and `snt_dhis2_quality_of_care` — operator-visible, so it needs a
  migration rather than a rename. (Rule **R15**.)
- **Migrate the three lowercase-parameter pipelines to UPPERCASE** (rule **R11**):
  `snt_dhis2_quality_of_care` (`data_action`), `snt_seasonality_cases` and
  `snt_seasonality_rainfall` (`minimum_month_block_size`, `maximum_month_block_size`,
  `threshold_for_seasonality`, `threshold_proportion_seasonal_years`,
  `use_calendar_year_denominator`). Purely internal — these are notebook globals, not `@parameter`
  codes, so **no operator-visible name changes and no OpenHEXA UI churn**, unlike R15. Each is a
  contained three-part edit: the injected dict in `pipeline.py`, the `if (!exists("X"))` fallback
  cell, and every use inside the notebook and its `utils/*.r`. It must be atomic per pipeline —
  a missed use site fails only at runtime, in a workspace, with an "object not found" error.
  Cheapest sequencing: do it in the same PR as the R15 vocabulary migration for
  `snt_dhis2_quality_of_care`, since that notebook is being touched anyway.
- **Write a domain glossary** (`docs/GLOSSARY.md`) — **the largest documentation gap left, and the
  one nobody but the SNT team can fill.** The code is full of domain terms that cannot be inferred
  from it: `N1_METHOD` with choices `PRES` / `SUSP-TEST`, `CSB` (care-seeking behaviour), `FOSA`,
  `PfPR`, `ITN`, `IRS`, epi-weeks, `ADM1`/`ADM2`, "incidence adjusted for reporting", the five
  outlier methods (what is "Magic Glasses"? what does the "PATH" method do?), and the difference
  between `ANALYTICS_ORG_UNITS_LEVEL` and `DHIS2_ADMINISTRATION_2` in *epidemiological* rather than
  structural terms. Without these, a newcomer — human or AI — guesses, and the guess ends up in a
  `readme.md`.

  Doing it properly needs a full sweep of the codebase (`pipeline.py` help strings, `choices`
  values, notebook markdown cells, output column names in `SNT_metadata.json`, config keys) to
  collect every term, then a pass by someone with the domain knowledge to define them. Suggested
  split: an agent produces the *list* with each term's call sites and a proposed definition where
  the code makes it unambiguous; Giulia (or a malaria epidemiologist) fills in and corrects the
  rest. Mark anything unconfirmed rather than shipping a plausible guess.
- **Documentation open items are tracked separately** in
  [`docs/OWNERSHIP.md` §4](docs/OWNERSHIP.md#4-open-items): filling in the per-pipeline responsible
  persons, confirming the (provisional MIT) licence, linking an external published reference for the
  SNT method, and removing the unadopted `.github/CODEOWNERS`.
- **Give the `outliers_detected` DB table a provenance discriminator** (method + run id, or
  append-with-run-id instead of overwrite) before its consumer is resumed. The dataset *files*
  are fine as they are — overwriting is the intended override mechanism and their companion
  `{CC}_parameters.json` records the method. The table has no such companion.

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
| **R2** | Notebook outputs stripped before commit | `convention` → [nbstripout git filter](#suggestions-logged-for-later-evaluation-giulia) |
| **R3** | `pipeline.py` orchestrates, never computes | `convention` |
| **R4** | An output only exists if it is passed to `add_files_to_dataset(...)` | `convention` |
| **R5** | Publish only from the `snt-development` workspace | `convention` — [why](#always-publish-from-snt-development) |
| **R6** | Every new notebook / `.r` file registered in `pull_scripts_from_repository(...)` | `convention` |
| **R7** | Every data file prefixed `{CC}_`, uppercase country code | `⚠ exceptions` — `data/worldpop/rasters/{cc_lower}_pop_*.tif` (see [Traps](#traps)) |
| **R8** | Parquet is the machine contract; write the `.csv` twin beside it | `convention` |
| **R9** | `{CC}_parameters.json` published beside the data, via `save_pipeline_parameters(...)` | `⚠ exceptions` — ERA5 stamps `pipeline_name="snt_era5_aggregate"`; healthcare_access stores the `File` object, not `.path` |
| **R10** | All **column** names UPPERCASE in every published artefact | `convention` |
| **R11** | All **notebook parameter** globals UPPERCASE, injected side and `exists()` side alike | `⚠ exceptions` — [TODO: migrate 3 pipelines](#suggestions-logged-for-later-evaluation-giulia) |
| **R12** | Every injected parameter has a matching `if (!exists("X")) X <- …` fallback, spelled identically | `convention` |
| **R13** | Admin levels read from config, never hardcoded | `convention` — [Schema](#schema) |
| **R14** | Standard flags named `run_report_only` / `pull_scripts` / `overwrite` | `⚠ exceptions` — `run_reports_only` in `snt_dhs_indicators` |
| **R15** | One vocabulary per concept in operator-facing `choices=[...]` | `⚠ exceptions` — 3 routine-data vocabularies ([TODO](#suggestions-logged-for-later-evaluation-giulia)) |
| **R16** | `readme.md` updated in the same PR as the `pipeline.py` change it describes | `convention` — [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md) |
| **R17** | R failure messages prefixed `[ERROR]` or `[WARNING]`, chosen deliberately | `convention` — [Logging](#logging--error-labels) |
| **R18** | Python: snake_case, line-length 110, numpydoc docstrings with `Returns` | `ruff` — configured, but [nothing runs it in CI](#suggestions-logged-for-later-evaluation-giulia) |
| **R19** | Agents never run destructive / history-rewriting `git` or `gh` commands | `enforced` — hook + `permissions.deny` in `.claude/` ([how](#how-this-is-enforced-r19)) |

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
- `.gitignore` blocks `*.json` except `configuration/SNT_config_*.json` and
  `.claude/settings.json` — a new config file needs a deliberate negation, not a force-add. The
  second negation is load-bearing: without it the destructive-git guardrail (**R19**) is ignored by
  git and never reaches a colleague's clone.

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
- **Notebook parameter globals are UPPERCASE** (**R11**) — `ROUTINE_DATA_CHOICE`, `SNT_ROOT_PATH`,
  `DEVIATION_IQR`. This distinguishes an injected pipeline parameter from an ordinary R local at a
  glance, and matches the UPPERCASE column convention. Three pipelines predate the rule and use
  lowercase — see [Traps](#traps). New parameters are UPPERCASE even when added to one of those
  three, *unless* that would leave a single notebook mixing both: converting a violator is an
  all-at-once change, not a drive-by.
- **A notebook change needs a domain reviewer**, not just any reviewer. Every change to `main` goes
  via PR, and a change to `pipelines/*/code/`, `pipelines/*/utils/` or `code/` should be reviewed by
  the person responsible for that pipeline — listed in
  [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md#2-pipeline-responsibilities). This is a convention only:
  nothing blocks an unreviewed merge. `.github/CODEOWNERS` exists but was **not adopted** by the
  team — do not treat it as the rule, and see
  [`docs/OWNERSHIP.md` §4](docs/OWNERSHIP.md#4-open-items) for its pending removal.
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
    needs a method/run discriminator column before it is depended on again.
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
  removed" is `raw_without_outliers` in `snt_dhis2_incidence`, `outliers_removed` in the two
  `reporting_rate_*` pipelines, and `removed` under a differently-named parameter (`data_action`)
  in `snt_dhis2_quality_of_care`. Check the target pipeline's `choices=[...]` before assuming.
- **Three pipelines break the UPPERCASE parameter rule (R11).** `snt_dhis2_quality_of_care`
  (`data_action`), `snt_seasonality_cases` and `snt_seasonality_rainfall` inject lowercase globals.
  Each is internally self-consistent, so it works — but it means you cannot assume the case of a
  parameter without checking. Read the pipeline's injected dict before writing the notebook's
  `exists()` cell. **Not a permitted variant**: logged for migration below. Do not half-convert
  one — a notebook mixing `data_action` and `DATA_ACTION` is worse than either.
  - Beware the near-miss in `snt_healthcare_access`: it injects UPPERCASE
    (`INPUT_FOSA_FILE`, `WORLDPOP_YEAR`) into the notebook but records lowercase keys in its
    parameters JSON. Both are intentional; only the notebook side is governed by R11.
- **Selecting "Pregnant Women" in `snt_dhis2_incidence` fails** — confirmed defect. The mapped
  value `PREGNANT_WOMAN` (singular) correctly drives the indicator suffix but composes
  `POP_PREGNANT_WOMAN`, while every producer writes `POP_PREGNANT_WOMEN` (plural). It stops loudly,
  so no bad data — but the pipeline's own help text makes the bug read as expected behaviour. Fix
  belongs in `select_population_column()`, not in the mapping. See
  [`docs/DATA_ARCHITECTURE.md` §6.1](docs/DATA_ARCHITECTURE.md).
- **`snt_dhis2_reporting_rate_*` is the reference implementation for routine-file selection** —
  its `resolve_routine_filename()` is explicit and total, and it verifies the file exists with
  `dataset_file_exists()` before running anything. Copy that shape rather than inventing another.
- **`snt_assemble_results` is being deprecated** — the SNT Explorer will read the OpenHEXA datasets
  directly instead. Don't extend it, and treat `configuration/SNT_metadata.json` as mid-change.

---

## Handover checklist

Before calling a change done — each item maps to a rule in the [Register](#register):

- [ ] `ruff check <changed dirs>` clean, in the `snt_development` conda env. *(R18)*
- [ ] Notebook outputs stripped; no `.csv`/`.parquet`/Zone.Identifier files staged. *(R1, R2)*
- [ ] `pipeline.py` parameters ↔ notebook `exists()` fallbacks agree, name for name and **case for
      case**; new globals are UPPERCASE. *(R11, R12)*
- [ ] New outputs are in `add_files_to_dataset(...)`, in the pipeline `readme.md`, and in
      `docs/DATA_ARCHITECTURE.md`. *(R4, R16)*
- [ ] New notebook/`.r` filenames registered in `pull_scripts_from_repository(...)`. *(R6)*
- [ ] New pipeline: workflow file added with the name updated in *all* places, `workspace:` left
      as `snt-development`. *(R5)*
- [ ] `readme.md` re-verified against the code, not patched by memory —
      [`docs/PIPELINE_README_STANDARD.md` §3](docs/PIPELINE_README_STANDARD.md). *(R16)*
- [ ] Handover states: which country/workspace it was tested in (or that it was not), and that
      operators must run with **`Pull scripts` = ON** to pick up notebook changes.


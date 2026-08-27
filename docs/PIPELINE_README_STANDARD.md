# Pipeline `readme.md` standard

Every `<pipeline_name>/readme.md` is that pipeline's **user-facing contract** — the thing an
operator reads before pressing Run in the OpenHEXA UI, and the thing a developer reads before
changing it. This document defines what one must contain and how to check it against the code.

It is deliberately **tool-agnostic**: it works as instructions for a human, or as a prompt for any
assistant (Claude, Gemini, …). Keeping it in the repo means everyone writes to the same target,
it is reviewable in a PR, and it improves by pull request rather than by whoever happens to own a
private prompt.

Related: [`CLAUDE.md`](../CLAUDE.md) · [`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md)

*Provenance: consolidates the "SNT Pipeline Documentation Specialist" instructions previously used
as a private Gemini Gem, plus the structure observed across all 20 existing readmes and the
findings of the 2026-08 code audit.*

---

## 0. Source of truth — read this before writing anything

**The code is the only authority.** Everything else in the repository is a claim about the code
that may or may not still be true.

Before writing or updating a readme, read, in this order:

1. **`<pipeline_name>/pipeline.py`** — parameters, orchestration, guards, and above all the
   `add_files_to_dataset(...)` calls.
2. **The core notebook(s)** under `pipelines/<pipeline_name>/code/*.ipynb` — the actual analytics,
   the dataset ids resolved from `config_json$SNT_DATASET_IDENTIFIERS$…`, and the grouping logic
   that determines resolution. **Ignore `*_report.ipynb`** — reporting notebooks render results,
   they do not define the contract.
3. **`pipelines/<pipeline_name>/utils/*.r`** — where filename resolution and column selection
   usually live.

Supplementary material — the existing readme, code comments, a draft, a ticket, a colleague's
description — may be used to *enrich* wording. It may never be used as evidence.

**The verification rule.** Where a comment, docstring, draft or previous readme disagrees with the
executable code, **document what the code actually does**. Then say so: add a short note after the
readme, in the PR description, listing every discrepancy found. Do not silently paper over one —
a comment that no longer matches the code is a bug someone needs to see. Real examples from the
2026-08 audit, all invisible without reading the code:

- `snt_era5_climate_data` writes daily, weekly, epi-weekly and monthly parquets but publishes
  **only the monthly one**; the module-level `ERA5_VARIABLES` list has both temperature variables
  commented out, so only precipitation is fetched.
- `snt_worldpop_extract` writes per-year rasters and parquets but publishes **only the
  concatenation**.
- `snt_dhs_indicators` publishes a new dataset version even in report-only mode, because
  `add_files_to_dataset(...)` sits outside the `if not run_reports_only:` guard.
- `snt_era5_climate_data` stamps `pipeline_name="snt_era5_aggregate"` into its parameters JSON —
  the name of a pipeline that no longer exists.

Never describe a pipeline from memory, and never edit only the sentence you happen to be looking
at — re-run the §3 checklist.

---

## 1. Required structure

All 20 existing readmes already follow this. Keep the section order exactly.

```markdown
# SNT <Human Readable Name> Pipeline

<2–3 sentences: what the pipeline does, its primary objective, what it publishes and to which
 dataset. Name the dataset identifier in bold, e.g. **`DHIS2_DATASET_FORMATTED`**.>

## Parameters
## Functionality Overview
## Inputs
## Outputs

> **Notes for the Data Analyst:** …
```

Optional additions, when they earn their place:

- `## Key aggregations (reference)` — a table of the actual arithmetic, when the pipeline
  aggregates (see `snt_assemble_results`).
- `### Part 1 / Part 2 / …` subsections under `## Parameters`, when a pipeline has many
  parameters in distinct groups (see `snt_dhis2_population_transformation`).

## 2. What goes in each section

### `## Parameters`

One bullet per **domain** parameter, in the order they appear in `pipeline.py`:

```markdown
* **`parameter_code`** (Type, Required/Optional):
  * **Name:** <the `name=` shown in the OpenHEXA UI>
  * **Description:** <what it does, and what each choice means>
  * **Choices:** `a`, `b`, `c` — omit if unconstrained.
  * **Default:** `value`.
```

Rules:

- Use the **`code`** (the Python argument name), not the UI label, as the bullet's identifier —
  that is what appears in the parameters JSON and in the notebook.
- Map `@parameter(...)` decorators straight across: `name=`, `type=`, `required=`, `default=`,
  `choices=`.
- Document **every `choices=[...]` value**, not just that choices exist. Vocabularies differ
  between pipelines for the same concept (see [`CLAUDE.md`](../CLAUDE.md) → Traps), so this is
  load-bearing.
- **Skip the generic ones.** Do not document `run_report_only` / `pull_scripts` / `overwrite`, and
  do not describe the loading of `SNT_config.json` as if it were a parameter — they behave
  identically everywhere and are documented centrally. *Exception:* if a pipeline's variant differs
  at all (e.g. `run_reports_only` in `snt_dhs_indicators`, which also publishes in that mode), say
  so explicitly instead of omitting it.
- If the pipeline takes no domain parameters, say so in one line rather than deleting the section.

### `## Functionality Overview`

A numbered list of the key logical steps — data loading, pre-processing, calculation, export — in
execution order. Each step names the **artefact** it produces or the **notebook** it runs, in
backticks.

Be specific about **resolution**, spatial and temporal, and infer it from the grouping logic in the
code rather than from prose: `group_by(ADM2_ID, YEAR, MONTH)` means ADM2 × monthly, whatever the
old readme says.

State branch conditions explicitly — especially **skip conditions**, since missing inputs skip
rather than fail across this codebase:

> When **`[COUNTRY_CODE]_dhis2_raw_shapes.parquet`** exists on **`DHIS2_DATASET_EXTRACTS`**, run
> `…_shapes.ipynb` to produce **`[COUNTRY_CODE]_shapes.geojson`**.

### `## Inputs`

Every file the pipeline reads, with **the dataset it comes from** and whether it is required or
optional. Include `configuration/SNT_config.json` and the specific keys used — it is skipped as a
*parameter*, but it is a real input and the keys are part of the contract. Include operator
uploads. Include implicit inputs — a shared raster cache, or a geojson fetched only to define
boundaries, still counts.

### `## Outputs`

Split into what lands on the **workspace filesystem** and what is **published to a dataset** —
they are not the same set, and conflating them is the most common readme error in this repo.

Use `[COUNTRY_CODE]` as the placeholder, matching existing readmes. State explicitly when a file
is written but *not* published.

### `> **Notes for the Data Analyst:**`

A blockquote of nested bullets, for column definitions, data types and caveats that an analyst
reading the output needs but an operator pressing Run does not:

```markdown
> **Notes for the Data Analyst:**
>
> - **`COLUMN_NAME`**: what the column is.
>   - Sub-point explaining conditional logic (e.g. "If `use_transformed_population` is True, …").
> - **`VAR1`** & **`VAR2`**: combined description for related fields.
> - **Grain:** <monthly/yearly × facility/ADM2>.
> - **Guarded execution:** <what is skipped rather than failed, and when>.
```

## 3. Verification checklist

Check each claim against the code — see §0 for why this is not optional:

| Section | Verify against |
|---|---|
| Parameters | the `@parameter(...)` decorators in `pipeline.py` — name, type, default, required, every `choices` value |
| Functionality Overview | the pipeline function body, in order; note each `if`/`return`/`dataset_file_exists` guard |
| Resolution claims | the `group_by(...)` / aggregation keys in the notebook, not the prose |
| Inputs | every `get_file_from_dataset` / `dataset_file_exists` / `load_configuration_snt` call, **plus** the dataset ids resolved inside the R notebook (`config_json$SNT_DATASET_IDENTIFIERS$…`) |
| Outputs — filesystem | every `write_parquet` / `to_parquet` / `write_csv` / `export_data` path |
| Outputs — published | **only** the paths actually passed to `add_files_to_dataset(...)` |
| Notebook parameters | the injected dict matches the notebook's `if (!exists("X"))` fallback cell, name for name and case for case |

**The published-outputs row is where drift hides** — see the ERA5 and WorldPop examples in §0.

## 4. Tone, style and format

- **Tone:** professional, technical, objective. No marketing language.
- **Brevity:** bullet points and numbered lists, not long paragraphs.
- **Emphasis:** bold for variable names, file names and dataset identifiers; backticks for code
  identifiers (`ADM2_ID`, `pipeline.py`, `add_files_to_dataset`).
- **Language:** English.
- **Filename:** `readme.md`, lowercase, in the pipeline's Python directory (`<pipeline_name>/`),
  not under `pipelines/<pipeline_name>/`.
- **Markdown:** GitHub-flavored, standard headers (`#`, `##`, `###`).

## 5. Drift

A readme is updated at the discretion of whoever edits `pipeline.py`, and nothing detects one that
has fallen behind. Until a version-stamp mechanism exists
([`DATA_ARCHITECTURE.md` §7.2](DATA_ARCHITECTURE.md)), the mitigation is procedural:

- Changing `pipeline.py` parameters, outputs, or dataset publication means updating `readme.md`
  **in the same PR**.
- Re-run the §3 checklist rather than patching the sentence you happen to be looking at.

## 6. Template

```markdown
# SNT <Name> Pipeline

The **SNT <Name>** pipeline <what it does, 2–3 sentences>. It publishes <artefacts> to
**`<DATASET_ID>`** and runs the <name> reporting notebook.

## Parameters

* **`<code>`** (<Type>, <Required|Optional>):
  * **Name:** <UI label>
  * **Description:** <behaviour; enumerate every choice>
  * **Choices:** `<a>`, `<b>`
  * **Default:** `<value>`.

## Functionality Overview

1. **Configuration:** Load and validate **`SNT_config.json`**, resolve **`COUNTRY_CODE`** and the
   dataset identifiers used below.
2. **<Stage>:** <what runs, under what condition, producing what, at what resolution>.
3. **Publish:** Save the pipeline parameters JSON and upload <files> to **`<DATASET_ID>`**.
4. **Reporting:** Run **`<report>.ipynb`**.

## Inputs

* **`[COUNTRY_CODE]_<file>.parquet`** on **`<UPSTREAM_DATASET_ID>`** — <required|optional; what
  happens when missing>.
* **`configuration/SNT_config.json`** for **`<KEYS USED>`**.

## Outputs

**Workspace filesystem**

* **`data/<path>/[COUNTRY_CODE]_<file>.parquet`** and **`.csv`**
* **Pipeline parameters JSON** in the same directory
* **Report outputs** under **`pipelines/<name>/reporting/outputs/`**

**Published to `<DATASET_ID>`**

* <only the subset actually passed to `add_files_to_dataset`>

> **Notes for the Data Analyst:**
>
> - **`<COLUMN>`**: <definition>.
>   - <conditional logic, if any>.
> - **Grain:** <monthly/yearly × facility/ADM2>.
> - **Guarded execution:** <what is skipped rather than failed, and when>.
```

# Ownership & review — who to ask about what

<!-- ▼▼▼ Update this block on every substantive edit. Keep it directly under the H1. ▼▼▼ -->

| | |
|---|---|
| **Revision** | `0.1` — first draft, names not yet filled in |
| **Last updated** | **2026-08-31** |
| **Status** | 🚧 **Placeholder.** Every `Responsible` cell is `_TBD_`. See [BACKLOG DOC-4](BACKLOG.md#doc-4). |
| **Owner** | @sPuntinG |

<!-- ▲▲▲ End of version block ▲▲▲ -->

Companion documents: [`../README.md`](../README.md) (start here) ·
[`../CLAUDE.md`](../CLAUDE.md) (working rules) ·
[`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md) (lineage)

---

## 1. The review convention

**Every change to `main` goes through a pull request, and every PR should be reviewed by a team
member.** For a change to a pipeline's analytics, that reviewer should be the person listed for
that pipeline in [§2](#2-pipeline-responsibilities).

This is a **convention, not an enforced rule**. Nothing in GitHub blocks an unreviewed merge, and
there is no `CODEOWNERS` file — one was drafted and the team decided against it
([ADR 0005](decisions/0005-no-codeowners.md)). So the convention holds only because people follow it. Two consequences worth stating:

- **The burden is on the author.** Requesting the right reviewer is your job, not the reviewer's.
- **Silence is not approval.** If nobody with context has read a change to `pipelines/*/code/` or
  `code/`, it has not been reviewed, regardless of how the merge button looked.

**Why this matters more here than in most repositories.** These pipelines compute epidemiological
indicators that inform where malaria interventions are deployed. A change that shifts a number
does not fail, does not warn, and is not caught by any test — there are none, as
[`../README.md`](../README.md) records under *Status and known limitations*. It simply produces different
districts. Treat a one-line change to a formula as a bigger deal than a hundred-line change to
orchestration.

**What needs a domain reviewer, not just any reviewer:**

| Change to | Review needed |
|---|---|
| `pipelines/*/code/*.ipynb`, `pipelines/*/utils/*.r`, `code/*.r` | The pipeline's responsible person **and** someone who understands the method |
| `deprecated/configuration/SNT_metadata.json` | Old format, no reader since `snt_assemble_results` was retired. Layers now live in `catalogs/SNT_metadata_master.json` |
| `<pipeline>/pipeline.py`, `requirements.txt`, `.github/workflows/` | Any team member |
| `docs/`, `readme.md`, `CLAUDE.md` | Any team member |

---

## 2. Pipeline responsibilities

Ask this person before changing the pipeline's analytics; ask them to review the PR.

> **`_TBD_` means the name is not recorded yet — not that nobody is responsible.** Ask
> @sPuntinG until the cell is filled in.

| Stage | Pipeline | Responsible | Backup |
|---|---|---|---|
| A | `snt_dhis2_extract` | `_TBD_` | `_TBD_` |
| **B** | `snt_dhis2_formatting` ⚠️ *the hinge — everything downstream depends on it* | `_TBD_` | `_TBD_` |
| C | `snt_dhis2_outliers_imputation_iqr` | `_TBD_` | `_TBD_` |
| C | `snt_dhis2_outliers_imputation_mean` | `_TBD_` | `_TBD_` |
| C | `snt_dhis2_outliers_imputation_median` | `_TBD_` | `_TBD_` |
| C | `snt_dhis2_outliers_imputation_magic_glasses` | `_TBD_` | `_TBD_` |
| C | `snt_dhis2_outliers_imputation_path` | `_TBD_` | `_TBD_` |
| C | `snt_dhis2_population_transformation` | `_TBD_` | `_TBD_` |
| D | `snt_dhis2_incidence` | `_TBD_` | `_TBD_` |
| D | `snt_dhis2_reporting_rate_dataelement` | `_TBD_` | `_TBD_` |
| D | `snt_dhis2_reporting_rate_dataset` | `_TBD_` | `_TBD_` |
| D | `snt_dhis2_quality_of_care` | `_TBD_` | `_TBD_` |
| D | `snt_seasonality_cases` | `_TBD_` | `_TBD_` |
| D | `snt_seasonality_rainfall` | `_TBD_` | `_TBD_` |
| A′ | `snt_era5_climate_data` | `_TBD_` | `_TBD_` |
| A′ | `snt_worldpop_extract` | `_TBD_` | `_TBD_` |
| A′ | `snt_map_extracts` | `_TBD_` | `_TBD_` |
| A′ | `snt_healthcare_access` | `_TBD_` | `_TBD_` |
| A′ | `snt_dhs_indicators` | `_TBD_` | `_TBD_` |

Stages are those of [`DATA_ARCHITECTURE.md` §1.1](DATA_ARCHITECTURE.md#11-the-pipelines-at-a-glance).

---

## 3. Cross-cutting assets

Not owned by any one pipeline; a change here affects many.

| Asset | Responsible | Note |
|---|---|---|
| `code/snt_utils.r`, `code/snt_report.r`, `code/snt_palettes.r` | `_TBD_` | Sourced by most notebooks. A signature change breaks pipelines silently. |
| `deprecated/configuration/SNT_config_<CC>.json` | `_TBD_` | Archived reference copies, examples only. The live file is renamed by hand inside each workspace. |
| `deprecated/configuration/SNT_metadata.json` | `_TBD_` | Old-format column list. Its only reader, `snt_assemble_results`, is retired. |
| `.github/workflows/push_snt_*.yaml` | `_TBD_` | Never change `workspace: "snt-development"`. |
| `CLAUDE.md`, `.claude/` | @sPuntinG | Agent guardrails and the rule Register. |
| `docs/` | @sPuntinG | Architecture, glossary, standards. |
| `snt_lib` / `snt_utils` ([external repo](https://github.com/BLSQ/snt_utils)) | `_TBD_` | **Outside this repo**, and unpinned — changes there reach every pipeline on its next deploy. |

**Country-specific escape hatches** exist in two forms: branches hardcoded in
`snt_dhis2_extract` (BFA, NER), and `<generic_name>_<CC>.ipynb` notebook variants that run instead
of the generic notebook in one country's workspace. Whoever owns the country engagement should be
consulted before changing either — and note that a variant lives in the workspace, not in `main`,
so this repository is not a reliable record of which ones exist.
→ [Traps](../CLAUDE.md#traps) · [Country-specific notebook
variants](../pipelines/CLAUDE.md#country-specific-notebook-variants)

---

## 4. Open items

In [`BACKLOG.md`](BACKLOG.md#documentation): filling in the `_TBD_` cells ([DOC-4](BACKLOG.md#doc-4)),
confirming the licence ([DOC-5](BACKLOG.md#doc-5)) and linking a published reference for the SNT
method ([DOC-6](BACKLOG.md#doc-6)).

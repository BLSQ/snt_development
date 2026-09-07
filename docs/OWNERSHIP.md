# Ownership & review — who to ask about what

<!-- ▼▼▼ Update this block on every substantive edit. Keep it directly under the H1. ▼▼▼ -->

| | |
|---|---|
| **Revision** | `0.1` — first draft, names not yet filled in |
| **Last updated** | **2026-08-31** |
| **Status** | 🚧 **Placeholder.** Every `Responsible` cell is `_TBD_`. See [§4](#4-open-items). |
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
there is no `CODEOWNERS` file — one was drafted and the team decided against it. So the convention
holds only because people follow it. Two consequences worth stating:

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
| `configuration/SNT_metadata.json` | Whoever owns the results contract — a missing entry makes a column vanish silently |
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
| E | `snt_assemble_results` ⚠️ *being deprecated* | `_TBD_` | `_TBD_` |

Stages are those of [`DATA_ARCHITECTURE.md` §1.1](DATA_ARCHITECTURE.md#11-the-20-pipelines-at-a-glance).

---

## 3. Cross-cutting assets

Not owned by any one pipeline; a change here affects many.

| Asset | Responsible | Note |
|---|---|---|
| `code/snt_utils.r`, `code/snt_report.r`, `code/snt_palettes.r` | `_TBD_` | Sourced by most notebooks. A signature change breaks pipelines silently. |
| `configuration/SNT_config_<CC>.json` | `_TBD_` | Reference copies. The live file is renamed by hand inside each workspace. |
| `configuration/SNT_metadata.json` | `_TBD_` | Declares the results columns. An undeclared column is dropped without warning. |
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
variants](../CLAUDE.md#country-specific-notebook-variants)

---

## 4. Open items

`[TODO: Giulia]` — these need a decision or information that cannot be derived from the code.

- [ ] **Fill in every `_TBD_`** in [§2](#2-pipeline-responsibilities) and
      [§3](#3-cross-cutting-assets). Decide whether a `Backup` column is worth maintaining or
      should be dropped.
- [ ] **Confirm the licence.** [`../LICENSE`](../LICENSE) is currently **MIT**, added
      provisionally so a public repository is not left unlicensed. Rationale: maximally permissive,
      universally understood, and it lets ministries and national malaria programmes adapt the code
      without legal review. **Worth discussing with the team:**
      - **Apache-2.0** — same permissions, plus an explicit patent grant and contribution terms.
        The usual choice for organisation-backed public projects; slightly more paperwork.
      - **Consistency with sibling BLSQ repositories** (`openhexa`, `openhexa-toolbox`,
        `snt_utils`) — matching them is probably the strongest argument either way. Not verified
        here.
      - **The documentation.** This repo is unusually documentation-heavy. Some organisations
        license docs separately (CC-BY-4.0). Simpler to keep one licence for everything unless
        someone objects.
      - Confirm the copyright holder string and year in `LICENSE`.
- [ ] **Link the published method reference.** Neither the README nor the glossary points to an
      authoritative external description of the SNT method — WHO subnational-tailoring guidance, a
      protocol document, or a paper. External readers (epidemiologists reading the R code without
      workspace access) currently have this repository as their only context. Add the link to
      [`../README.md`](../README.md) §*Start here* and to [`GLOSSARY.md`](GLOSSARY.md) §1.1.
- [ ] **Delete `.github/CODEOWNERS`.** It names @sPuntinG as required approver for two notebooks,
      but the team decided not to adopt it — so it is either misleading (if GitHub is not
      enforcing it) or an unintended gate. Removing it makes [§1](#1-the-review-convention) the
      single statement of how review works. *Deliberately left in place for now: deleting it is a
      one-line PR that should be a conscious decision, not a side effect of writing this file.*

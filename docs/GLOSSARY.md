# GLOSSARY — domain vocabulary of `snt_development`

<!-- ▼▼▼ Update this block on every substantive edit. Keep it directly under the H1. ▼▼▼ -->

| | |
|---|---|
| **Revision** | `0.1` — first draft |
| **Last updated** | **2026-08-31** |
| **Status** | 🚧 **Draft — out for review.** 16 items still open in [§7](#7-open-questions). |
| **Verified against** | branch `SNT25-640_codebase-documentation`, at commit `11e10f0` (working tree clean) |
| **Owner** | @sPuntinG |

> **Reviewers:** this file is generated *from the code* for everything marked ✅, and from standard
> usage or guesswork for everything marked 🟡 / ❓. Your review is most valuable on the 🟡 and ❓
> entries — see [§7](#7-open-questions) for the shortlist. Please do not "fix" a ✅ entry without
> also changing the code it was read from.
>
> **Editors:** bump *Revision*, set *Last updated* to the edit date, and re-point *Verified against*
> at the commit you actually re-checked the ✅ entries against. A `Verified against` that lags the
> repository by many commits is the signal that this document has gone stale.

<!-- ▲▲▲ End of version block ▲▲▲ -->

---

The terms in this repository that **cannot be inferred from the code**: malaria-epidemiology
concepts, DHIS2 indicator codes, OpenHEXA platform vocabulary and the names of the statistical
methods. Everything a newcomer — human or agent — would otherwise have to guess at.

Companion documents: [`CLAUDE.md`](../CLAUDE.md) (working rules),
[`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md) (lineage and dataset contracts),
[`PIPELINE_README_STANDARD.md`](PIPELINE_README_STANDARD.md) (how to write a pipeline `readme.md`).

> **Related, but not authoritative here:** the team also keeps an
> [SNT Pipelines Data glossary](https://docs.google.com/spreadsheets/d/1qvZMsmCWU6cVLgGZTEXsd5xmoecIxb4LAd-g_2qzYdw/edit?usp=sharing)
> Google Sheet, linked from the incidence notebook. It is *not* synchronised with this file. When
> the two disagree, resolve it deliberately rather than assuming either side is stale.

---

## How to read this file

**Status** is the single most important column. It says how much you can trust the definition:

| Marker | Meaning |
|---|---|
| ✅ | **Code-confirmed.** The definition is derived from a formula, a `choices=[...]`, a config key or a `SNT_metadata.json` entry, and can be re-verified from the call sites given. |
| 🟡 | **Proposed.** Consistent with the code and with standard malaria-programme usage, but no artefact in this repo states it. Usable; confirm before quoting it in an operator-facing document. |
| ❓ | **Unconfirmed.** The term appears in the code, but its meaning is a guess. **Do not propagate it into a `readme.md`.** Listed here so it is visible, not so it is reusable. |

Every ❓ is collected again in [§7 Open questions](#7-open-questions) so they can be cleared in one pass.

**The `FR` column is deliberate.** A French version of the documentation is planned. Where a French
label already exists — almost all of them come from `configuration/SNT_metadata.json`, which is the
Explorer's user-facing vocabulary — it is recorded here so the English and French halves stay tied
to the same term. `—` means no French label exists yet, not that none is needed. When translating,
translate the *Definition* column and keep `Term` untouched: the terms are column names and
parameter values, and they do not localise.

---

## 1. Malaria epidemiology and SNT concepts

### 1.1. What SNT is

| Term | FR | Definition | Status |
|---|---|---|---|
| **SNT** | *Stratification nationale du paludisme / ciblage infranational* | **Subnational Tailoring.** The WHO-promoted exercise of stratifying a country by district and choosing the malaria intervention mix per stratum, rather than applying one national package. Everything in this repository exists to produce the one-row-per-ADM2 evidence table that exercise consumes. | 🟡 |
| **SNT Explorer** | — | The downstream web application that reads the results. Its column vocabulary is `configuration/SNT_metadata.json` — a results column absent from that file is silently dropped. Being migrated to read the OpenHEXA datasets directly instead of `snt_assemble_results`. | ✅ |
| **Routine data** | *Données de routine* | Monthly aggregate counts reported by health facilities into DHIS2 (cases, tests, treatments, deaths). The primary input. Distinct from *survey* data (DHS) and *modelled* data (MAP). | ✅ |

### 1.2. The incidence cascade — `N1`, `N2`, `N3`

The core calculation of the whole repository, in
[`pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb`](../pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb).
It follows the standard WHO approach for estimating malaria incidence from routine health
information systems (WHO, 2023). Each step corrects for one way the routine system *undercounts*
true cases, so each successive value should be **greater than or equal to** the previous one — the
notebook checks exactly that and warns when it is violated.

| Term | FR | Definition | Status |
|---|---|---|---|
| **TPR** | *Taux de positivité* | **Test Positivity Rate** = `CONF` / `TEST`. The share of malaria tests that came back positive. Should always be ≤ 1 — more confirmed than tested is a data error, and the notebook checks for it. | ✅ |
| **N1** | — | Cases **adjusted for testing gaps**: `N1 = CONF + (PRES × TPR)`. Presumed (untested) cases are counted at the rate at which tested patients turn out positive, instead of being counted in full or discarded. Computed monthly. | ✅ |
| **N2** | — | Cases **adjusted for testing and reporting**: `N2 = N1 / REPORTING_RATE`. Scales up for facilities that did not submit a report that month. A missing monthly reporting rate propagates `NA` into `N2`. | ✅ |
| **N3** | — | Cases **adjusted for testing, reporting and care-seeking**: `N3 = N2 / CARESEEKING`. Scales up for people who were ill but never presented at a facility. **Optional** — computed only when care-seeking data is supplied. | ✅ |
| **`N1_METHOD`** | — | Pipeline parameter choosing how presumed cases are obtained: `PRES` (a directly reported presumed-cases indicator) or `SUSP-TEST` (suspected minus tested, i.e. the suspected cases that were never tested). `PRES` requires the indicator to exist in *both* the config and the routine data; otherwise the notebook falls back to `SUSP-TEST`. | ✅ |
| **CSB / care-seeking** | *Recherche de soins ("taux recherche soins")* | **Care-Seeking Behaviour.** The proportion of people with malaria symptoms who seek care at a facility that reports into DHIS2. Supplied as an optional user file to `snt_dhis2_incidence` and used as the `N3` denominator. Related but **not identical** to the DHS `PCT_PUBLIC_CARE` / `PCT_PRIVATE_CARE` indicators. | 🟡 |
| **Crude incidence** | *Incidence brute* | `CONF / POPULATION × 1000` — confirmed cases per 1000 population per year, with no adjustment. The floor of the cascade. | ✅ |
| **Adjusted incidence** | *Incidence ajustée* | `N1` / `N2` / `N3` divided by population × 1000, giving `INCIDENCE_ADJ_TESTING`, `INCIDENCE_ADJ_REPORTING` and `INCIDENCE_ADJ_CARESEEKING` respectively. Note the naming is **cumulative**: `INCIDENCE_ADJ_REPORTING` is adjusted for testing *and* reporting. | ✅ |

> **Trap.** `INCIDENCE_ADJ_REPORTING` does not mean "adjusted for reporting only". Each name states
> the *last* adjustment applied, not the only one.

### 1.3. Reporting and completeness

| Term | FR | Definition | Status |
|---|---|---|---|
| **Reporting rate** | *Taux de rapportage* | Reports actually received by DHIS2 ÷ reports expected, for a given period and set of facilities. Two pipelines compute it by different routes — see [§4.3](#43-reporting-rate-methods). Must be **monthly** for the incidence pipeline to consume it. | ✅ |
| **Actual reports** | *Rapports reçus* | `ACTUAL_REPORTS` — the numerator: reports submitted. In the DHIS2 dataset method this is a DHIS2-computed metric identified by a UID in `REPORTING_RATE_PRODUCT_UID`. | ✅ |
| **Expected reports** | *Rapports attendus* | `EXPECTED_REPORTS` — the denominator: reports DHIS2 considers due. Its definition is the crux of the whole indicator; the three denominator choices are in [§4.3](#43-reporting-rate-methods). | ✅ |
| **Weighted reporting rate** | — | A reporting rate where each facility contributes in proportion to its *volume of activity* (measured on the `volume_activity_indicators`), rather than counting one facility as one report. Turns "80% of facilities reported" into "facilities accounting for 80% of the caseload reported". | ✅ |
| **Active facility** | *Formation sanitaire active* | A facility with a strictly positive value in a month on any of the selected `activity_indicators` (default `CONF`, `PRES`). Clinical activity, as opposed to merely existing in the pyramid. | ✅ |
| **Open facility** | — | A facility whose pyramid opening/closing dates bracket the month in question. Structural existence, as opposed to clinical activity. | ✅ |

### 1.4. Seasonality

| Term | FR | Definition | Status |
|---|---|---|---|
| **Seasonality** | *Saisonnalité* | Whether a district's malaria burden (or rainfall) is concentrated into a short part of the year. Computed twice, independently: `SEASONALITY_CASES` from confirmed cases, `SEASONALITY_RAINFALL` from ERA5 precipitation. Published as a 0/1 flag per ADM2. | ✅ |
| **Seasonal block** | *Bloc / période saisonnière* | A run of consecutive months (3–5, configurable) evaluated as a candidate concentration window. A district-year is seasonal if some block holds at least `threshold_for_seasonality` (default 0.6) of the annual total. | ✅ |
| **Block duration** | *Durée de la période saisonnière de transmission* | `SEASONAL_BLOCK_DURATION_RAINFALL` — the length in months of the qualifying block. Published as an SNT results column. | ✅ |
| **Onset month** | *Mois de début* | The first month of the qualifying seasonal block — when transmission season starts. Drives the timing of seasonal interventions. | 🟡 |
| **12-month forward-looking sliding window** | — | The default annualisation ("WHO approach"): each month starts a 12-month window used as the denominator, so a season straddling 31 December is not split. The alternative, `use_calendar_year_denominator = TRUE`, uses strict Jan–Dec. | ✅ |
| **Proportion of seasonal years** | — | `threshold_proportion_seasonal_years` (default 0.5): a district is seasonal *overall* only if at least this fraction of its individual years were classified seasonal. Guards against one anomalous year. | ✅ |

### 1.5. Interventions and survey indicators

| Term | FR | Definition | Status |
|---|---|---|---|
| **ITN** | *MII — Moustiquaire Imprégnée d'Insecticide* | **Insecticide-Treated Net.** Bed net treated with insecticide; the primary vector-control intervention. Reported both as **access** (a net is available in the household) and **use** (someone actually slept under it) — the gap between the two is itself a programme indicator. | 🟡 |
| **LLIN / MILDA** | *MILDA — Moustiquaire Imprégnée à Longue Durée d'Action* | **Long-Lasting Insecticidal Net** — an ITN whose insecticide survives repeated washing. Appears in this repo only as the DHIS2 indicator codes `MILDA_CPN` and `MILDA_VAR` ([§2](#2-dhis2-indicator-codes)). | ❓ |
| **IRS** | *PID — Pulvérisation Intra-Domiciliaire* | **Indoor Residual Spraying.** Spraying interior wall surfaces with a residual insecticide. In the results as `IRS_COVERAGE_RATE` (MAP), expressed per 100 households. | 🟡 |
| **RDT** | *TDR — Test de Diagnostic Rapide* | **Rapid Diagnostic Test.** Point-of-care malaria test. `PCT_U5_PREV_RDT_SAMPLE_AVERAGE` is DHS-measured prevalence in children 6–59 months by RDT. | ✅ |
| **DTP1 / DTP2 / DTP3** | *DTC1 / DTC2 / DTC3* | Doses 1, 2 and 3 of the **diphtheria–tetanus–pertussis** vaccine. Not malaria indicators: they are used as a **proxy for health-system reach and routine-service contact**, and the *dropout* between doses as a proxy for continuity of care. | 🟡 |
| **DTP dropout** | *Abandon vaccinal* | `PCT_DROPOUT_DTP_1_2` / `_2_3` / `_1_3` — the share of children who received the earlier dose but not the later one. | ✅ |
| **U5MR** | *Mortalité infanto-juvénile* | **Under-5 Mortality Rate**: probability a child dies before their fifth birthday, per 1000 live births, over the 10 years preceding the survey. DHS-derived. | ✅ |
| **PfPR** | *Prévalence du Pf* | ***P. falciparum* Parasite Rate** — the share of a population carrying *Plasmodium falciparum* parasites. Conventionally standardised to ages 2–10 (`PfPR₂₋₁₀`), which is what `PF_PR_RATE` (MAP) reports. The standard measure of transmission intensity. | ✅ |
| **Pf** | — | ***Plasmodium falciparum***, the malaria parasite species responsible for nearly all severe disease and death in sub-Saharan Africa. `PF_INCIDENCE_RATE`, `PF_MORTALITY_RATE`, `PF_PR_RATE` are all MAP model outputs for this species specifically. | 🟡 |
| **Antimalarial EFT** | — | `ANTIMALARIAL_EFT_RATE` (MAP), "Treatment with Antimalarial", per 100 cases. **EFT** is most likely *Effective Treatment* — the share of cases receiving an effective antimalarial. **Unconfirmed.** | ❓ |
| **`SAMPLE_AVERAGE` suffix** | — | On a DHS-derived results column, marks a value averaged across the survey sample rather than a modelled or routine figure (e.g. `U5MR_PERMIL_SAMPLE_AVERAGE`). Exact weighting is defined in the per-family computation notebooks under `pipelines/snt_dhs_indicators/code/`, which are the source of truth. | 🟡 |

### 1.6. Quality of care

All computed at **district-year** grain by `snt_dhis2_quality_of_care`, from outlier-treated routine
data. Rates return `NA` rather than zero when the denominator is zero.

| Term | FR | Definition | Status |
|---|---|---|---|
| **Testing rate** | *Taux de dépistage* | `TEST / SUSP` — of the suspected cases, how many were actually tested. | ✅ |
| **Treatment rate** | *Taux de traitement* | `MALTREAT / CONF` — of the confirmed cases, how many were treated. | ✅ |
| **Case fatality rate** | *Létalité* | `MALDTH / MALADM` — of malaria patients **admitted** to hospital, how many died. Note the denominator is admissions, not all cases: this is in-facility severe-case lethality, not population mortality. | ✅ |
| **`prop_adm_malaria`** | — | `MALADM / ALLADM` — malaria's share of all hospital admissions. A burden-of-disease proxy. | ✅ |
| **`prop_malaria_deaths`** | — | `MALDTH / ALLDTH` — malaria's share of all in-facility deaths. (The notebook also emits `prop_deaths_malaria` as an alias.) | ✅ |

---

## 2. DHIS2 indicator codes

The keys of `DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS` in
`configuration/SNT_config_<CC>.json`. Each maps a short SNT code to a **list of DHIS2 data-element
UIDs for that country** — the mapping is per-country, and an empty list means the country does not
report that indicator. The codes then become the `INDICATOR` values (long form) or column names
(wide form) throughout the routine data.

Several codes are French-derived abbreviations, which is why they do not decode from the English.

### 2.1. Core malaria case codes

These four drive the incidence cascade and are the ones that appear in `choices=[...]`.

| Code | FR | Definition | Status |
|---|---|---|---|
| `SUSP` | *Cas suspects* | **Suspected** malaria cases — presented with symptoms consistent with malaria. | ✅ |
| `TEST` | *Cas testés* | **Tested** cases — suspected cases that received a diagnostic test (RDT or microscopy). | ✅ |
| `CONF` | *Cas confirmés* | **Confirmed** cases — tested and positive. The numerator of crude incidence and the input to seasonality. | ✅ |
| `PRES` | *Cas présumés* | **Presumed** cases — treated as malaria without diagnostic confirmation. Either reported directly, or derived as `SUSP - TEST`; the `N1_METHOD` parameter chooses. | ✅ |

### 2.2. Severity, treatment and outcome codes

| Code | FR | Definition | Status |
|---|---|---|---|
| `PRESSEV` | — | Presumed **severe** malaria cases. (`PRES` + *sévère*.) Empty in every config copy in this repo. | 🟡 |
| `MALSIMP` | *Paludisme simple* | **Uncomplicated** (simple) malaria cases. | 🟡 |
| `MALSEV` | *Paludisme sévère* | **Severe** malaria cases. | 🟡 |
| `MALTREAT` | — | Malaria cases **treated** (`MAL` + *treatment*). Numerator of the treatment rate. | ✅ |
| `MALADM` | — | Malaria **admissions** — malaria cases hospitalised. Denominator of case fatality. | ✅ |
| `MALDTH` | — | Malaria **deaths** recorded in facilities. | ✅ |
| `CASRECU` | — | (`CAS` + *reçus*?) **Cases received.** Meaning and relation to `SUSP` unconfirmed; empty in every config copy here. | ❓ |

### 2.3. All-cause denominators

Used by `snt_dhis2_quality_of_care` to express malaria as a share of total facility activity.

| Code | FR | Definition | Status |
|---|---|---|---|
| `ALLOUT` | — | **All-cause outpatient** consultations. In the quality-of-care output it is carried as `non_malaria_all_cause_outpatients`. | ✅ |
| `ALLADM` | — | **All-cause admissions**. Denominator of `prop_adm_malaria`. | ✅ |
| `ALLDTH` | — | **All-cause deaths** in facility. Denominator of `prop_malaria_deaths`. | ✅ |

> ⚠️ **Naming mismatch, worth confirming.** The quality-of-care column is named
> `non_malaria_all_cause_outpatients` but is a direct district-year sum of `ALLOUT`, with no malaria
> subtraction anywhere in the notebook. Either the column name or the computation is wrong. Flagged
> in [§7](#7-open-questions).

### 2.4. Antenatal care and prevention codes

These are French-derived and appear in only some country configs.

| Code | FR | Definition | Status |
|---|---|---|---|
| `CPN` / `CPN1` | *CPN — Consultation PréNatale* | **Antenatal care** visits (`CPN1` = first visit). Used as a denominator for pregnancy-related malaria prevention coverage. | 🟡 |
| `TPI1` / `TPI3` | *TPI — Traitement Préventif Intermittent* | **Intermittent Preventive Treatment** in pregnancy (IPTp), doses 1 and 3. Preventive antimalarial given at antenatal visits. | 🟡 |
| `MILDA_CPN` | — | LLINs distributed **at antenatal consultations**. | 🟡 |
| `MILDA_VAR` | — | LLINs distributed at… (`VAR` = *vaccination anti-rougeoleuse*, measles vaccination?) **Unconfirmed.** | ❓ |
| `VAR1` | — | (*Vaccination Anti-Rougeoleuse*, measles dose 1?) **Unconfirmed.** | ❓ |
| `UTILISATION` | *Utilisation* | A service-utilisation indicator. Scope (which service, which denominator) **unconfirmed**. | ❓ |

### 2.5. Disaggregation suffixes

Any core code may carry an age or pregnancy suffix. **Spelling is inconsistent across country
configs** — both `_UNDER5` and `_UNDER_5` occur, as do `_ABOVE5` and `_ABOVE_5`. Read the target
country's config rather than assuming.

| Suffix | FR | Definition | Status |
|---|---|---|---|
| `_UNDER_5` / `_UNDER5` | *moins de 5 ans* | Children under 5 years old — the highest-risk group and the default population for stratification. | ✅ |
| `_ABOVE_5` / `_ABOVE5` | *5 ans et plus* | Age 5 and over. | ✅ |
| `_PREGNANT_WOMAN` | *femmes enceintes* | Pregnant women — the second high-risk group. | ✅ |

> 🐛 **Known defect.** Selecting "Pregnant Women" in `snt_dhis2_incidence` **fails**: the indicator
> suffix is `_PREGNANT_WOMAN` (singular) but the population column every producer writes is
> `POP_PREGNANT_WOMEN` (plural). It stops loudly, so no bad data is produced. See
> [`DATA_ARCHITECTURE.md` §6.1](DATA_ARCHITECTURE.md) and `CLAUDE.md` → Traps.

### 2.6. Population codes

Keys of `DHIS2_DATA_DEFINITIONS.POPULATION_INDICATOR_DEFINITIONS`. Denominators for every rate.

| Code | Definition | Status |
|---|---|---|
| `POPULATION` | Total population of the org unit. The only column `snt_dhis2_population_transformation` operates on. | ✅ |
| `POP_UNDER_5` | Population under 5 years. | ✅ |
| `POP_PREGNANT_WOMEN` | Population of pregnant women (note: **plural**, unlike the indicator suffix). | ✅ |
| `POP_0_1_Y`, `POP_1_2_Y`, `POP_5_10_Y`, `POP_50_PLUS` | Age bands, in years. | ✅ |
| `POP_5_36_M` | Age band **in months** (5–36 months) — the odd one out; the `_M` is the unit. | 🟡 |

---

## 3. Structural and platform vocabulary

### 3.1. Geography and the org-unit hierarchy

| Term | FR | Definition | Status |
|---|---|---|---|
| **Org unit (OU)** | *Unité d'organisation* | DHIS2's unit of reporting geography. In this repo `OU_ID` almost always means a **health facility**, the leaf of the hierarchy. | ✅ |
| **Pyramid** | *Pyramide sanitaire* | The DHIS2 organisation-unit **hierarchy** — the country's nested administrative and health-facility tree, with each unit's level, parent, name, coordinates and opening/closing dates. Extracted as `{CC}_pyramid.parquet` and used to know which facilities *should* have reported. | ✅ |
| **FOSA** | *FOrmation SAnitaire* | **Health facility** (French). The term is used interchangeably with "facility" in `snt_healthcare_access`, whose `input_fosa_file` supplies facility coordinates when the pyramid's are unusable. | ✅ |
| **ADM1** | *Niveau administratif 1* | First subnational administrative level — region/province. The grain of DHS-derived indicators (survey samples are not powered below it). | ✅ |
| **ADM2** | *District sanitaire* | Second subnational administrative level — **health district**. **The grain of the whole SNT exercise**: the final results table is one row per `ADM2_ID`, and it is the unit at which interventions are tailored. | ✅ |
| **Shapes** | *Contours / limites administratives* | ADM2 polygon geometries, published as `{CC}_shapes.geojson` by `snt_dhis2_formatting`. Every external-source pipeline (ERA5, MAP, WorldPop, healthcare access) starts by fetching this file — they look like independent roots but are not. | ✅ |

> **The two admin-level config keys are not interchangeable.** `ANALYTICS_ORG_UNITS_LEVEL` (an
> integer, e.g. `6`) is the **facility** level at which routine analytics are extracted.
> `DHIS2_ADMINISTRATION_2` (a string, e.g. `"level_4_name"`) is the **district** level at which
> population, shapes and reporting indicators live. `DHIS2_ADMINISTRATION_1` is the region level.
> All three differ per country. Never hardcode any of them (rule **R13**).

### 3.2. Time

| Term | Definition | Status |
|---|---|---|
| **`PERIOD`** | A DHIS2 period string. In this repo effectively always monthly, `YYYYMM` (e.g. `202403`). | ✅ |
| **`YEAR` / `MONTH`** | The decomposed period, as integers. Join keys alongside `ADM2_ID`. | ✅ |
| **Epi week** | **Epidemiological week** — a week numbered so that surveillance data aligns year to year, rather than by calendar date. In `snt_era5_climate_data` it is DHIS2's Thursday-anchored week type (`WeekType.WEEK_THURSDAY`), as opposed to the plain `WEEK`. Written to disk as `{CC}_{variable}_epi_weekly.parquet` but **not published to any dataset** — monthly is the only published ERA5 grain. | ✅ |

### 3.3. OpenHEXA

| Term | Definition | Status |
|---|---|---|
| **Workspace** | An OpenHEXA tenant. **One workspace = one country**, holding its own `SNT_config.json`, its `data/` filesystem, its datasets and its DHIS2 connection. | ✅ |
| **Dataset** | OpenHEXA's versioned, named file store. **The contract between pipelines** — a file not added via `add_files_to_dataset(...)` is invisible downstream, regardless of whether it exists in `data/`. Identifiers are in `SNT_DATASET_IDENTIFIERS`. | ✅ |
| **Template pipeline** | The published, versioned artefact that country workspaces subscribe to. Publishing from `snt-development` issues a new version of the SNT template; publishing from anywhere else creates a competing duplicate (rule **R5**). | ✅ |
| **`Pull scripts`** | The operator-facing flag that refreshes notebooks and `.r` helpers from this repository into the workspace. **Merging to `main` does not update any workspace's R code** — only a run with this flag ON does. Overwrites local workspace edits. | ✅ |
| **Papermill** | The tool that executes a notebook with parameters injected from `pipeline.py`. Why every notebook needs an `if (!exists("X")) X <- …` fallback cell (rules **R11**/**R12**) — so it stays runnable by hand. | ✅ |
| **`snt_lib`** | The shared Python helper library, `github.com/BLSQ/snt_utils`. External to this repo and **unpinned**. Read the upstream source or an existing call site; do not guess signatures. | ✅ |

### 3.4. External data sources

| Term | Definition | Status |
|---|---|---|
| **DHIS2** | *District Health Information Software 2* — the open-source national health information system that holds the routine data. The primary source. | ✅ |
| **DHS** | *Demographic and Health Surveys* — nationally representative household surveys (USAID-funded). Source of care-seeking, ITN, vaccination, mortality and prevalence indicators, at **ADM1** grain only. | ✅ |
| **MAP** | *Malaria Atlas Project* — modelled, gridded malaria surfaces (parasite rate, incidence, mortality, ITN and IRS coverage). Downloaded as rasters and reduced to ADM2 by zonal statistics. **Modelled, not measured** — it is an independent cross-check on the routine data, not a second observation of it. | ✅ |
| **WorldPop** | Gridded population estimates (~100 m). Supplies the population raster used to weight zonal statistics and to compute healthcare-access coverage. | ✅ |
| **ERA5 / ERA5-Land** | ECMWF's global climate **reanalysis** — a modelled reconstruction of past weather on a regular grid. Source of the precipitation and temperature series behind rainfall seasonality. Fetched from the **CDS** (Copernicus Climate Data Store). | ✅ |
| **UN-adjusted** | A WorldPop raster variant rescaled so national totals match UN population projections. | 🟡 |
| **Constrained PPP** | The WorldPop product used here: **P**eople **P**er **P**ixel, *constrained* meaning population is allocated only to cells where buildings were detected, rather than smoothed across uninhabited land. | 🟡 |

---

## 4. Method names

### 4.1. Outlier detection and imputation

Five pipelines implement five different methods. All write the **same three filenames** to the
**same dataset** — this is deliberate: the analyst tries alternatives and the last run wins.
The file alone does not say which method produced it; the `{CC}_parameters.json` beside it does.

| Method | Pipeline | How it flags an outlier | Status |
|---|---|---|---|
| **mean ± k·SD** | `snt_dhis2_outliers_imputation_mean` | Value outside `mean ± k × SD`, computed per `OU_ID × INDICATOR` over the facility's whole history. `k` = `deviation_mean`, default 3. Flag column `OUTLIER_MEAN{k}SD`. | ✅ |
| **median ± k·MAD** | `snt_dhis2_outliers_imputation_median` | Same shape, but around the **median** using **MAD** — robust to the very outliers being looked for. `k` = `deviation_median`, default 3. | ✅ |
| **IQR fences** | `snt_dhis2_outliers_imputation_iqr` | Value outside `[q1 − m·IQR, q3 + m·IQR]`, quartiles per `OU_ID × INDICATOR`. `m` = `deviation_iqr`, default 1.5. Flag column `OUTLIER_IQR{m}`. | ✅ |
| **Magic Glasses** | `snt_dhis2_outliers_imputation_magic_glasses` | Staged. **MAD15** on all rows, then **MAD10** on what MAD15 left → `OUTLIER_MAD15_MAD10`. In `complete` mode, seasonal decomposition passes (**seasonal5**, then **seasonal3**) on the survivors → `OUTLIER_SEASONAL5_SEASONAL3`. `partial` ≈ 7 min; `complete` can take hours. Adapted from **[Magic Glasses 2](https://github.com/jpainter/MagicGlasses2)** — see the note below. | ✅ |
| **PATH** | `snt_dhis2_outliers_imputation_path` | Trims to the **central 80%** of positive monthly values per `OU × INDICATOR`, derives `MEAN_80` / `SD_80` from that trimmed sample, and flags `VALUE > MEAN_80 + k × SD_80`. Then applies **epidemiological exception rules** for suspected stock-outs and epidemics, so a genuine outbreak is not erased as noise. Developed with **[PATH](https://www.path.org/)** — see the note below. | ✅ |

> **📌 Magic Glasses — scope of the adaptation.** The upstream tool,
> **[Magic Glasses 2](https://github.com/jpainter/MagicGlasses2)**, is a full DHIS2 data-quality
> application; SNT reimplements **only the part concerning outlier detection**, and only partly.
> Do not describe this pipeline as "running Magic Glasses", and do not assume upstream behaviour
> that is not in the notebook. If a definition is needed, the SNT notebook is the source of truth
> for what SNT does; the repository above is the source of truth for the original method.

> **📌 PATH — [TODO: SNT team] this method needs a fuller write-up.** The method comes from the
> team's collaboration with [PATH](https://www.path.org/), and what is documented here was
> reconstructed from the code alone. Still missing, and best filled in by whoever worked on the
> collaboration:
>
> - The **rationale** for the central-80% trim and for `MEAN_DEVIATION = 10` — why these and not
>   the conventional 3 SD.
> - The exact **stock-out** and **epidemic** exception rules, in epidemiological rather than
>   algorithmic terms (`detect_possible_stockout`, `detect_possible_epidemic`).
> - The **low-count floors** that suppress flags (`TEST`/`PRES` < 50, `CONF` < 10) — where the
>   thresholds come from.
> - Whether there is a **citable PATH publication or internal note** to point at, and how faithful
>   this implementation is to it.
> - The known **parameter-wiring mismatch**: `pipeline.py` injects `DEVIATION_MEAN` while the R
>   notebook reads `MEAN_DEVIATION` (default 10), so the OpenHEXA field may not be driving the
>   live threshold. Documented in the pipeline's own
>   [`readme.md`](../snt_dhis2_outliers_imputation_path/readme.md); it should be fixed, not just
>   described.

| Term | Definition | Status |
|---|---|---|
| **MAD** | **Median Absolute Deviation** — the median of the absolute deviations from the median. A robust alternative to standard deviation: a few extreme values barely move it. | ✅ |
| **MAD15 / MAD10** | Magic Glasses stages at thresholds 15 and 10 MADs — a very permissive first pass (only egregious values), then a stricter one on what remains. | ✅ |
| **seasonal5 / seasonal3** | Magic Glasses `complete`-mode passes that flag deviations from a **seasonally decomposed** expectation, at thresholds 5 and 3. They catch values that are wrong *for that month of the year* but unremarkable against the full-year spread. | 🟡 |
| **"Magic Glasses"** | Named after **[Magic Glasses 2](https://github.com/jpainter/MagicGlasses2)**, an open-source DHIS2 data-quality tool. SNT replicates *part* of its outlier-detection process — Magic Glasses 2 does considerably more. Treat it as the upstream reference for the method, not as a description of what this pipeline does. | ✅ |
| **"PATH"** | Named after **[PATH](https://www.path.org/)**, the global health non-profit; the method comes out of the SNT team's collaboration with them. | ✅ |
| **`detected` / `imputed` / `removed`** | The three output files. `detected` is a long table of flags with `OUTLIER_DETECTED` and `OUTLIER_METHOD`; `imputed` replaces flagged values with an estimate; `removed` nulls them. Downstream pipelines choose between `imputed` and `removed`. | ✅ |
| **Imputation** | Replacing a flagged value with a plausible substitute rather than dropping it. Magic Glasses uses a **three-month centred rolling mean** of the non-outlier values; PATH substitutes `MEAN_80`. | ✅ |
| **Possible stock-out** | A PATH exception: a drop consistent with a facility running out of tests or drugs, rather than a genuine fall in cases. | 🟡 |
| **Possible epidemic** | A PATH exception: a spike consistent with a real outbreak, which must **not** be smoothed away. | 🟡 |
| **SARIMA** | *Seasonal AutoRegressive Integrated Moving Average* — the time-series model used by the seasonality pipelines to fill gaps in a district's monthly series before classifying it, subject to a cap on how much may be missing. | 🟡 |

### 4.2. Routine-data selection vocabulary

> ⚠️ **The same concept has three different names.** "Routine data with outliers removed" is spelled
> differently in three places. Check the target pipeline's `choices=[...]` before assuming. This is a
> known violation of rule **R15**, logged for migration in `CLAUDE.md`.

| Concept | `snt_dhis2_incidence` | `snt_dhis2_reporting_rate_*` | `snt_dhis2_quality_of_care` |
|---|---|---|---|
| Parameter name | `routine_data_choice` | `routine_data_choice` | **`data_action`** |
| Formatted, untreated | `raw` | `raw` | *(not offered)* |
| Outliers nulled | **`raw_without_outliers`** | **`outliers_removed`** | **`removed`** |
| Outliers replaced | `imputed` | `imputed` | `imputed` |

### 4.3. Reporting-rate methods

| Term | Definition | Status |
|---|---|---|
| **Dataset method** | `snt_dhis2_reporting_rate_dataset` — uses DHIS2's **own** reporting-rate metrics (`ACTUAL_REPORTS` / `EXPECTED_REPORTS`, identified by `REPORTING_RATE_PRODUCT_UID`). Takes DHIS2's definition of "expected" as given. Can have incomplete coverage, which propagates `NA` into `N2`. | ✅ |
| **Data element method** | `snt_dhis2_reporting_rate_dataelement` — **reconstructs** the rate: builds a full `PERIOD × OU_ID` grid from the pyramid and counts a facility as having reported if it has a positive value on the chosen `activity_indicators`. Independent of how DHIS2 configures datasets. | ✅ |
| `ROUTINE_ACTIVE_FACILITIES` | `dataelement_method_denominator` choice. Denominator = facilities seen to be clinically active in the routine data. | ✅ |
| `PYRAMID_OPEN_FACILITIES` | `dataelement_method_denominator` choice. Denominator = facilities the pyramid says were open, whether or not they reported anything. Stricter, and usually gives a lower rate. | ✅ |

> **Do not reuse the deprecated vocabulary.** The single pre-split pipeline
> (`deprecated/DEPRECATED_snt_dhis2_reporting_rate`) chose between the two routes with a
> `reporting_rate_method` parameter (`DATASET` / `DATAELEMENT`) and offered a third denominator,
> `DHIS2_EXPECTED_REPORTS`. None of these exist in the live pipelines — the two routes are now two
> separate pipelines. `deprecated/` is history, never a template.

### 4.4. Spatial methods

| Term | Definition | Status |
|---|---|---|
| **Zonal statistics** | Reducing a raster to one value per polygon — the mean (or sum) of the grid cells falling inside each ADM2. How every gridded source (MAP, WorldPop, ERA5) becomes an ADM2 column. | ✅ |
| **Population-weighted** | A zonal mean where each cell is weighted by its population, so the figure describes the average *person* in the district rather than the average *hectare*. Requires the metric grid to be aligned to the population raster first. Added as a `population_weighted` column by `snt_map_extracts`. | ✅ |
| **Healthcare access / population covered** | The share of a district's population living within a **straight-line (Euclidean) buffer** — 5000 m — of an active facility. **Not** a travel-time or road-network measure; do not describe it as one. | ✅ |
| **CRS** | *Coordinate Reference System*. Buffers are computed in a **metric** CRS (so radii are in metres); the notebook fixes which one. Very large countries may need methodological review at the edges. | ✅ |
| **LCI / UCI** | **Lower** and **Upper Confidence Interval** bands, delivered by MAP as separate raster bands alongside `Data`. Zonal statistics are computed for each band, so the uncertainty survives the aggregation. | 🟡 |
| **`GRAY_INDEX`** | A raster band name that appears in some MAP products, handled alongside `Data`/`LCI`/`UCI`. Its meaning in MAP's schema is **unconfirmed**. | ❓ |

---

## 5. Results-table columns

The 29 columns of the final ADM2 results table, declared in `configuration/SNT_metadata.json`.
That file is authoritative: a column absent from it is **silently dropped** by
`snt_assemble_results`. French labels below are quoted verbatim from it and are what an SNT
Explorer user sees.

| Column | Source | FR label | Units | Status |
|---|---|---|---|---|
| `POPULATION` | DHIS2 | Population totale | persons | ✅ |
| `U5MR_PERMIL_SAMPLE_AVERAGE` | DHS | Mortalité infanto-juvénile | per 1000 live births | ✅ |
| `REPORTING_RATE` | DHIS2 | Taux de rapportage | proportion | ✅ |
| `PCT_PUBLIC_CARE` | DHS | Recherche des soins dans le secteur public | per 100 children | ✅ |
| `PCT_PRIVATE_CARE` | DHS | Recherche des soins dans le secteur privé | per 100 children | ✅ |
| `PCT_NO_CARE` | DHS | Non-recherche des soins | per 100 children | ✅ |
| `INCIDENCE_CRUDE` | DHIS2 | Incidence brute | per 1000 persons | ✅ |
| `INCIDENCE_ADJ_TESTING` | DHIS2 | Incidence ajustée pour le dépistage | per 1000 persons | ✅ |
| `INCIDENCE_ADJ_REPORTING` | DHIS2 | Incidence ajustée pour le taux de rapportage | per 1000 persons | ✅ |
| `INCIDENCE_ADJ_CARESEEKING` | DHIS2 | Incidence ajustée pour la recherche de soins | per 1000 persons | ✅ |
| `PF_INCIDENCE_RATE` | MAP | Incidence du Pf | per 1000 persons | ✅ |
| `PF_PR_RATE` | MAP | Prévalence du Pf chez les enfants de 2 à 10 ans | per 100 children | ✅ |
| `PCT_U5_PREV_RDT_SAMPLE_AVERAGE` | DHS | Prévalence du paludisme chez les enfants de 6 à 59 mois | per 100 children | ✅ |
| `PF_MORTALITY_RATE` | MAP | Taux de mortalité par Pf | per 100 000 persons | ✅ |
| `SEASONALITY_CASES` | DHIS2 | Saisonnalité des cas | 0 / 1 | ✅ |
| `SEASONALITY_RAINFALL` | ERA5 | Saisonnalité des précipitations | 0 / 1 | ✅ |
| `SEASONAL_BLOCK_DURATION_RAINFALL` | ERA5 | Durée de la période saisonnière de transmission | months | ✅ |
| `PCT_DTP1/2/3_SAMPLE_AVERAGE` | DHS | Vaccination DTP1/2/3 | per 100 children | ✅ |
| `PCT_DROPOUT_DTP_1_2 / _2_3 / _1_3` | DHS | Abandon vaccinal entre DTP*n* et DTP*m* | per 100 children | ✅ |
| `PCT_ITN_ACCESS_SAMPLE_AVERAGE` | DHS | Accès aux moustiquaires imprégnées d'insecticide | per 100 persons | ✅ |
| `PCT_ITN_USE_SAMPLE_AVERAGE` | DHS | Utilisation des moustiquaires imprégnées d'insecticide | per 100 persons | ✅ |
| `ITN_ACCESS_RATE` | MAP | Accès aux moustiquaires imprégnées d'insecticide | per 100 persons | ✅ |
| `ITN_USE_RATE_RATE` | MAP | Utilisation des moustiquaires imprégnées d'insecticide | per 100 persons | ✅ |
| `IRS_COVERAGE_RATE` | MAP | Pulvérisation intra-domiciliaire | per 100 households | ✅ |
| `ANTIMALARIAL_EFT_RATE` | MAP | *Treatment with Antimalarial* | per 100 cases | ✅ |

> **Two things to note.** ITN access and use appear **twice** — once from DHS (survey-measured,
> ADM1, `PCT_*_SAMPLE_AVERAGE`) and once from MAP (modelled, ADM2, `*_RATE`). They are not
> duplicates and should not be reconciled. And `ITN_USE_RATE_RATE` has a doubled suffix, while
> `ANTIMALARIAL_EFT_RATE` is the one entry whose French `LABEL` was never translated — both are in
> [§7](#7-open-questions).

---

## 6. Abbreviations, at a glance

| | | | |
|---|---|---|---|
| **ADM1** region/province | **ADM2** health district | **CDS** Copernicus Climate Data Store | **CPN** consultation prénatale (ANC) |
| **CRS** coordinate reference system | **CSB** care-seeking behaviour | **DHIS2** District Health Information Software 2 | **DHS** Demographic and Health Surveys |
| **DTP** diphtheria–tetanus–pertussis | **EFT** effective treatment ❓ | **FOSA** formation sanitaire (health facility) | **IQR** interquartile range |
| **IRS** indoor residual spraying | **ITN** insecticide-treated net | **LCI/UCI** lower/upper confidence interval | **LLIN/MILDA** long-lasting insecticidal net |
| **MAD** median absolute deviation | **MAP** Malaria Atlas Project | **OU** organisation unit | **Pf** *Plasmodium falciparum* |
| **PfPR** *P. falciparum* parasite rate | **PPP** people per pixel (WorldPop) | **RDT/TDR** rapid diagnostic test | **SARIMA** seasonal ARIMA |
| **SNT** subnational tailoring | **TPI/IPTp** intermittent preventive treatment | **TPR** test positivity rate | **U5MR** under-5 mortality rate |

---

## 7. Open questions

Everything marked ❓ above, plus inconsistencies the sweep turned up. Each needs a person with the
domain knowledge — not another pass over the code.

**Indicator codes whose meaning is genuinely unknown:**

1. `CASRECU` — "cas reçus"? How does it differ from `SUSP`? (Empty in all five config copies.)
2. `UTILISATION` — utilisation of *what*, over *what* denominator?
3. `VAR1` — measles vaccination dose 1 (*vaccination anti-rougeoleuse*)?
4. `MILDA_VAR` — LLINs distributed at measles-vaccination contacts? If so, `MILDA_CPN` and
   `MILDA_VAR` are the two routine distribution channels, which is worth saying explicitly.
5. `PRESSEV`, `MALSIMP`, `MALSEV`, `CPN`/`CPN1`, `TPI1`/`TPI3` — proposed definitions above are
   from standard usage, not from anything in the repo. Confirm or correct.
6. `POP_5_36_M` — confirm the band is 5–36 **months** and not years.

**Method provenance:**

7. How closely does the SNT **MAD15 → MAD10 → seasonal5 → seasonal3** chain track
   [Magic Glasses 2](https://github.com/jpainter/MagicGlasses2)? Were the thresholds taken from
   upstream, or chosen here?
8. What exactly do the **seasonal5 / seasonal3** passes decompose against?
9. The **PATH** method needs a fuller write-up — rationale for the central-80% trim and the
   deviation multiplier, the stock-out and epidemic rules in epidemiological terms, the low-count
   floors, and a citable reference if one exists. Itemised in
   [§4.1](#41-outlier-detection-and-imputation); for the SNT team.

**Definitions that affect how results are read:**

10. **EFT** in `ANTIMALARIAL_EFT_RATE` — "effective treatment"? Its `LABEL` in `SNT_metadata.json`
    is the only English one in an otherwise French file, which suggests it was never finished.
11. **CSB vs DHS care-seeking** — is the `N3` care-seeking denominator meant to be
    `PCT_PUBLIC_CARE`, some combination, or always an externally supplied national figure?
12. `GRAY_INDEX` — what MAP delivers in that band, and whether averaging it is meaningful.

**Naming problems found during the sweep** (each is a code question, not just a glossary one):

13. `non_malaria_all_cause_outpatients` in `snt_dhis2_quality_of_care` is a plain sum of `ALLOUT`
    with no malaria subtraction. Is the name wrong, or the computation?
14. `ITN_USE_RATE_RATE` — doubled suffix in `SNT_metadata.json`. Renaming it is operator-visible,
    so it needs a migration rather than an edit.
15. `_PREGNANT_WOMAN` (indicator suffix, singular) vs `POP_PREGNANT_WOMEN` (population column,
    plural) — the confirmed defect in `select_population_column()`. Fixing the code would also
    let this glossary state one spelling.
16. `_UNDER5` vs `_UNDER_5` and `_ABOVE5` vs `_ABOVE_5` vary across country configs. Is one
    canonical, or are both genuinely in use in the respective DHIS2 instances?

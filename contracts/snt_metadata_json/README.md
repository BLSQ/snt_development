# `SNT_metadata.json` — schema and conventions

> **Status: work in progress.** The shape described here is the one dated **2026-09-08**, currently
> being validated with the IASO developers. It is written down so the work is not lost and can be
> built on progressively — not because it is settled. The open questions are the `META-*` items in
> [`docs/BACKLOG.md`](../../docs/BACKLOG.md#snt-explorer-metadata); do not treat a silence there as a
> decision. Why the format looks like this: [ADR 0007](../../docs/decisions/0007-snt-metadata-json-format.md).

| File | Role |
|---|---|
| [`snt_metadata.schema.json`](snt_metadata.schema.json) | **Source of truth for the shape.** JSON Schema (draft 2020-12) describing the 2026-09-08 format and nothing else. Every field carries a `description`. |
| [`SNT_metadata.example.json`](SNT_metadata.example.json) | The canonical valid instance — the 2026-09-08 file, verbatim except that its `//` comments were removed. Everything those comments said is preserved in the schema's `description` fields and in this document. |
| [`SNT_metadata_NER.json`](SNT_metadata_NER.json) | **Niger's catalogue**, converted from Niger's old-format file. Nine layers, schema-valid. Read [§6](#snt_metadata_nerjson) before using it: three MAP layers could not be carried over, and two seasonality layers reference a table with no `YEAR` column. |
| [`catalogs/SNT_metadata_master.json`](../../catalogs/SNT_metadata_master.json) | **The master list**: every layer the current pipelines can publish in the wide "one layer = one column" shape — 58 layers, schema-valid. Text is a **draft written from the code**, awaiting review; `CATEGORY` is empty and scales are defaults. It is what `snt_metadata_json_trim` trims per workspace. See [`catalogs/README.md`](../../catalogs/README.md), and [§6](#the-master-list) for what it leaves out and why. |

> **Not yet in the schema:** the SNT Explorer also supports a third `TYPE`, `"Linear"` (gradient
> legend, `SCALE` = `[start, end]`). It is blocked for now by an Explorer-side bug, so the schema enum
> still rejects it. See [§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each).

**Strict JSON only — no comments.** The SNT Explorer does not accept JSONC, so a `//` comment
anywhere in the file breaks ingestion. Notes belong in the schema's `description` fields or in this
document, never in the data file. (They would not survive the future editing webapp either: an app
that reads, edits and rewrites JSON drops comments silently.)

**These two files describe the target format. They do not describe what is deployed today.** The
old-format `SNT_metadata.json` (repo copy archived in `deprecated/configuration/`) is a different,
older shape and has not been migrated. A workspace may still hold an old-format copy in its
`configuration/` folder; see [BACKLOG DOC-3](../../docs/BACKLOG.md#doc-3) for what still reads it.

---

## 1. What this file is for

`SNT_metadata.json` is the catalogue of **data layers** the SNT Explorer offers a user for one
country. The Explorer is a read-only exploration tool — click around, combine layers, compare
districts — so this file affects presentation and nothing else. It never changes the underlying
data, and no analytics pipeline reads it.

One entry does two jobs at once:

- **Where the values are** — `SOURCE_DATA`: dataset, file, column.
- **How to present them** — label, description, source credit, units, category, colour scaling.

```jsonc
"INCIDENCE_CRUDE": {                              // ← the layer id
    "SOURCE_DATA": {                              // ── where the values are ──
        "DATASET": { "NAME": "DHIS2_INCIDENCE", "VERSION": "latest" },   // ← a SNT_config.json key
        "FILENAME": "{COUNTRY_CODE}_incidence.csv",
        "COLUMN": "INCIDENCE_CRUDE"
    },
    "LABEL":       { "EN": "Crude incidence (DHIS2)", "FR": "Incidence brute (DHIS2)" },
    "DESCRIPTION": { "EN": "Number of malaria cases relative to the population.", "FR": "…" },
    "SOURCE":      { "EN": "DHIS2", "FR": "DHIS2" },
    "UNITS":       { "EN": "Per 1000 people", "FR": "Pour 1000 personnes" },
    "CATEGORY":    { "EN": "Epidemiological indicators", "FR": "Indicateurs épidémiologiques" },
    "TYPE":  "Threshold",                         // ── how to present them ──
    "SCALE": [50, 150, 250, 350, 450, 1000],
    "UNIT_SYMBOL": null
}
```

## 2. The decisions worth understanding

Everything else in the format follows from these six. The reasoning behind them, and what the old
format did instead, is in [ADR 0007](../../docs/decisions/0007-snt-metadata-json-format.md).

### 2.1 `SOURCE_DATA` — the file became a pointer, not just a caption

Each entry is **self-resolving**: given this file and the country's `SNT_config.json`, a consumer can
locate the values without any hardcoded mapping. That is what lets the IASO-based Explorer add a
layer through configuration rather than through a code change.

The consequence to keep in mind: this file now makes a **promise about data that lives elsewhere**.
Rename a column or a file in a pipeline and this file becomes wrong, with nothing to detect the
break until a layer fails to load. Treat a change to a published output filename or column name as
a change that must be mirrored here — the same discipline the repo already applies to
`docs/DATA_ARCHITECTURE.md`.

### 2.2 The layer id is not the column name

The top-level key is a **layer id**, and it need not equal the column:

```jsonc
"REPORTING_RATE_DATAELEMENT": { … "FILENAME": "{COUNTRY_CODE}_reporting_rate_dataelement.csv",
                                  "COLUMN": "REPORTING_RATE" },
"REPORTING_RATE_DATASET":     { … "FILENAME": "{COUNTRY_CODE}_reporting_rate_dataset.csv",
                                  "COLUMN": "REPORTING_RATE" },
```

Two genuinely different layers — reporting rate computed per data element, and per dataset — that
carry the same column name in different files. Hence the rule: **the layer id is unique and
arbitrary; `COLUMN` need not be unique.**

Practical consequences:

- Choose a layer id that stays true. It is what the Explorer and any saved user selection refer to,
  so renaming one is a breaking change, not a tidy-up.
- When two layers differ only by method or by source, put the discriminator in the id
  (`…_DATAELEMENT` / `…_DATASET`), not in the column.

### 2.3 Bilingual by construction

`LABEL`, `DESCRIPTION`, `SOURCE`, `UNITS` and `CATEGORY` are `{ "EN": …, "FR": … }` objects.

Both keys are **required even when empty**, so an untranslated field shows up as `""` — visible,
greppable, and countable — rather than as an absent key that reads like the format changed. What
the Explorer should *do* with an empty string is [BACKLOG META-1](../../docs/BACKLOG.md#meta-1).

### 2.4 Time is carried by the data, via the `YEAR` column

This file says nothing about *when* a layer's values apply, and that is deliberate.

Instead, the referenced table carries a **`YEAR` column**, the Explorer imports it alongside
`COLUMN`, and the user picks which year to display. So one metadata entry yields a layer the user
can step through in time, with no per-year entries and nothing to update here when a new year of
data arrives.

> **Rule: every pipeline output table must contain a `YEAR` column.** Data points are always
> attached to a year, so this costs nothing and is what makes a table usable as an Explorer layer.
> A table without `YEAR` cannot be displayed over time — and since neither this file nor the schema
> can see inside the referenced table, the omission surfaces only when the layer fails to load.

Two consequences when adding a layer:

- **Check the table actually has `YEAR`** before writing the entry. It is not verifiable from here.
- For a monthly output, `YEAR` is still required — how the Explorer handles a finer grain
  (aggregate, or offer a month selector) is [BACKLOG META-7](../../docs/BACKLOG.md#meta-7).
- **Some layers genuinely have no year.** A result computed *across* years — rainfall seasonality is
  the known case — has nothing to put in `YEAR`, and adding one would misrepresent the analysis. How
  the Explorer should treat such a layer is [BACKLOG META-9](../../docs/BACKLOG.md#meta-9). Do not
  resolve it by inventing a `YEAR`.

### 2.5 `DATASET.NAME` is a config key, not a dataset name

Despite the field name, `DATASET.NAME` does **not** hold the dataset's display name from the
OpenHEXA UI, and does not hold its slug either. It holds **a key of `SNT_DATASET_IDENTIFIERS` in
that country's `SNT_config.json`**, which the SNT Explorer looks up to get the dataset identifier:

```
SNT_metadata.json          SNT_config.json                      OpenHEXA
"NAME": "DHIS2_INCIDENCE"  →  SNT_DATASET_IDENTIFIERS:          →  the dataset
                                "DHIS2_INCIDENCE":
                                  "snt-dhis2-incidence"
```

The indirection is what makes the metadata file **country-portable**: the same entry works in every
workspace, because each country's config maps the key to its own dataset. Write a slug or a UI name
here and it resolves to nothing — the layer simply fails to load.

Two things follow:

- **A `NAME` is only valid if that key exists in the target country's config.** The set is not
  identical across countries — NER's config, for instance, carries three keys the others do not
  (`DHIS2_OUTLIERS_DETECTION`, `DHIS2_OUTLIERS_REMOVAL_IMPUTATION`, `SNT_SEASONALITY`). A metadata
  file meant to serve every country can only use keys present in **all** of their configs.
- **Cross-check it.** This is the one `SOURCE_DATA` field that *can* be validated offline, against
  the config — the validator in [§5](#5-validating) does exactly that. Do run it: a bad key is
  invisible until the Explorer tries to load the layer.

> **An easy mistake.** The config keys follow no single convention — some carry an `SNT_` prefix and
> some do not — so the key rarely looks like the dataset it points at, and writing the dataset's
> display name instead (`SNT_DHIS2_INCIDENCE` for `DHIS2_INCIDENCE`) is a natural slip. A cleanup of
> the key names is [BACKLOG CFG-1](../../docs/BACKLOG.md#cfg-1); renaming the field itself is
> [META-5](../../docs/BACKLOG.md#meta-5).

### 2.6 `TYPE`: `Threshold` versus `Ordinal`, and what `SCALE` means under each

`TYPE` says how the Explorer turns values into colours, and it **changes how `SCALE` is read**. That
coupling is the thing to get right; the two fields cannot be reasoned about separately.

| `TYPE` | For | `SCALE` holds | Outer values | *n* values give |
|---|---|---|---|---|
| `"Threshold"` | A continuous measure — incidence, population, a rate | the **cut points** between bins | **excluded** — the Explorer adds bottom and top | *n+1* bins |
| `"Ordinal"` | A discrete measure with few ordered values — a 0/1 flag, a duration in whole months | the **complete ordered list of values** the column can take | **included** — there is nothing to add | *n* classes |
| `"Linear"` ⚠ | A continuous measure shown as a colour **gradient** instead of bins | exactly **2 values**: the start and the end of the gradient | **included** — they are the two values | a gradient legend |

```jsonc
"INCIDENCE_CRUDE":                   { "TYPE": "Threshold", "SCALE": [100, 250, 450, 1000] },  // 5 bins
"SEASONALITY_RAINFALL":              { "TYPE": "Ordinal",   "SCALE": [0, 1] },                 // 2 classes
"SEASONAL_BLOCK_DURATION_RAINFALL":  { "TYPE": "Ordinal",   "SCALE": [3, 4, 5] },              // 3 classes
"PCT_HEALTH_ACCESS":                 { "TYPE": "Linear",    "SCALE": [0, 100] },               // gradient (not usable yet)
```

**⚠ `"Linear"` is documented but not usable yet.** The SNT Explorer supports it, but a small
Explorer-side bug currently blocks it; the IASO developers are aware and will fix it. Until then, do
not use it in `catalogs/SNT_metadata_master.json`. The schema `TYPE` enum deliberately still rejects
it, so the validator in [§5](#5-validating) fails on it. Once the bug is fixed, add `"Linear"` to the
enum in [`snt_metadata.schema.json`](snt_metadata.schema.json) ([BACKLOG META-4](../../docs/BACKLOG.md#meta-4)).

The SNT Explorer might later compute the end value as max() of the loaded data instead of reading it
from `SCALE` ([BACKLOG META-10](../../docs/BACKLOG.md#meta-10)). That matters most for **population**
layers. They always run from 0 to max(), but max() differs massively between countries, so no generic
default for the end value — or for `Threshold` cut points — can be meaningful.

Getting it backwards fails quietly in one of two ways: a `Threshold` scale written *with* its limits
produces two degenerate empty bins at the ends, and an `Ordinal` scale written *without* them loses a
class. Neither is a validation error — the schema can check the type of `SCALE`, not its meaning.

**`Ordinal` is still numeric.** `SCALE` remains an array of numbers, so a categorical layer is
expressed by the numbers its column actually holds (`[0, 1]`), not by the labels those numbers stand
for. The labels belong in `UNITS` — `"Not seasonal (0), Seasonal (1)"`. A layer whose values are
genuinely non-numeric strings still has no home in this format; that is
[BACKLOG META-4](../../docs/BACKLOG.md#meta-4).

## 3. Field reference

The authoritative description of every field is the `description` text inside
[`snt_metadata.schema.json`](snt_metadata.schema.json). Summary:

| Field | Type | Notes |
|---|---|---|
| *(top-level key)* | `UPPER_SNAKE_CASE` | The layer id. Unique, stable, arbitrary. |
| `SOURCE_DATA.DATASET.NAME` | `UPPER_SNAKE_CASE` | **A key of `SNT_DATASET_IDENTIFIERS` in the country's `SNT_config.json`** — see [§2.5](#25-datasetname-is-a-config-key-not-a-dataset-name). |
| `SOURCE_DATA.DATASET.VERSION` | string | `"latest"` everywhere so far. |
| `SOURCE_DATA.FILENAME` | string | **Must** start with the literal `{COUNTRY_CODE}`. |
| `SOURCE_DATA.COLUMN` | `UPPER_SNAKE_CASE` | Repo rule R10. Need not be unique. |
| `LABEL` | `{EN, FR}` | Legend-length name. House style ends with the source in parentheses. |
| `DESCRIPTION` | `{EN, FR}` | 1–2 sentences for a non-technical reader. |
| `SOURCE` | `{EN, FR}` | Display credit (`DHIS2`, `DHS`, …), not a machine reference. |
| `UNITS` | `{EN, FR}` | Spelled out (`Per 1000 people`). |
| `CATEGORY` | `{EN, FR}` | Grouping in the layer list. Must match sibling layers **exactly**. |
| `TYPE` | `"Threshold"` or `"Ordinal"` | Continuous or discrete. Determines how `SCALE` is read — [§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each). The schema rejects any other value deliberately. A third value, `"Linear"` (gradient), is documented in §2.6 but not accepted yet. |
| `SCALE` | array of numbers | Ascending. Bin **cut points** (outer limits excluded) under `Threshold`; the **full list of values** (outer limits included) under `Ordinal`; exactly **`[start, end]`** under `Linear` (not usable yet). |
| `UNIT_SYMBOL` | string or `null` | Short symbol (`"%"`), or `null`. |

No field is optional: the schema requires all nine, and rejects unknown ones. That is deliberate —
a typo like `SCALEE` should fail loudly instead of silently producing a layer with default styling.

### `SCALE`: two rules that are easy to get wrong

1. **Under `TYPE: "Threshold"`, do not include the outer limits.** Write `[50, 150, 250]`, not
   `[0, 50, 150, 250, 1000]`. The Explorer adds the bottom and top itself. *n* cut points give *n+1*
   bins. Including them yields two degenerate empty bins at the ends. **Under `TYPE: "Ordinal"` the
   rule inverts** — list every value, limits included ([§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each)).
2. **Ascending order is required, and the schema cannot check it.** JSON Schema has no way to
   express "sorted". The validator below does check it.

`SCALE` is a real JSON array (`[50, 150]`), not a string.

## 4. How to make common changes

### Add a layer

1. Confirm the values are actually published — the file is in the OpenHEXA dataset and the column
   is in it. Per repo rule **R4**, an output that was never passed to `add_files_to_dataset(...)`
   does not exist downstream, however real it looks in `data/`.
2. Confirm the table has a **`YEAR` column** ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)).
   Without it the layer cannot be displayed over time.
3. Get `DATASET.NAME` right: it is a **key of `SNT_DATASET_IDENTIFIERS`**, not the dataset's name in
   the OpenHEXA UI ([§2.5](#25-datasetname-is-a-config-key-not-a-dataset-name)). Copy it from the
   config, do not type it from memory.
4. Pick a layer id: `UPPER_SNAKE_CASE`, unique, and carrying the discriminator if a sibling layer
   differs only by method or source.
5. Copy an existing entry and fill in all nine fields. Do not omit a translation — put `""` in it.
6. Reuse an existing `CATEGORY` **string for string, in both languages**, or you will create a
   second group in the UI that looks identical to the first.
7. Choose `SCALE` from the real distribution of the values, not from round numbers alone, and
   leave the outer limits out.
8. Validate ([§5](#5-validating)).

### Adapt the file for another country

Usually you do not need to. `{COUNTRY_CODE}` in `FILENAME` is what makes one file serve every
country, so the same catalogue works as long as the country publishes the same outputs.

Genuine per-country edits are: **dropping** layers the country does not produce, and **adjusting
`SCALE`** where the country's value range makes the shared bins useless (a country with incidence
an order of magnitude lower than the bins is rendered as one flat colour). Prefer both of those
over editing `FILENAME`.

**Never replace `{COUNTRY_CODE}` with a literal code.** A file with `"COD_incidence.csv"` copied to
another country silently reads Congo's data under that country's name — wrong, and plausible enough
to go unnoticed. The schema rejects a hardcoded code for exactly this reason.

### Add a new `TYPE` or a new language

Both are format changes, not content changes: agree them with the IASO developers first, then
update the schema, this document and the example together, and date the change.

## 5. Validating

The schema is only useful if it is run. `jsonschema` is **not** currently in
[`dev/environment.yml`](../../dev/environment.yml) — install it, or use a system Python that has
it.

The validator is
[`.claude/skills/add-explorer-layer/validate_metadata.py`](../../.claude/skills/add-explorer-layer/validate_metadata.py)
(also used by the `add-explorer-layer` agent skill). It runs the schema, plus the two checks JSON
Schema cannot express on its own: `SCALE` ordering, and whether every `DATASET.NAME` is a key of
`SNT_DATASET_IDENTIFIERS` in the config skeleton
([`contracts/snt_config_json/SNT_config.skeleton.json`](../snt_config_json/SNT_config.skeleton.json) —
the trusted key set; the archived per-country copies were never verified).

```bash
python3 .claude/skills/add-explorer-layer/validate_metadata.py            # the master list
python3 .claude/skills/add-explorer-layer/validate_metadata.py <file>     # any other file
```

What this still **cannot** catch, and a human must:

- That `FILENAME` and `COLUMN` actually exist in the resolved dataset — the most likely real
  breakage, and only detectable against a live workspace.
- That the referenced table has a **`YEAR`** column ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)) — same reason.
- That `CATEGORY` strings are consistent across layers.
- That the bins in `SCALE` suit the country's actual value range.

## 6. The reference files

Every `SOURCE_DATA` pointer in both files below was traced to the code that writes the column (the
`add_files_to_dataset(...)` calls in `pipeline.py`, and the notebook cells that build the final
table). Nothing was checked against a live workspace.

### `SNT_metadata_NER.json`

Niger's catalogue: nine layers, converted from Niger's old-format file.

| Layers | `DATASET.NAME` | `FILENAME` | Written by |
|---|---|---|---|
| `POPULATION`, `POPULATION_FE`, `POPULATION_U5` | `DHIS2_DATASET_FORMATTED` | `{COUNTRY_CODE}_population.csv` | `snt_dhis2_formatting` |
| `REPORTING_RATE` | `DHIS2_REPORTING_RATE` | `{COUNTRY_CODE}_reporting_rate_dataelement.csv` | `snt_dhis2_reporting_rate_dataelement` |
| `INCIDENCE_CRUDE`, `INCIDENCE_ADJ_TESTING`, `INCIDENCE_ADJ_REPORTING` | `DHIS2_INCIDENCE` | `{COUNTRY_CODE}_incidence.csv` | `snt_dhis2_incidence` |
| `SEASONALITY_RAINFALL`, `SEASONAL_BLOCK_DURATION_RAINFALL` | `SNT_SEASONALITY_RAINFALL` | `{COUNTRY_CODE}_rainfall_seasonality.csv` | `snt_seasonality_rainfall` |

What to know before using it:

- **Population columns:** `POPULATION_FE` ("femmes enceintes") reads `POP_PREGNANT_WOMEN`, and
  `POPULATION_U5` reads `POP_UNDER_5`.
- **`REPORTING_RATE` is on 0–1**, so its `SCALE` is fractional (`[0.5, 0.8, 0.9, 0.95]`) with
  `UNIT_SYMBOL: null` ([BACKLOG META-2](../../docs/BACKLOG.md#meta-2)).
- **`SOURCE` credits `SNIS`**, Niger's national HIS, where the generic example credits `DHIS2`.
- **The three MAP layers** (`PF_INCIDENCE_RATE`, `PF_PR_RATE`, `PF_MORTALITY_RATE`) are not in the
  file: the MAP output is long-format, one file per year, which the format cannot express
  ([BACKLOG META-14](../../docs/BACKLOG.md#meta-14)).
- **The two seasonality layers reference a table with no `YEAR` column.** It is keyed on `ADM2_ID`
  alone, and the layers may fail to load ([BACKLOG META-9](../../docs/BACKLOG.md#meta-9)). In Niger
  the country variant `snt_seasonality_rainfall_NER.ipynb` is what runs; both columns are present in
  it as well as in the generic notebook.
- **Still to confirm:** the reporting-rate method, the English translations and the population layer
  ids ([BACKLOG META-17](../../docs/BACKLOG.md#meta-17)).

### The master list

[`catalogs/SNT_metadata_master.json`](../../catalogs/SNT_metadata_master.json) declares **every
column the current pipelines can publish** that fits this format. `snt_metadata_json_trim` fetches it
from this repository and keeps, per workspace, the layers whose data exists. It is edited by hand
([ADR 0008](../../docs/decisions/0008-master-list-maintained-by-hand.md)); how to maintain it:
[`catalogs/README.md`](../../catalogs/README.md). Validate it with [§5](#5-validating) (the
validator checks the master list by default). `snt_assemble_results` is out of scope (retired,
[ADR 0006](../../docs/decisions/0006-retire-snt-assemble-results.md)).

#### Draft content

- **Text.** `LABEL`, `DESCRIPTION`, `SOURCE` and `UNITS` hold a draft in EN and FR for all 58
  layers, written from the code that computes each column, not yet reviewed
  ([BACKLOG META-13](../../docs/BACKLOG.md#meta-13)). `CATEGORY` is `""` in both languages, to be
  decided. Conventions used in the draft:
  - `SOURCE` is `DHIS2`, `ERA5` or `WorldPop`, and `""` where the source depends on the run or mixes
    several: every `*_TRANSFORMED` layer (DHIS2 or user-provided input), `POPULATION_INCIDENCE`
    (operator-chosen population source), and `POP_COVERED` / `PCT_HEALTH_ACCESS` (WorldPop population
    plus DHIS2 or user-provided facilities). `LABEL` ends with `(SOURCE)` only when `SOURCE` is set.
  - `UNITS` says `Proportion` where the column is a 0–1 fraction (reporting rates, `*_PROPORTION`), in
    line with [BACKLOG META-2](../../docs/BACKLOG.md#meta-2).
- **`SCALE`.** No data was inspected. Where a layer also exists in
  [`SNT_metadata.example.json`](SNT_metadata.example.json) or [`SNT_metadata_NER.json`](SNT_metadata_NER.json)
  its scale is reused; otherwise it is a generic default for the unit traced in the code. Entries
  stop matching this table as real scales are written:

  | Unit (from the code) | `TYPE` | `SCALE` | `UNIT_SYMBOL` |
  |---|---|---|---|
  | Total population (count) | Threshold | example `POPULATION` bins | `null` |
  | `POP_UNDER_5` / `POP_PREGNANT_WOMEN` | Threshold | NER bins | `null` |
  | Other `POP_*` sub-populations | Threshold | `[5000, 10000, 20000, 40000]` | `null` |
  | Reporting rate, 0–1 | Threshold | NER bins `[0.5, 0.8, 0.9, 0.95]` | `null` |
  | Incidence, per 1000 | Threshold | example bins | `null` |
  | Case / death counts per ADM2-year | Threshold | `[10, 100, 1000, 10000, 100000]` | `null` |
  | Seasonality flag | Ordinal | `[0, 1]` | `null` |
  | Seasonal block duration, months | Ordinal | `[3, 4, 5]` (the block-size parameter choices) | `null` |
  | Seasonal block start month | Ordinal | `[1 … 12]` | `null` |
  | Share captured by the seasonal block, 0–1 | Threshold | `[0.5, 0.6, 0.7, 0.8]` | `null` |
  | Precipitation, mm per month | Threshold | `[25, 50, 100, 200, 300]` | `"mm"` |
  | Percentage, 0–100 | Threshold | `[25, 50, 75]` | `"%"` |

#### Layer ids and where they point

The layer id is the column name, plus a qualifier where two layers would otherwise collide
([§2.2](#22-the-layer-id-is-not-the-column-name)). Ids already in the example are kept.

| Layers | Dataset key | File | Pipeline |
|---|---|---|---|
| `POPULATION`, `POP_*` (8) | `DHIS2_DATASET_FORMATTED` | `{COUNTRY_CODE}_population.csv` | `snt_dhis2_formatting` |
| `POPULATION_TRANSFORMED`, `POP_*_TRANSFORMED` (8) | `DHIS2_POPULATION_TRANSFORMATION` | `{COUNTRY_CODE}_population.csv` | `snt_dhis2_population_transformation` |
| `REPORTING_RATE_DATAELEMENT`, `REPORTING_RATE_DATASET` | `DHIS2_REPORTING_RATE` | `…_reporting_rate_dataelement_yearly.csv`, `…_reporting_rate_dataset.csv` | `snt_dhis2_reporting_rate_*` |
| `POPULATION_INCIDENCE`, `INCIDENCE_*` (4) | `DHIS2_INCIDENCE` | `{COUNTRY_CODE}_incidence.csv` | `snt_dhis2_incidence` |
| `<COUNT>_QOC_IMPUTED`, `<COUNT>_QOC_REMOVED` (10 + 10) | `DHIS2_QUALITY_OF_CARE` | `…_quality_of_care_district_year_{imputed,removed}.csv` | `snt_dhis2_quality_of_care` |
| `SEASONALITY_*`, `SEASONAL_BLOCK_*`, `*_PROPORTION` (4 + 4) | `SNT_SEASONALITY_{CASES,RAINFALL}` | `…_{cases,rainfall}_seasonality.csv` | `snt_seasonality_*` |
| `PRECIPITATION_{MEAN,MIN,MAX}` | `ERA5_DATASET_CLIMATE` | `{COUNTRY_CODE}_total_precipitation_monthly.parquet` | `snt_era5_climate_data` |
| `POPULATION_WORLDPOP` | `WORLDPOP_DATASET_EXTRACT` | `{COUNTRY_CODE}_worldpop_population.csv` | `snt_worldpop_extract` |
| `POP_TOTAL`, `POP_COVERED`, `PCT_HEALTH_ACCESS` | `SNT_HEALTHCARE_ACCESS` | `{COUNTRY_CODE}_population_covered_health.csv` | `snt_healthcare_access` |

The population ids differ from Niger's file (`POP_UNDER_5` here, `POPULATION_U5` there); see
[BACKLOG META-17](../../docs/BACKLOG.md#meta-17).

#### Comparison with older catalogues

Some layers also exist in an older catalogue. To compare by hand if needed:

| Master layer | `deprecated/configuration/SNT_metadata.json` (old format, FR only) | [`SNT_metadata_NER.json`](SNT_metadata_NER.json) (EN + FR) |
|---|---|---|
| `POPULATION` | `POPULATION` | `POPULATION` |
| `POP_UNDER_5` | — | `POPULATION_U5` |
| `POP_PREGNANT_WOMEN` | — | `POPULATION_FE` |
| `REPORTING_RATE_DATAELEMENT` | — | `REPORTING_RATE` (same method: facility activity) |
| `REPORTING_RATE_DATASET` | `REPORTING_RATE` (same wording: reports received / expected) | — |
| `INCIDENCE_CRUDE` | `INCIDENCE_CRUDE` | `INCIDENCE_CRUDE` |
| `INCIDENCE_ADJ_TESTING` | `INCIDENCE_ADJ_TESTING` | `INCIDENCE_ADJ_TESTING` |
| `INCIDENCE_ADJ_REPORTING` | `INCIDENCE_ADJ_REPORTING` | `INCIDENCE_ADJ_REPORTING` |
| `INCIDENCE_ADJ_CARESEEKING` | `INCIDENCE_ADJ_CARESEEKING` | — |
| `SEASONALITY_CASES` | `SEASONALITY_CASES` | — |
| `SEASONALITY_RAINFALL` | `SEASONALITY_RAINFALL` | `SEASONALITY_RAINFALL` |
| `SEASONAL_BLOCK_DURATION_RAINFALL` | `SEASONAL_BLOCK_DURATION_RAINFALL` | `SEASONAL_BLOCK_DURATION_RAINFALL` |

**8 layers have an exact id match in `deprecated/configuration/SNT_metadata.json`:** `POPULATION`,
the four `INCIDENCE_*`, `SEASONALITY_CASES`, `SEASONALITY_RAINFALL` and
`SEASONAL_BLOCK_DURATION_RAINFALL`. The old file's other 21 entries (`REPORTING_RATE`, DHS, MAP) have
no master layer with the same id ([What was left out](#what-was-left-out-and-why)). Where the draft
text differs from the older text:

- **Incidence `UNITS`** is `Cases per 1000 people per year` (older: `Per 1000 people`).
- **Incidence adjustments are described as stacking** — reporting builds on testing, care-seeking
  builds on both — as the code computes them.
- **`INCIDENCE_ADJ_CARESEEKING`**: the code adjusts for **public-sector** care-seeking only, so the
  draft says so.
- **Labels say "district"**, not "zone spécifique", and the source is `DHIS2`, not NER's `SNIS`.

#### Layers that will not load everywhere

"Every column that *can* be published" includes columns that only some countries or parameter
choices produce. Expect those layers to fail to load where the column is absent:

- **`POP_*` in the formatted population** exist only where `POPULATION_INDICATOR_DEFINITIONS` gives
  that key DHIS2 ids. In the reference configs that is CMR (`UNDER_5`, `PREGNANT_WOMEN`, `0_1_Y`,
  `1_2_Y`, `50_PLUS`) and NER (`UNDER_5`, `PREGNANT_WOMEN`). `POP_5_10_Y` and `POP_5_36_M` have no ids
  in any reference config.
- **`POP_*_TRANSFORMED`** exist only when the operator sets the matching disaggregation parameter,
  or supplies a disaggregation file.
- **The quality-of-care counts** are written only if the column is in the routine data, so they vary
  with each country's `DHIS2_INDICATOR_DEFINITIONS` (`ALLADM`, `ALLDTH`, `ALLOUT` in particular).
- **Only one variant of an alternative is present at a time.** Quality of care writes the `imputed`
  or the `removed` file, per run. Both reporting-rate files can coexist, but see the dataset-version
  caveat below.
- **`INCIDENCE_ADJ_CARESEEKING`** is always written, but is all `NA` when no care-seeking data was
  available.

#### Tables that break other assumptions of the format

Included, but likely to exercise open questions:

- **No `YEAR` column:** both seasonality tables and the healthcare-access table
  ([BACKLOG META-9](../../docs/BACKLOG.md#meta-9)). Healthcare access is computed for a single
  WorldPop year, which is not stored in the table.
- **Monthly grain** (`YEAR` + `MONTH`): the `REPORTING_RATE_DATASET` table and ERA5
  ([BACKLOG META-7](../../docs/BACKLOG.md#meta-7)). `REPORTING_RATE_DATAELEMENT` reads the yearly
  table (`YEAR`, no `MONTH`).
- **`YEAR` stored as text:** the WorldPop table.
- **Parquet only:** ERA5 publishes no `.csv` twin, so its `FILENAME` is the `.parquet`.
- **Dataset versions (UNVERIFIED):** if `add_files_to_dataset` creates a version holding only the
  files of that run, `"latest"` on `DHIS2_REPORTING_RATE` holds only the variant run last, and the
  same goes for the two quality-of-care files.

#### What was left out, and why

| Left out | Pipeline | Why |
|---|---|---|
| All MAP layers (`{CC}_map_data_{year}.*`) | `snt_map_extracts` | Long format, one file per year ([BACKLOG META-14](../../docs/BACKLOG.md#meta-14)). |
| All DHS layers (12 files `{CC}_DHS_ADM1_*`: ITN access/use, care-seeking, U5 mortality, U5 RDT prevalence, DTP1–3 and dropout, each with CI bounds) | `snt_dhs_indicators` | **ADM1 grain**, not ADM2, and no `YEAR` column (the survey year is computed but not written) ([BACKLOG META-15](../../docs/BACKLOG.md#meta-15)). |
| The five quality-of-care rates: `testing_rate`, `treatment_rate`, `case_fatality_rate`, `prop_adm_malaria`, `prop_malaria_deaths` | `snt_dhis2_quality_of_care` | Written in **lowercase**, which `COLUMN`'s pattern rejects and which breaks repo rule R10 ([BACKLOG META-16](../../docs/BACKLOG.md#meta-16)). |
| `non_malaria_all_cause_outpatients`, `presumed_cases` | `snt_dhis2_quality_of_care` | Lowercase, as above, and copies of `ALLOUT` / `PRES`, which are included. |
| Routine indicators (`{CC}_routine.*`) | `snt_dhis2_formatting` | **Facility × month** grain, not ADM2. The Explorer would need aggregation the format cannot express. |
| `{CC}_routine_outliers_imputed.parquet`, `…_removed.parquet` | the five `snt_dhis2_outliers_imputation_*` | Facility × month grain, as above. |
| `{CC}_routine_outliers_detected.parquet` | the five `snt_dhis2_outliers_imputation_*` | Long format (`INDICATOR` / `VALUE`), facility grain. |
| `{CC}_reporting.*` | `snt_dhis2_formatting` | Long format (`PRODUCT_METRIC` / `VALUE`). |
| `{CC}_pyramid.*` | `snt_dhis2_formatting` | Facility list (coordinates, opening/closing dates); no values to map. |
| All raw extracts (`{CC}_dhis2_raw_*.parquet`) | `snt_dhis2_extract` | Long format, raw DHIS2 ids, no `YEAR`. |
| Extra columns from an uploaded disaggregation file | `snt_dhis2_population_transformation` | Arbitrary operator-chosen names, not uppercased. Only the template's `POP_*` columns are included. |
| Seasonality admin columns `ADM1_LEVEL_NAME`, `ADM2_LEVEL_NAME` | `snt_seasonality_*`, `snt_healthcare_access` | Admin columns. |
| `SNT_RESULTS` | `snt_assemble_results` | Retired; no longer written. |

`snt_workspace_check` publishes nothing to a dataset.

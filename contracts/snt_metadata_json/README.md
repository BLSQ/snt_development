# `SNT_metadata.json` — schema and conventions

> **Status: work in progress.** The shape described here is the one dated **2026-09-08**, currently
> being validated with the IASO developers. Open questions are the `META-*` items in
> [`docs/BACKLOG.md`](../../docs/BACKLOG.md#snt-explorer-metadata); do not treat a silence there as a
> decision. Why the format looks like this: [ADR 0007](../../docs/decisions/0007-snt-metadata-json-format.md).

| File | Role |
|---|---|
| [`snt_metadata.schema.json`](snt_metadata.schema.json) | **Source of truth for the shape.** JSON Schema (draft 2020-12). Every field carries a `description`. |
| [`SNT_metadata.example.json`](SNT_metadata.example.json) | A minimal valid instance: five layers taken from the master list, covering `Threshold`, `Ordinal`, and two layers that share a column ([§2.2](#22-the-layer-id-is-not-the-column-name)). `CATEGORY` is filled in for illustration; the master list's categories are still undecided ([BACKLOG META-3](../../docs/BACKLOG.md#meta-3)). |

The **content** — every layer the pipelines can publish — is the master list,
[`catalogs/SNT_metadata_master.json`](../../catalogs/SNT_metadata_master.json). What it contains,
leaves out and why is in [`catalogs/README.md`](../../catalogs/README.md). Older catalogues (the
old-format file and Niger's 2026-09 conversion) are archived in
[`deprecated/configuration/`](../../deprecated/configuration/README.md).

**Strict JSON only — no comments.** The SNT Explorer does not accept JSONC, so a `//` comment
anywhere in the file breaks ingestion. Notes belong in the schema's `description` fields or in this
document, never in the data file.

**This is the target format, not what every workspace holds today.** A workspace may still hold an
old-format copy in its `configuration/` folder; see [BACKLOG DOC-3](../../docs/BACKLOG.md#doc-3) for
what still reads it.

---

## 1. What this file is for

`SNT_metadata.json` is the catalogue of **data layers** the SNT Explorer offers a user for one
country. The Explorer is a read-only exploration tool, so this file affects presentation and nothing
else. It never changes the underlying data, and no analytics pipeline reads it.

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
    "DESCRIPTION": { "EN": "Number of confirmed malaria cases over the year, …", "FR": "…" },
    "SOURCE":      { "EN": "DHIS2", "FR": "DHIS2" },
    "UNITS":       { "EN": "Cases per 1000 people per year", "FR": "Cas pour 1000 personnes par an" },
    "CATEGORY":    { "EN": "Epidemiological indicators", "FR": "Indicateurs épidémiologiques" },
    "TYPE":  "Threshold",                         // ── how to present them ──
    "SCALE": [50, 150, 250, 350, 450, 1000],
    "UNIT_SYMBOL": null
}
```

## 2. The decisions worth understanding

What each decision means when you write an entry. Why it was made, and what the old format did
instead: [ADR 0007](../../docs/decisions/0007-snt-metadata-json-format.md).

### 2.1 `SOURCE_DATA` — the file became a pointer, not just a caption

Each entry is **self-resolving**: given this file and the country's `SNT_config.json`, a consumer can
locate the values without any hardcoded mapping.

The consequence: this file makes a **promise about data that lives elsewhere**. Rename a column or a
file in a pipeline and this file becomes wrong, with nothing to detect the break until a layer fails
to load. A change to a published output filename or column name must be mirrored in the master list,
in the same PR (CLAUDE.md handover checklist).

### 2.2 The layer id is not the column name

The top-level key is a **layer id**, and it need not equal the column:

```jsonc
"REPORTING_RATE_DATAELEMENT": { … "FILENAME": "{COUNTRY_CODE}_reporting_rate_dataelement_yearly.csv",
                                  "COLUMN": "REPORTING_RATE" },
"REPORTING_RATE_DATASET":     { … "FILENAME": "{COUNTRY_CODE}_reporting_rate_dataset.csv",
                                  "COLUMN": "REPORTING_RATE" },
```

Two genuinely different layers carry the same column name in different files. Hence: **the layer id
is unique and arbitrary; `COLUMN` need not be unique.**

- Choose a layer id that stays true. The Explorer and any saved user selection refer to it, so
  renaming one is a breaking change, not a tidy-up.
- When two layers differ only by method or by source, put the discriminator in the id
  (`…_DATAELEMENT` / `…_DATASET`), not in the column.

### 2.3 Bilingual by construction

`LABEL`, `DESCRIPTION`, `SOURCE`, `UNITS` and `CATEGORY` are `{ "EN": …, "FR": … }` objects. Both
keys are **required even when empty**, so an untranslated field shows up as `""` — visible and
greppable — rather than as an absent key. What the Explorer should *do* with `""` is
[BACKLOG META-1](../../docs/BACKLOG.md#meta-1).

### 2.4 Time is carried by the data, via the `YEAR` column

This file says nothing about *when* a layer's values apply. The referenced table carries a **`YEAR`
column** (repo rule [R21](../../docs/CONVENTIONS.md#r21)), the Explorer imports it alongside
`COLUMN`, and the user picks the year to display. One entry yields a layer the user can step through
in time, with nothing to update here when a new year of data arrives.

Neither this file nor the schema can see inside the referenced table, so a missing `YEAR` surfaces
only when the layer fails to load. When adding a layer:

- **Check the table actually has `YEAR`.** It is not verifiable from here.
- A monthly output still needs `YEAR`; how the Explorer handles a finer grain is
  [BACKLOG META-7](../../docs/BACKLOG.md#meta-7).
- **Some layers genuinely have no year** — a result computed *across* years, such as rainfall
  seasonality. Do not invent a `YEAR` for it; see [BACKLOG META-9](../../docs/BACKLOG.md#meta-9).

### 2.5 `DATASET.NAME` is a config key, not a dataset name

`DATASET.NAME` holds neither the dataset's display name in the OpenHEXA UI nor its slug. It holds
**a key of `SNT_DATASET_IDENTIFIERS` in that country's `SNT_config.json`**, which the SNT Explorer
looks up to get the dataset identifier:

```
SNT_metadata.json          SNT_config.json                      OpenHEXA
"NAME": "DHIS2_INCIDENCE"  →  SNT_DATASET_IDENTIFIERS:          →  the dataset
                                "DHIS2_INCIDENCE":
                                  "snt-dhis2-incidence"
```

The indirection makes an entry **country-portable**: each country's config maps the key to its own
dataset. Write a slug or a UI name here and it resolves to nothing — the layer fails to load.

- **A `NAME` is only valid if that key exists in the target country's config.** The key set is not
  identical across countries (NER's config carries `DHIS2_OUTLIERS_DETECTION`,
  `DHIS2_OUTLIERS_REMOVAL_IMPUTATION` and `SNT_SEASONALITY`, which the others do not). A file meant
  for every country can only use keys present in all of them.
- **Cross-check it** with the validator in [§5](#5-validating) — the one `SOURCE_DATA` field that
  can be checked offline.

> **An easy mistake.** The config keys follow no single convention — some carry an `SNT_` prefix and
> some do not — so writing the dataset's display name (`SNT_DHIS2_INCIDENCE` for `DHIS2_INCIDENCE`)
> is a natural slip. Cleaning the key names is [BACKLOG CFG-1](../../docs/BACKLOG.md#cfg-1); renaming
> the field is [META-5](../../docs/BACKLOG.md#meta-5).

### 2.6 `TYPE`: `Threshold` versus `Ordinal`, and what `SCALE` means under each

`TYPE` says how the Explorer turns values into colours, and it **changes how `SCALE` is read**. The
two fields cannot be reasoned about separately.

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

**⚠ `"Linear"` is not usable yet.** The SNT Explorer supports it, but an Explorer-side bug blocks it.
Until it is fixed, the schema's `TYPE` enum deliberately rejects it, so the validator fails on it —
do not use it in the master list. When the bug is fixed, add it to the enum
([BACKLOG META-4](../../docs/BACKLOG.md#meta-4)). The Explorer might later take the end value from
max() of the loaded data — useful for population layers, whose range differs too much between
countries for a generic default ([BACKLOG META-10](../../docs/BACKLOG.md#meta-10)).

Getting `Threshold` / `Ordinal` backwards fails quietly: a `Threshold` scale written *with* its limits
produces two degenerate empty bins at the ends, and an `Ordinal` scale written *without* them loses a
class. Neither is a validation error.

**`Ordinal` is still numeric.** A categorical layer is expressed by the numbers its column holds
(`[0, 1]`), with the labels in `UNITS` — `"Not seasonal (0), Seasonal (1)"`. A layer whose values are
non-numeric strings has no home in this format yet ([BACKLOG META-4](../../docs/BACKLOG.md#meta-4)).

## 3. Field reference

The authoritative description of every field is the `description` text inside
[`snt_metadata.schema.json`](snt_metadata.schema.json). Summary:

| Field | Type | Notes |
|---|---|---|
| *(top-level key)* | `UPPER_SNAKE_CASE` | The layer id. Unique, stable, arbitrary. |
| `SOURCE_DATA.DATASET.NAME` | `UPPER_SNAKE_CASE` | **A key of `SNT_DATASET_IDENTIFIERS` in the country's `SNT_config.json`** — [§2.5](#25-datasetname-is-a-config-key-not-a-dataset-name). |
| `SOURCE_DATA.DATASET.VERSION` | string | `"latest"` everywhere so far ([META-6](../../docs/BACKLOG.md#meta-6)). |
| `SOURCE_DATA.FILENAME` | string | **Must** start with the literal `{COUNTRY_CODE}`. |
| `SOURCE_DATA.COLUMN` | `UPPER_SNAKE_CASE` | Repo rule R10. Need not be unique. |
| `LABEL` | `{EN, FR}` | Legend-length name. House style ends with the source in parentheses. |
| `DESCRIPTION` | `{EN, FR}` | 1–2 sentences for a non-technical reader. |
| `SOURCE` | `{EN, FR}` | Display credit (`DHIS2`, `ERA5`, …), not a machine reference. |
| `UNITS` | `{EN, FR}` | Spelled out (`Cases per 1000 people per year`). |
| `CATEGORY` | `{EN, FR}` | Grouping in the layer list. Must match sibling layers **exactly**. |
| `TYPE` | `"Threshold"` or `"Ordinal"` | Determines how `SCALE` is read — [§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each). |
| `SCALE` | array of numbers | Ascending. Meaning depends on `TYPE` — [§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each). |
| `UNIT_SYMBOL` | string or `null` | Short symbol (`"%"`), or `null`. Must agree with `SCALE`: `"%"` means 0–100, a 0–1 fraction takes `null` ([META-2](../../docs/BACKLOG.md#meta-2)). |

No field is optional: the schema requires all nine, and rejects unknown ones, so a typo like `SCALEE`
fails loudly instead of silently producing a layer with default styling.

### `SCALE`: two rules that are easy to get wrong

1. **Under `TYPE: "Threshold"`, do not include the outer limits.** Write `[50, 150, 250]`, not
   `[0, 50, 150, 250, 1000]`. **Under `TYPE: "Ordinal"` the rule inverts** — list every value,
   limits included.
2. **Ascending order is required, and the schema cannot check it.** The validator does.

`SCALE` is a real JSON array (`[50, 150]`), not a string.

## 4. How to make common changes

### Add a layer

1. Confirm the values are actually published — the file is in the OpenHEXA dataset and the column
   is in it. Per repo rule **R4**, an output never passed to `add_files_to_dataset(...)` does not
   exist downstream, however real it looks in `data/`.
2. Confirm the table has a **`YEAR` column** ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)).
3. Get `DATASET.NAME` right: a **key of `SNT_DATASET_IDENTIFIERS`**, copied from the config, not
   typed from memory ([§2.5](#25-datasetname-is-a-config-key-not-a-dataset-name)).
4. Pick a layer id: `UPPER_SNAKE_CASE`, unique, carrying the discriminator if a sibling layer
   differs only by method or source.
5. Copy an existing entry and fill in all nine fields. Do not omit a translation — put `""` in it.
6. Reuse an existing `CATEGORY` **string for string, in both languages**, or you create a second
   group in the UI that looks identical to the first.
7. Choose `SCALE` from the real distribution of the values, and leave the outer limits out under
   `Threshold`.
8. Validate ([§5](#5-validating)).

### Adapt the file for another country

Usually you do not need to: `{COUNTRY_CODE}` in `FILENAME` is what makes one file serve every
country. Genuine per-country edits are **dropping** layers the country does not produce (what
`snt_metadata_json_trim` does) and **adjusting `SCALE`** where the shared bins are useless for the
country's value range. Where such tuning should live is [BACKLOG META-12](../../docs/BACKLOG.md#meta-12).

**Never replace `{COUNTRY_CODE}` with a literal code.** A file with `"COD_incidence.csv"` copied to
another country silently reads Congo's data under that country's name. The schema rejects a
hardcoded code for exactly this reason.

### Add a new `TYPE` or a new language

Both are format changes, not content changes: agree them with the IASO developers first, then
update the schema, this document and the example together, and date the change.

## 5. Validating

`jsonschema` is **not** in [`dev/environment.yml`](../../dev/environment.yml) — install it, or use a
system Python that has it.

The validator is
[`.claude/skills/add-explorer-layer/validate_metadata.py`](../../.claude/skills/add-explorer-layer/validate_metadata.py)
(also used by the `add-explorer-layer` skill). It runs the schema, plus the two checks JSON Schema
cannot express: `SCALE` ordering, and whether every `DATASET.NAME` is a key of
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
- That the referenced table has a **`YEAR`** column ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)).
- That `CATEGORY` strings are consistent across layers.
- That the bins in `SCALE` suit the country's actual value range.

# `SNT_metadata.json` — schema and conventions

> **Status: work in progress.** The shape described here is the one dated **2026-09-08**, currently
> being validated with the IASO developers. It is written down so the work is not lost and can be
> built on progressively — not because it is settled. Section [Open questions](#open-questions)
> lists what is still undecided; do not treat a silence there as a decision.

| File | Role |
|---|---|
| [`snt_metadata.schema.json`](snt_metadata.schema.json) | **Source of truth for the shape.** JSON Schema (draft 2020-12) describing the 2026-09-08 format and nothing else. Every field carries a `description`. |
| [`SNT_metadata.example.json`](SNT_metadata.example.json) | The canonical valid instance — the 2026-09-08 file, verbatim except that its `//` comments were removed. Everything those comments said is preserved in the schema's `description` fields and in this document. |
| [`SNT_metadata_NER.json`](SNT_metadata_NER.json) | **Niger's catalogue**, converted from the old-format `SNT_metadata_20260226.json` on 2026-09-08. Nine layers, schema-valid. Read [§7](#7-the-ner-conversion-2026-09-08) before using it: three MAP layers could not be carried over, and two seasonality layers reference a table with no `YEAR` column. |

> **Format change, 2026-09-08:** `TYPE` now also accepts `"Ordinal"`, and `SCALE` is read differently
> under it. See [§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each).

**Strict JSON only — no comments.** The SNT Explorer does not accept JSONC, so a `//` comment
anywhere in the file breaks ingestion. Notes belong in the schema's `description` fields or in this
document, never in the data file. (They would not survive the future editing webapp either: an app
that reads, edits and rewrites JSON drops comments silently.)

**These two files describe the target format. They do not describe what is deployed today.**
`configuration/SNT_metadata.json` in this repo is still the older shape, and
`snt_assemble_results` still reads that older shape. Nothing has been migrated. See
[Relationship to what exists today](#6-relationship-to-what-exists-today).

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

Everything else in the format follows from these six.

### 2.1 `SOURCE_DATA` — the file became a pointer, not just a caption

The older format described a layer (`LABEL`, `UNITS`, `SCALE`…) but never said where its values
came from. The consumer had to already know. That works when the consumer is one pipeline written
by the same team; it does not work when the consumer is a general-purpose tool ingesting layers it
was not built for.

Adding `SOURCE_DATA` makes each entry **self-resolving**: given this file and the country's
`SNT_config.json`, a consumer can locate the values without any hardcoded mapping. That is what
lets the IASO-based Explorer add a layer through configuration rather than through a code change,
and it is the single biggest reason the format changed.

The consequence to keep in mind: this file now makes a **promise about data that lives elsewhere**.
Rename a column or a file in a pipeline and this file becomes wrong, with nothing to detect the
break until a layer fails to load. Treat a change to a published output filename or column name as
a change that must be mirrored here — the same discipline the repo already applies to
`docs/DATA_ARCHITECTURE.md`.

### 2.2 The layer id is not the column name

The top-level key is a **layer id**. In the old format it was effectively the column name, and the
two happened to coincide. They no longer must:

```jsonc
"REPORTING_RATE_DATAELEMENT": { … "FILENAME": "{COUNTRY_CODE}_reporting_rate_dataelement.csv",
                                  "COLUMN": "REPORTING_RATE" },
"REPORTING_RATE_DATASET":     { … "FILENAME": "{COUNTRY_CODE}_reporting_rate_dataset.csv",
                                  "COLUMN": "REPORTING_RATE" },
```

Two genuinely different layers — reporting rate computed per data element, and per dataset — that
carry the same column name in different files. Under the old rule one would have overwritten the
other. Hence the rule now: **the layer id is unique and arbitrary; `COLUMN` need not be unique.**

Practical consequences:

- Choose a layer id that stays true. It is what the Explorer and any saved user selection refer to,
  so renaming one is a breaking change, not a tidy-up.
- When two layers differ only by method or by source, put the discriminator in the id
  (`…_DATAELEMENT` / `…_DATASET`), not in the column.

### 2.3 Bilingual by construction

`LABEL`, `DESCRIPTION`, `SOURCE`, `UNITS` and `CATEGORY` are `{ "EN": …, "FR": … }` objects. The
old format held a single string, in practice French, which left no room for an English Explorer
and no way to tell a deliberate choice from a missing translation.

Both keys are **required even when empty**, so an untranslated field shows up as `""` — visible,
greppable, and countable — rather than as an absent key that reads like the format changed. What
the Explorer should *do* with an empty string is [Open question #1](#open-questions).

### 2.4 Time is carried by the data, via the `YEAR` column

This file says nothing about *when* a layer's values apply, and that is deliberate. The old format
had a `PERIOD` field that `snt_assemble_results` wrote back at runtime; it is gone.

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
  (aggregate, or offer a month selector) is [Open question #7](#open-questions).
- **Some layers genuinely have no year.** A result computed *across* years — rainfall seasonality is
  the known case — has nothing to put in `YEAR`, and adding one would misrepresent the analysis. How
  the Explorer should treat such a layer is [Open question #9](#open-questions). Do not resolve it by
  inventing a `YEAR`.

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

> **Correction applied.** The 2026-09-08 working file used `SNT_DHIS2_INCIDENCE` and
> `SNT_DHIS2_REPORTING_RATE` — the datasets' *display names*, which are not config keys. The
> committed example uses the correct keys `DHIS2_INCIDENCE` and `DHIS2_REPORTING_RATE`
> (6 of its 7 entries were affected); all now resolve in all five country configs.
> `configuration/SNT_metadata_20260908.json` still has the old values and has not been touched.
>
> The confusion is understandable, because the config keys follow no single convention — some carry
> an `SNT_` prefix and some do not, so the key rarely looks like the dataset it points at. That
> cleanup is logged for Giulia in
> [`CLAUDE.md`](../../CLAUDE.md#suggestions-logged-for-later-evaluation-giulia).

### 2.6 `TYPE`: `Threshold` versus `Ordinal`, and what `SCALE` means under each

`TYPE` says how the Explorer turns values into colours, and it **changes how `SCALE` is read**. That
coupling is the thing to get right; the two fields cannot be reasoned about separately.

| `TYPE` | For | `SCALE` holds | Outer values | *n* values give |
|---|---|---|---|---|
| `"Threshold"` | A continuous measure — incidence, population, a rate | the **cut points** between bins | **excluded** — the Explorer adds bottom and top | *n+1* bins |
| `"Ordinal"` | A discrete measure with few ordered values — a 0/1 flag, a duration in whole months | the **complete ordered list of values** the column can take | **included** — there is nothing to add | *n* classes |

```jsonc
"INCIDENCE_CRUDE":                   { "TYPE": "Threshold", "SCALE": [100, 250, 450, 1000] },  // 5 bins
"SEASONALITY_RAINFALL":              { "TYPE": "Ordinal",   "SCALE": [0, 1] },                 // 2 classes
"SEASONAL_BLOCK_DURATION_RAINFALL":  { "TYPE": "Ordinal",   "SCALE": [3, 4, 5] },              // 3 classes
```

Getting it backwards fails quietly in one of two ways: a `Threshold` scale written *with* its limits
produces two degenerate empty bins at the ends, and an `Ordinal` scale written *without* them loses a
class. Neither is a validation error — the schema can check the type of `SCALE`, not its meaning.

**`Ordinal` is still numeric.** `SCALE` remains an array of numbers, so a categorical layer is
expressed by the numbers its column actually holds (`[0, 1]`), not by the labels those numbers stand
for. The labels belong in `UNITS` — `"Not seasonal (0), Seasonal (1)"` — which is where the old
format's `"[not-seasonal, seasonal]"` content now lives. A layer whose values are genuinely
non-numeric strings still has no home in this format; that remains
[Open question #4](#open-questions).

> **Added 2026-09-08**, when Niger's catalogue was converted: both of its seasonality layers are
> ordinal, and forcing them into `Threshold` would have meant expressing a 0/1 flag as a single cut
> point. `"Threshold"` was the only accepted value before this change.

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
| `TYPE` | `"Threshold"` or `"Ordinal"` | Continuous or discrete. Determines how `SCALE` is read — [§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each). The schema rejects any other value deliberately. |
| `SCALE` | array of numbers | Ascending. Bin **cut points** (outer limits excluded) under `Threshold`; the **full list of values** (outer limits included) under `Ordinal`. |
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

`SCALE` also changed type: it was a *stringified* array (`"[50, 150]"`) and is now a real JSON
array (`[50, 150]`). Every consumer had to parse that string by hand; now it is just data.

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
it:

This runs the schema, plus the two checks JSON Schema cannot express on its own: `SCALE` ordering,
and whether every `DATASET.NAME` resolves against the country configs.

```bash
python3 -c "
import glob, json, sys
from jsonschema import Draft202012Validator

METADATA = 'docs/schemas/SNT_metadata.example.json'   # <- the file being checked
CONFIGS  = 'configuration/SNT_config_*.json'          # <- configs to resolve NAME against

schema = json.load(open('docs/schemas/snt_metadata.schema.json', encoding='utf-8'))
inst   = json.load(open(METADATA, encoding='utf-8'))
bad = []

for e in sorted(Draft202012Validator(schema).iter_errors(inst), key=lambda e: list(e.path)):
    bad.append(('/'.join(map(str, e.path)) or '(root)') + ': ' + e.message)

# SCALE must ascend - not expressible in JSON Schema
for lid, layer in inst.items():
    s = layer.get('SCALE', [])
    if s != sorted(s):
        bad.append(lid + ': SCALE is not in ascending order')

# every DATASET.NAME must be a key of SNT_DATASET_IDENTIFIERS, in every country
names = {lid: l['SOURCE_DATA']['DATASET']['NAME'] for lid, l in inst.items()}
for f in sorted(glob.glob(CONFIGS)):
    keys = json.load(open(f, encoding='utf-8')).get('SNT_DATASET_IDENTIFIERS', {})
    for lid, n in names.items():
        if n not in keys:
            bad.append(lid + ': DATASET.NAME ' + repr(n) + ' is not a key of SNT_DATASET_IDENTIFIERS in ' + f)

print(*bad, sep='\n')
print('FAIL' if bad else 'OK')
sys.exit(1 if bad else 0)
"
```

Point `METADATA` at whichever file you are checking. What this still **cannot** catch, and a human
must:

- That `FILENAME` and `COLUMN` actually exist in the resolved dataset — the most likely real
  breakage, and only detectable against a live workspace.
- That the referenced table has a **`YEAR`** column ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)) — same reason.
- That `CATEGORY` strings are consistent across layers.
- That the bins in `SCALE` suit the country's actual value range.

## 6. Relationship to what exists today

| | Status |
|---|---|
| `configuration/SNT_metadata.json` | **Old format**, still what is deployed. Single-language strings, stringified `SCALE`, no `SOURCE_DATA`, plus `ORDER` and `PERIOD`. Not migrated. |
| `configuration/SNT_metadata_20260908.json` | The working file this schema was derived from. It carries `//` comments; the committed example does not, because the Explorer cannot read JSONC (see the note at the top of this document). |
| `snt_assemble_results` | Reads the **old** format, and writes `PERIOD` back into it at runtime via `update_metadata(...)`. Being deprecated — the Explorer will read the OpenHEXA datasets directly. |

**Two fields were dropped:**

- **`ORDER`** — an integer that ordered layers in the list. No longer needed.
- **`PERIOD`** — the data's time coverage, which `snt_assemble_results` *wrote back into the file at
  runtime*. It is not needed either, and its removal ends the oddity of a pipeline mutating a
  configuration file. Time is now carried by the **data**, not by this file: see
  [§2.4](#24-time-is-carried-by-the-data-via-the-year-column).

**Where the file will live.** A new OpenHEXA dataset, `SNT_CONFIGURATION`, will hold
`SNT_config.json` and `SNT_metadata.json`, and the `configuration/` folder in this repo is expected
to be deprecated. A user-facing OpenHEXA webapp will let non-technical users edit them, most likely
writing through a Python pipeline that validates the structure first — which is where this schema
becomes an executable gate rather than documentation. **None of that exists yet**; nothing in this
folder assumes it, and the paths above are the ones that are real today.

## 7. The NER conversion (2026-09-08)

[`SNT_metadata_NER.json`](SNT_metadata_NER.json) was produced from Niger's old-format
`SNT_metadata_20260226.json` (12 layers). **Nine layers carried over; three did not.** Every
`SOURCE_DATA` pointer was traced to the pipeline that writes the column — none was guessed — and the
result passes the validator in [§5](#5-validating), including `DATASET.NAME` resolution against all
five country configs.

### What the nine layers point at

| Layers | `DATASET.NAME` | `FILENAME` | Written by |
|---|---|---|---|
| `POPULATION`, `POPULATION_FE`, `POPULATION_U5` | `DHIS2_DATASET_FORMATTED` | `{COUNTRY_CODE}_population.csv` | `snt_dhis2_formatting` |
| `REPORTING_RATE` | `DHIS2_REPORTING_RATE` | `{COUNTRY_CODE}_reporting_rate_dataelement.csv` | `snt_dhis2_reporting_rate_dataelement` |
| `INCIDENCE_CRUDE`, `INCIDENCE_ADJ_TESTING`, `INCIDENCE_ADJ_REPORTING` | `DHIS2_INCIDENCE` | `{COUNTRY_CODE}_incidence.csv` | `snt_dhis2_incidence` |
| `SEASONALITY_RAINFALL`, `SEASONAL_BLOCK_DURATION_RAINFALL` | `SNT_SEASONALITY_RAINFALL` | `{COUNTRY_CODE}_rainfall_seasonality.csv` | `snt_seasonality_rainfall` |

The old file named no columns, so three had to be resolved from
`POPULATION_INDICATOR_DEFINITIONS` in [`SNT_config_NER.json`](../../configuration/SNT_config_NER.json):
`POPULATION` → `POPULATION`, `POPULATION_FE` ("femmes enceintes") → `POP_PREGNANT_WOMEN`,
`POPULATION_U5` → `POP_UNDER_5`. All three have real DHIS2 ids in Niger's config.

`REPORTING_RATE` was ambiguous between the two reporting-rate pipelines, which both publish to
`DHIS2_REPORTING_RATE`. The **data element** variant was chosen because the old description read
*"Taux de rapportage sur base de l'activité des FOSAs"*, and that pipeline is the one that derives
reporting from facility activity indicators; the dataset variant uses DHIS2's own pre-computed
actual/expected report counts instead. **Worth confirming with whoever wrote the original file** —
the two produce different numbers from the same dataset, and nothing in the file records which was
meant.

### The three MAP layers were omitted

`PF_INCIDENCE_RATE`, `PF_PR_RATE` and `PF_MORTALITY_RATE` **cannot be expressed in this format
today.** `snt_map_extracts` publishes a **long-format** table, **one file per year**:

```
{CC}_map_data_2022.csv, {CC}_map_data_2023.csv, …
  columns: METRIC_CATEGORY, METRIC_NAME, STATISTIC, VALUE, YEAR, VERSION
```

So locating `PF_INCIDENCE_RATE` needs two things `SOURCE_DATA` has no field for: a **row filter**
(`METRIC_NAME == "Pf_Incidence_Rate"`, `STATISTIC == …`) and a **year wildcard** in `FILENAME`,
whose schema pattern requires a fixed name. `COLUMN` would be `VALUE` for all three layers, which
carries no information.

The one wide source is `{CC}_results_dataset.csv` on `SNT_RESULTS`, where `snt_assemble_results`
pivots `METRIC_NAME` into real `PF_*` columns — but it **filters each metric to a single year and
drops `YEAR`**, so the layer could not be stepped through time ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)),
and that pipeline is being deprecated precisely so the Explorer can read source datasets directly.
Pointing at it would encode the dependency the format exists to remove.

**To restore these three layers, one of the following has to happen** (a decision for the SNT and
IASO developers together, not a metadata edit):

1. `snt_map_extracts` also publishes a **wide** table — one column per metric, one row per
   `ADM2_ID` × `YEAR`, in a single year-independent file. Then each layer is an ordinary entry and
   nothing else changes. *Cheapest, and consistent with every other pipeline in the repo.*
2. `SOURCE_DATA` gains an optional **`FILTER`** field (`{"METRIC_NAME": "Pf_Incidence_Rate"}`), plus
   a way to express the per-year filenames. A format change, and it makes every consumer implement
   row filtering.

### Two layers reference a table with no `YEAR` column

`SEASONALITY_RAINFALL` and `SEASONAL_BLOCK_DURATION_RAINFALL` are included and validate, but
`{CC}_rainfall_seasonality.csv` is `admin_seasonality_wide_dt` — seasonality **per admin unit,
irrespective of year**, aggregated across the whole rainfall series. It is keyed on `ADM2_ID` alone.

That breaks [§2.4](#24-time-is-carried-by-the-data-via-the-year-column) and repo rule **R20**, and it
is not a metadata problem: the two entries are as correct as they can be, and **the layers may
simply fail to load** if the Explorer requires `YEAR`. The rule and the analysis genuinely disagree
here — seasonality is *defined* over multiple years, so there is no year to attach and adding one
would misrepresent the result. **This is now [Open question #9](#open-questions)**, which sets out the
options; it needs a decision from the IASO developers rather than a fix in this repo. **Unverified
either way** — neither the table nor the Explorer's behaviour was checked against a live workspace.

Also relevant for Niger specifically: `snt_seasonality_rainfall` has a country-specific variant,
`pipelines/snt_seasonality_rainfall/code/snt_seasonality_rainfall_NER.ipynb`, and it is in an
**active** location, so it is what actually runs in the NER workspace. Both columns were confirmed
present in that variant as well as in the generic notebook.

### Other changes made while converting

- **Encoding.** The source file was mojibake throughout (`spÃ©cifique`, `santÃ©`) — a UTF-8 file read
  as Latin-1. All accents restored.
- **`SNIS` as the source credit.** Niger's file credits `SNIS` (the national HIS) where the generic
  example credits `DHIS2`. Kept as `SNIS`, since `SOURCE` is a display credit for the user
  ([§3](#3-field-reference)) and Niger's operators use that name.
- **Two inconsistencies in the source were normalised**, both flagged rather than silent:
  `POPULATION_U5` was labelled `(DHIS2)` while its own `SOURCE` said `SNIS`, and
  `INCIDENCE_ADJ_TESTING` was labelled `(DHI2)` — a typo. Both now read `(SNIS)`.
- **English translations were written for all nine layers.** The source file was French-only, so
  every `EN` value is new. They are a faithful translation of the French, but they have **not been
  reviewed by a domain speaker** — worth a pass, particularly `REPORTING_RATE`, where the French
  said only "based on FOSA activity" and the English states the numerator and denominator explicitly.
- **`CATEGORY` strings were reused verbatim** from [`SNT_metadata.example.json`](SNT_metadata.example.json)
  where one existed (`Variable of population and health`, `Epidemiological indicators`), so the two
  files do not split one Explorer group into two. `Data quality` and `Environmental indicators` are
  new English strings — the example leaves the former blank.
- **`REPORTING_RATE`'s `SCALE` was rescaled** from `[50, 80, 90, 95, 100]` to `[0.5, 0.8, 0.9, 0.95]`
  to match what the pipeline writes, and the trailing limit dropped. See
  [Open question #2](#open-questions).

### Where the file lives, and why not `configuration/`

`SNT_metadata_NER.json` is in `docs/schemas/` rather than beside the country configs because
**`.gitignore` would silently swallow it in `configuration/`**: line 51 ignores `*.json` and the only
negations are `configuration/SNT_config_*.json`, `.claude/settings.json` and `docs/schemas/*.json`.
`configuration/SNT_metadata.json` is tracked only because it predates that rule — a *new*
`configuration/SNT_metadata_NER.json` would not be, and `git status` would never mention it.

Moving it there means adding a negation (`!configuration/SNT_metadata_*.json`) in the same commit.
Since [§6](#6-relationship-to-what-exists-today) expects `configuration/` to be deprecated in favour
of an `SNT_CONFIGURATION` dataset anyway, `docs/schemas/` is the safer home for a file that is a
reference copy either way.

## Open questions

Carried forward from the working file and from writing this up. Unresolved — flag rather than
guess.

1. **Empty translations.** `""` appears in several fields. Should a consumer fall back to the other
   language, show the layer id, or hide the layer? *Until decided: fill both languages where you
   can, and treat `""` as a visible to-do.*
2. **Percentage or proportion.** `REPORTING_RATE_*` in [`SNT_metadata.example.json`](SNT_metadata.example.json)
   has `SCALE: [0.25, 0.5, 0.75]` (fractions) with `UNIT_SYMBOL: "%"` and `UNITS: "Proportion"`.
   Those disagree: as written, a legend would read "0.25 %". Either `SCALE` becomes `[25, 50, 75]`
   with `"%"`, or `UNIT_SYMBOL` becomes `null`. The schema documents the constraint but cannot
   enforce it — the underlying data's own scale decides.

   **Settled for the reporting rate itself, 2026-09-08:** `snt_dhis2_reporting_rate_dataelement`
   computes `REPORTING_RATE` as a bare ratio (`HF_ACTIVE_THIS_PERIOD_BY_ADM2 / HF_ACTIVE_THIS_YEAR_BY_ADM2`
   — no `× 100`), so the published column is on **0–1**. Any entry pointing at it therefore takes
   fractional cut points with `UNIT_SYMBOL: null`, which is what
   [`SNT_metadata_NER.json`](SNT_metadata_NER.json) does. The example file has **not** been changed;
   fixing it is a one-line edit whenever someone touches it. The general question — whether the
   Explorer ever scales a fraction to a percentage on ingest — is still open.
3. **Is `CATEGORY` a controlled vocabulary?** Free text today, so a typo silently creates a group.
   A fixed list (enum in the schema, dropdown in the webapp) would prevent that, at the cost of a
   schema edit whenever a category is added.
4. **Which rendering modes does IASO actually support?** `"Ordinal"` was added on 2026-09-08
   alongside `"Threshold"`, and `SCALE` does now carry a different meaning per `TYPE`
   ([§2.6](#26-type-threshold-versus-ordinal-and-what-scale-means-under-each)) — so the "may then
   need" in the original wording has come true. Two parts are still open: **(a)** whether the
   Explorer implements `"Ordinal"` at all, or silently falls back to threshold binning, which is the
   one thing that would make the two seasonality layers in
   [`SNT_metadata_NER.json`](SNT_metadata_NER.json) render wrongly rather than not at all; and
   **(b)** whether a genuinely **categorical** layer — unordered, or with non-numeric values — needs
   a third `TYPE`, since `SCALE` is an array of numbers and cannot hold labels. Confirm (a) with the
   IASO developers before treating `"Ordinal"` as delivered.
5. **Should `DATASET.NAME` be renamed?** It holds a config *key*, not a name ([§2.5](#25-datasetname-is-a-config-key-not-a-dataset-name)),
   and the mismatch already caused six wrong pointers in the working file. `DATASET_KEY`, or
   `IDENTIFIER_KEY`, would be self-explanatory. Renaming means a coordinated change with the IASO
   developers, so it is worth deciding **before** the format is adopted rather than after.
6. **`VERSION: "latest"` versus pinning.** `"latest"` means a published Explorer view changes under
   the user when a pipeline re-runs — usually wanted, occasionally not (a view cited in a report).
   Does IASO accept a specific version id here?
7. **Sub-annual data.** `YEAR` is the time column the Explorer uses ([§2.4](#24-time-is-carried-by-the-data-via-the-year-column)).
   For an output that is monthly or by epi-week, does the Explorer aggregate to the year, or can it
   offer a finer selector — and if so, which column does it expect (`MONTH`, `PERIOD`)?
8. **No `$schema` key in the data file.** The example does not point at its own schema, so an editor
   will not validate it automatically and nothing records which format version a given file follows.
   Worth adding once the shape settles — as a `$schema` property, or a `FORMAT_VERSION` field, or by
   letting the `SNT_CONFIGURATION` dataset version carry it.
9. **Year-independent layers — how does a layer with no single `YEAR` work?** Confirmed real, not an
   oversight in a pipeline. [§2.4](#24-time-is-carried-by-the-data-via-the-year-column) and repo rule
   **R20** assume every layer's table carries `YEAR`, but **rainfall seasonality is by construction a
   multi-year classification**: `snt_seasonality_rainfall` decides whether an `ADM2` is seasonal by
   how *often* — across the whole rainfall series — a month starts a concentrated block, so the
   published table is keyed on `ADM2_ID` alone. There is no year to attach, and inventing one would
   misrepresent the analysis. The two affected layers are in
   [`SNT_metadata_NER.json`](SNT_metadata_NER.json) and may simply fail to load.

   Nothing here is decidable from this repo alone; it needs the IASO developers. Roughly in order of
   preference:

   - **The Explorer tolerates a table with no `YEAR`** and renders the layer as time-invariant — no
     year selector, or the layer greyed out in the selector rather than dropped. Cheapest, and
     honest about what the data is. Needs confirming that a missing `YEAR` is a supported case and
     not a load failure.
   - **The metadata declares it**, e.g. an optional `TIME: "invariant"` (or `YEAR: null`) on the
     layer, so a consumer knows not to look for the column instead of discovering it missing. A
     format change, but it turns a silent failure into a declaration.
   - **The table carries its coverage instead of a year** — `YEAR_START` / `YEAR_END`, or a
     `PERIOD_LABEL` like `"2019–2024"` — which is genuinely useful provenance for a reader regardless
     of what the Explorer does with it. Costs a pipeline change and a new convention.
   - **Repeat the value for every year in the range.** Makes the layer step through time with no
     format change at all, but it fabricates a per-year result the analysis never produced and would
     read as a finding. Recorded to be rejected, not adopted.

   Until this is settled, **R20 has a genuine exception** and `CLAUDE.md` says the rule was never
   audited against existing outputs. Whichever way it goes, R20's wording should gain the exception
   so the next person does not "fix" the seasonality table by adding a meaningless `YEAR`.

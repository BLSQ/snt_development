# `SNT_metadata.json` — schema and conventions

> **Status: work in progress.** The shape described here is the one dated **2026-09-08**, currently
> being validated with the IASO developers. It is written down so the work is not lost and can be
> built on progressively — not because it is settled. Section [Open questions](#open-questions)
> lists what is still undecided; do not treat a silence there as a decision.

| File | Role |
|---|---|
| [`snt_metadata.schema.json`](snt_metadata.schema.json) | **Source of truth for the shape.** JSON Schema (draft 2020-12) describing the 2026-09-08 format and nothing else. Every field carries a `description`. |
| [`SNT_metadata.example.json`](SNT_metadata.example.json) | The canonical valid instance — the 2026-09-08 file, verbatim except that its `//` comments were removed. Everything those comments said is preserved in the schema's `description` fields and in this document. |

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

Everything else in the format follows from these five.

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
| `TYPE` | `"Threshold"` | Only value used so far; the schema rejects others deliberately. |
| `SCALE` | array of numbers | Bin cut points, ascending, **outer limits excluded**. |
| `UNIT_SYMBOL` | string or `null` | Short symbol (`"%"`), or `null`. |

No field is optional: the schema requires all nine, and rejects unknown ones. That is deliberate —
a typo like `SCALEE` should fail loudly instead of silently producing a layer with default styling.

### `SCALE`: two rules that are easy to get wrong

1. **Do not include the outer limits.** Write `[50, 150, 250]`, not `[0, 50, 150, 250, 1000]`. The
   Explorer adds the bottom and top itself. *n* cut points give *n+1* bins. Including them yields
   two degenerate empty bins at the ends.
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

## Open questions

Carried forward from the working file and from writing this up. Unresolved — flag rather than
guess.

1. **Empty translations.** `""` appears in several fields. Should a consumer fall back to the other
   language, show the layer id, or hide the layer? *Until decided: fill both languages where you
   can, and treat `""` as a visible to-do.*
2. **Percentage or proportion.** `REPORTING_RATE_*` has `SCALE: [0.25, 0.5, 0.75]` (fractions) with
   `UNIT_SYMBOL: "%"` and `UNITS: "Proportion"`. Those disagree: as written, a legend would read
   "0.25 %". Either `SCALE` becomes `[25, 50, 75]` with `"%"`, or `UNIT_SYMBOL` becomes `null`. The
   schema documents the constraint but cannot enforce it — the underlying data's own scale decides.
3. **Is `CATEGORY` a controlled vocabulary?** Free text today, so a typo silently creates a group.
   A fixed list (enum in the schema, dropdown in the webapp) would prevent that, at the cost of a
   schema edit whenever a category is added.
4. **Is `TYPE` ever anything but `"Threshold"`?** The schema restricts it to that one value so an
   unreviewed rendering mode fails validation. If IASO supports continuous or categorical scales,
   this needs extending — and `SCALE` may then need a different meaning per `TYPE`.
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

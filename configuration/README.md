# `configuration/`

Reference files for the SNT configuration. **Nothing in this folder is read by a pipeline.** In a
workspace, pipelines read `<workspace>/configuration/SNT_config.json`: one file, one country, one
workspace. That file is placed there by hand (or, in future, by the config webapp, see below).

| File | What it is | Trust it? |
|---|---|---|
| `SNT_config.skeleton.json` | The **fixed part** of `SNT_config.json`, which is the same in every country and workspace. The country-specific values are left empty. | **Yes.** This is the reference for the current structure. |
| `SNT_config_<CC>.json` | Past per-country copies (BDI, BFA, CMR, COD, NER). | **No.** Not checked against the running workspaces, so they may be out of date. Use them as examples only. |
| `SNT_metadata.json` | Column metadata used by `snt_assemble_results` (which is being deprecated). | See the root `CLAUDE.md`, *Traps*. |

---

## `SNT_config.skeleton.json`

### Why it exists

The per-country copies were never verified against what actually runs in each workspace, so none of
them can be trusted as "the current shape of the config". The skeleton fills that gap. It gives one
place to see the **latest version of the part of `SNT_config.json` that never changes between
countries**:

- the full set of keys and how they are nested;
- the values that are the same everywhere: every `SNT_DATASET_IDENTIFIERS` entry, and
  `"type": "dataElement"` on every `POPULATION_INDICATOR_DEFINITIONS` entry.

Country-specific values are left empty. The skeleton records *where* they go, not *what* goes
there.

### What it deliberately does not do

- **It does not explain how to fill in the country-specific values.** That belongs in a future
  `snt_config.schema.json`, on the same model as the schema proposed for `SNT_metadata.json` under
  [`docs/schemas/`](https://github.com/BLSQ/snt_development/tree/main/docs/schemas).
  `[TODO: Giulia — not created yet]`
- **It is not a working config.** Copying it into a workspace without filling it in will make the
  pipelines fail or skip steps.

### How empty values are written

| Kind of value | Empty value | Example |
|---|---|---|
| Single value (string or number) | `null` | `COUNTRY_CODE`, `ANALYTICS_ORG_UNITS_LEVEL`, `REPORTING_DATASETS[].DATASET`, `REPORTING_INDICATORS.*` |
| List | `[]` | `REPORTING_RATE_PRODUCT_UID`, every `ids`, every `DHIS2_INDICATOR_DEFINITIONS.*` |

> **Differs from the webapp repo.** The config editor webapp (below) uses `""` for an empty string
> value and `null` for an empty number. Keep this difference in mind when comparing the two
> skeletons.

---

## Where configuration is heading: the config editor webapp

Filling in `SNT_config.json` is moving to a dedicated OpenHEXA webapp with its own repository:
**[BLSQ/openhexa-webapps-edit-snt-config-json](https://github.com/BLSQ/openhexa-webapps-edit-snt-config-json)**.
Its [`CLAUDE.md`](https://github.com/BLSQ/openhexa-webapps-edit-snt-config-json/blob/main/CLAUDE.md)
goes deeper than this folder: what each part of `SNT_config.json` is used for, how to fill it in,
and validation rules. **Go there for anything about the country-specific values.**

### Known drift between the two skeletons

The webapp repo keeps its own copy of the skeleton, and it **may have drifted** from
`SNT_config.skeleton.json` here (on top of the `""`/`null` difference above). This is known and
accepted for now. Because configuration is moving to the webapp, the two will be reconciled **in the
webapp repo**, not here. Until that happens:

- **This file** describes the structure that the pipelines in `snt_development` expect today.
- If the two disagree, check the pipeline code (`snt_config[...]` lookups) before deciding which one
  is right.

---

## Open question: the shape of `REPORTING_DATASETS[].METRICS`

Today every config has:

```json
"METRICS": {
  "ACTUAL_REPORTS": "float",
  "EXPECTED_REPORTS": "float"
}
```

The value is always `"float"`, which looks like a placeholder that just avoids an empty value.  
It is possible that the type `"float"` *is* used during
extraction. No such use is visible in this repo. If it exists, it is probably in `snt_lib`
([BLSQ/snt_utils](https://github.com/BLSQ/snt_utils), external and unpinned) or somewhere else
outside this repo. **An automated search of this repo cannot detect it**, so do not take "no usage
found" as proof that the value is unused.
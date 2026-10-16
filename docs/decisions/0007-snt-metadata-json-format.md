# 0007. `SNT_metadata.json`: a pointer per layer, keyed on config keys, time via `YEAR`

Date: 2026-09-08
Status: Proposed — being validated with the IASO developers

## Context

The old format described a layer (`LABEL`, `UNITS`, `SCALE`…) but never said where its values came
from. That worked while its only consumer was one pipeline written by the same team
(`snt_assemble_results`). It does not work for a general-purpose tool, the IASO-based SNT Explorer,
that ingests layers it was not built for. The old format also held single-language strings (in
practice French), keyed layers on the column name, stored `SCALE` as a stringified array, and carried
an `ORDER` field and a `PERIOD` field that a pipeline wrote back at runtime.

## Decision

The format specified in [`contracts/snt_metadata_json/`](../../contracts/snt_metadata_json/README.md):

1. **`SOURCE_DATA`** (dataset, file, column) on every entry, so each one is self-resolving: given the
   file and the country's `SNT_config.json`, a consumer locates the values with no hardcoded mapping.
   This lets the Explorer add a layer through configuration instead of code. It is the main reason the
   format changed.
2. **The layer id is not the column name.** Ids are unique and arbitrary; `COLUMN` need not be unique,
   so two methods can publish the same column name in different files.
3. **Bilingual fields** (`{EN, FR}`), both keys required even when empty, so a missing translation is
   visible.
4. **Time is carried by the data**: the referenced table has a `YEAR` column. `PERIOD` and `ORDER` are
   dropped, ending a pipeline's mutation of a configuration file.
5. **`DATASET.NAME` holds a key of `SNT_DATASET_IDENTIFIERS`**, not a dataset name, so one entry is
   portable across countries.
6. **`SCALE` is a real JSON array**, read according to `TYPE`. `"Ordinal"` was added beside
   `"Threshold"` on 2026-09-08, because forcing Niger's 0/1 seasonality flag into `Threshold` would
   have meant expressing it as a single cut point.

## Consequences

- The file now **promises facts about data that lives elsewhere**. Renaming a published file or
  column breaks a layer with nothing to detect it until it fails to load.
- A table without `YEAR` cannot be shown over time, and some results have no year at all
  ([BACKLOG META-9](../BACKLOG.md#meta-9)).
- `DATASET.NAME` is easy to get wrong, and was: six of seven entries in the 2026-09-08 draft used
  display names. See [META-5](../BACKLOG.md#meta-5) and [CFG-1](../BACKLOG.md#cfg-1).
- Open questions on the format are the `META-*` items in [`BACKLOG.md`](../BACKLOG.md#snt-explorer-metadata).

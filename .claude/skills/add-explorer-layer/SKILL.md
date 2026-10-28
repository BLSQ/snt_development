---
name: add-explorer-layer
description: Add, edit or check a layer in the SNT Explorer master list (catalogs/SNT_metadata_master.json), tracing that the column is really published, has YEAR, and resolves to a valid dataset key, then validating the file. Use when a published column is added or renamed, or when asked to add/fix/check an Explorer layer or validate an SNT_metadata.json file.
---

# Add, edit or check an SNT Explorer layer

The file format and its reasoning are in
[`contracts/snt_metadata_json/README.md`](../../../contracts/snt_metadata_json/README.md) (below:
"contract README"); the master list's own rules in [`catalogs/README.md`](../../../catalogs/README.md).
Read the sections each step names — not the whole file. When only **checking** a layer, skip step 4.

Run commands from the repo root. Cite notebooks as `<file>.ipynb` cell *N* (0-based code+markdown
index), not raw JSON line numbers.

## Steps

1. **The column is published (R4).** Trace three things in the producing pipeline:
   - the file named in `FILENAME` (`{COUNTRY_CODE}_<file>`) is a path in an
     `add_files_to_dataset(...)` call in its `pipeline.py` (a file only in `data/` does not exist
     downstream);
   - the save call that writes it — often a helper in `pipelines/<name>/utils/*.r`;
   - where `COLUMN` is created or joined in, and that no later `select` drops it.

   Check for `<generic>_<CC>.ipynb` siblings in the same `code/` folder: a country with one runs
   that instead, and the trace must hold there too. If the column's *meaning* depends on an
   operator parameter (e.g. `POPULATION` after `select_population_column()`), flag it — it belongs
   in `DESCRIPTION`/`SOURCE`.
2. **The table has `YEAR` (R21)** — contract README §2.4. If it is missing: for an output that is
   genuinely cross-year (seasonality, healthcare access) report it under
   [BACKLOG META-9](../../../docs/BACKLOG.md#meta-9) and continue; never invent a year. Otherwise
   stop and report it — the layer cannot be displayed over time.
3. **`DATASET.NAME` is a config key** (contract README §2.5), not the dataset's UI name: it must be
   a key of `SNT_DATASET_IDENTIFIERS` in
   [`contracts/snt_config_json/SNT_config.skeleton.json`](../../../contracts/snt_config_json/SNT_config.skeleton.json)
   *and* the key the pipeline's `add_files_to_dataset(dataset_id=…)` resolves. Copy it when adding;
   confirm it matches when checking.
4. **Edit** following contract README §4 *Add a layer* (id, all nine fields, `CATEGORY` reused string
   for string in both languages, `SCALE` without outer limits — §3, `{COUNTRY_CODE}` never a literal
   code). In `LABEL` and `DESCRIPTION`, name admin levels **ADM1 / ADM2**, never "region" /
   "district" (FR: "région" / "district"): the list serves every country, and what ADM2 is on the
   ground differs between them.
5. **Validate:**

   ```bash
   python3 .claude/skills/add-explorer-layer/validate_metadata.py            # master list
   python3 .claude/skills/add-explorer-layer/validate_metadata.py <file>     # any other copy
   ```

   Exit code 1 = fix and rerun. If `jsonschema` is missing it exits with an install hint (it is not
   in `dev/environment.yml`).
6. **Report** the trace from steps 1–3 with `file:line` / cell evidence, the validator output, and
   what the validator cannot catch (contract README §5): that `FILENAME`/`COLUMN` exist in a *live*
   dataset, `CATEGORY` consistency, whether `SCALE` suits the country's value range. Those need a
   workspace or a human.

## Scope

Editing the master list in the repo changes nothing in a workspace until `snt_metadata_json_trim`
is run there. Do not run it — hand over instead (`handover` skill).

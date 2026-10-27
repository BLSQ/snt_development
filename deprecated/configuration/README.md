# `deprecated/configuration/` — archived configuration files

Kept for reference only. **No code reads these copies** (pipelines read the files in
`<workspace>/configuration/`, never this repo), and they were **not verified against the running
workspaces**: treat them as examples, never as the current state of any country.

| File | What it is |
|---|---|
| `SNT_config_<CC>.json` | Past per-country copies of `SNT_config.json` (BDI, BFA, CMR, COD, NER). May be out of date. |
| `SNT_metadata.json` | Old-format column metadata, read by `snt_assemble_results`, which is retired (see [`../DEPRECATED_snt_assemble_results/`](../DEPRECATED_snt_assemble_results/)). |
| `SNT_metadata_NER.json` | Niger's catalogue converted to the new format in 2026-09 (nine layers, schema-valid at the time; the three MAP layers could not be carried over). Nothing in this repo deploys or reads it; superseded by [`catalogs/SNT_metadata_master.json`](../../catalogs/README.md). How it differs from the master list: [BACKLOG META-17](../../docs/BACKLOG.md#meta-17). |

For the current structure of `SNT_config.json`, see
[`contracts/snt_config_json/`](../../contracts/snt_config_json/README.md). For the current
`SNT_metadata.json` format, see [`contracts/snt_metadata_json/`](../../contracts/snt_metadata_json/README.md).

# `metadata/` — the SNT Explorer master list

| File | Role |
|---|---|
| [`SNT_metadata_master.json`](SNT_metadata_master.json) | **The master list**: every SNT Explorer data layer the current SNT pipelines can publish, for any country. Generic — it describes the pipeline collection, not a workspace. |

The **format** (fields, `TYPE` / `SCALE` rules, JSON Schema, validator) is specified in
[`docs/schemas/snt_metadata_json/`](../docs/schemas/snt_metadata_json/README.md). This folder holds
only the shipped file.

## How it reaches a workspace

It is **not** deployed with any pipeline: `openhexa pipelines push` packages only
`.py .ipynb .txt .md .r .sql` files. Instead, the
[`snt_metadata_json_trim`](../snt_metadata_json_trim/readme.md) pipeline downloads it from this
repository at run time. The ref it downloads from is set by the `repository_ref` parameter:

- `latest` (default) — the latest published GitHub release;
- a release tag;
- a branch such as `main`.

The pipeline then keeps the layers whose source column exists in that workspace, and publishes the
result as `SNT_metadata_trimmed.json` to the `SNT_METADATA` dataset.

The file is part of each release like any other file in the repository. Its SHA256, recorded in the
trim report as `MASTER.SHA256`, can be checked against `release_manifest.json`.

**So a change here reaches workspaces only once it is in a release** — or when an operator sets
`repository_ref` to a branch on purpose.

## Maintaining it

- **The JSON is the source of truth.** Edit it by hand, in a PR. There is no generator any more. The
  one-off bootstrap script is in git history at commit `062cf0c`; do not resurrect it over curated
  content.
- **Two kinds of field, two kinds of review:**
  - **Objective:** the layer id and `SOURCE_DATA` (dataset key, filename, column). Check them against
    the code that writes the column.
  - **Curated:** `LABEL`, `DESCRIPTION`, `SOURCE`, `UNITS`, `CATEGORY`, `TYPE`, `SCALE`,
    `UNIT_SYMBOL`. These need a person with domain knowledge. Text is a draft written from the code
    (2026-10-15), not yet reviewed; `CATEGORY` is empty. Which layers also exist in older catalogues:
    [`docs/schemas/snt_metadata_json/README.md` §8](../docs/schemas/snt_metadata_json/README.md#where-the-draft-text-came-from-2026-10-15).
- **When a pipeline adds or renames a published column**, add or update the master list entry in the
  same PR (CLAUDE.md handover checklist). A column with no entry is invisible in the Explorer. An
  entry whose column was renamed is dropped by the trim pipeline as `COLUMN_NOT_FOUND` in every
  workspace.
- **Validate before merging** with the snippet in
  [`docs/schemas/snt_metadata_json/README.md` §5](../docs/schemas/snt_metadata_json/README.md#5-validating),
  with `METADATA = 'metadata/SNT_metadata_master.json'`.
- **What the bootstrap included, what it left out and why:**
  [`docs/schemas/snt_metadata_json/README.md` §8](../docs/schemas/snt_metadata_json/README.md#8-the-master-list-2026-09-28).

## Known open points

- **Country tuning has no home yet.** `SCALE` here is a generic default. If a country hand-tunes
  scales or labels in its trimmed file, the next trim run overwrites them. A per-country override
  mechanism is still to be decided.
- **Checking the master list against the pipelines is manual.** Scripted checks are logged as a later
  development item in [`CLAUDE.md`](../CLAUDE.md#suggestions-logged-for-later-evaluation-giulia).
  They would cover the schema, config keys, filenames against `add_files_to_dataset(...)`, and
  columns observed in workspaces. Columns built in R cannot be extracted statically at reasonable
  cost.

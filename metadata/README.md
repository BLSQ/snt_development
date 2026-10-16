# `metadata/` — the SNT Explorer master list

| File | Role |
|---|---|
| [`SNT_metadata_master.json`](SNT_metadata_master.json) | **The master list**: every SNT Explorer data layer the current SNT pipelines can publish, for any country. Generic — it describes the pipeline collection, not a workspace. |

The **format** (fields, `TYPE` / `SCALE` rules, JSON Schema, validator) is specified in
[`contracts/snt_metadata_json/`](../contracts/snt_metadata_json/README.md). This folder holds
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

- **The JSON is the source of truth.** Edit it by hand, in a PR. There is no generator. The
  one-off bootstrap script is in git history at commit `062cf0c`; do not resurrect it over curated
  content ([ADR 0008](../docs/decisions/0008-master-list-maintained-by-hand.md)).
- **Two kinds of field, two kinds of review:**
  - **Objective:** the layer id and `SOURCE_DATA` (dataset key, filename, column). Check them against
    the code that writes the column.
  - **Curated:** `LABEL`, `DESCRIPTION`, `SOURCE`, `UNITS`, `CATEGORY`, `TYPE`, `SCALE`,
    `UNIT_SYMBOL`. These need a person with domain knowledge. Text is a draft written from the code,
    not yet reviewed ([BACKLOG META-13](../docs/BACKLOG.md#meta-13)); `CATEGORY` is empty. Which
    layers also exist in older catalogues:
    [`contracts/snt_metadata_json/README.md` §6](../contracts/snt_metadata_json/README.md#comparison-with-older-catalogues).
- **When a pipeline adds or renames a published column**, add or update the master list entry in the
  same PR (CLAUDE.md handover checklist). A column with no entry is invisible in the Explorer. An
  entry whose column was renamed is dropped by the trim pipeline as `COLUMN_NOT_FOUND` in every
  workspace.
- **Validate before merging** with the snippet in
  [`contracts/snt_metadata_json/README.md` §5](../contracts/snt_metadata_json/README.md#5-validating),
  with `METADATA = 'metadata/SNT_metadata_master.json'`.
- **What the master list includes, what it leaves out and why:**
  [`contracts/snt_metadata_json/README.md` §6](../contracts/snt_metadata_json/README.md#the-master-list).

## Known open points

- **Country tuning has no home yet**: a trim run overwrites hand-tuned scales or labels
  ([BACKLOG META-12](../docs/BACKLOG.md#meta-12)).
- **Checking the master list against the pipelines is manual**
  ([BACKLOG META-11](../docs/BACKLOG.md#meta-11)).

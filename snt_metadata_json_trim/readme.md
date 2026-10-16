# SNT Metadata JSON Trim Pipeline

The **SNT Metadata JSON Trim** pipeline takes the master list of SNT Explorer data layers
(**`SNT_metadata_master.json`**, fetched from the `snt_development` repository) and keeps only the
layers whose data actually exists in this workspace. It publishes the trimmed catalogue, **`SNT_metadata_trimmed.json`**, and a run report to
the **`SNT_METADATA`** dataset, creating that dataset on first run.

> **Status:** a tool for NER only (`ner-snt-process`) for now. It is **not** published through
> `snt-development` and has no `push_*.yaml` workflow. Adopting it as an SNT template pipeline is a
> separate decision.

## Parameters

* **`repository_ref`** (String, Required, Default: `latest`):
  * **Name:** Master list version (release tag or branch)
  * **Description:** Which version of
    [`metadata/SNT_metadata_master.json`](../metadata/SNT_metadata_master.json) to fetch from
    `github.com/BLSQ/snt_development`:
    * `latest` — the latest published GitHub release (drafts and pre-releases excluded).
    * a release tag (e.g. `v2026.10.0`) — pins that release.
    * a branch (e.g. `main`) — the current state of that branch. GitHub may serve a copy up to a few
      minutes old after a push.

    Ignored when `metadata_file` is given.
* **`metadata_file`** (File, Optional):
  * **Name:** Master list file (optional override)
  * **Description:** A local master list to use instead of the repository one, e.g. to test an edited
    copy. A JSON object keyed by layer id, in the format described in
    `contracts/snt_metadata_json/`. Leave empty for normal runs.

## Functionality Overview

1. **Configuration:** Load **`configuration/SNT_config.json`** for **`COUNTRY_CODE`** and
   **`SNT_DATASET_IDENTIFIERS`**. If it cannot be loaded, the run fails (nothing written).
2. **Get the master list:** If `metadata_file` is given, read it. Otherwise resolve `repository_ref`
   (`latest` → the latest release's tag) and download `metadata/SNT_metadata_master.json` at that
   ref, unauthenticated (the repository is public). Its SHA256 is recorded in the report. If the
   download fails, or the content is not a JSON object, the run fails with an error naming the cause. The
   possible causes are: no release published yet, unknown ref, or a ref that predates the master
   list. The run never falls back to another ref.
3. **Check each distinct source once:** Group layers by (`DATASET.NAME`, `DATASET.VERSION`,
   `FILENAME` with `{COUNTRY_CODE}` substituted). Checks for each group, in order:
   1. `DATASET.NAME` is a key of `SNT_DATASET_IDENTIFIERS`.
   2. That dataset exists in the workspace.
   3. The version exists. `"latest"` means the most recent version, whatever its name; any other
      value is matched against version names, then version ids.
   4. The file is in that version, and is a `.csv` or `.parquet`.
   5. Only the column names are read (CSV header, or parquet schema). **Values are not inspected.**
4. **Trim:** A layer is kept only if its `COLUMN` is one of the file's columns, **exact match,
   case-sensitive**. Kept layers are copied unchanged and in their original order.
5. **Write** both files to **`pipelines/snt_metadata_json_trim/output/`**.
6. **Publish:** Create a new version of **`SNT_METADATA`**, named `SNT_[COUNTRY_CODE]_YYYYMMDD_HHMM`
   (UTC), holding both files. The dataset is created if it does not exist yet.

**The run fails whenever it does not publish a usable trimmed list.** An error-level message names
the cause, and the run status is *failed*:

| Cause | Written to the output folder | Published |
|---|---|---|
| `SNT_config.json` cannot be loaded (step 1) | nothing | nothing |
| Master list cannot be fetched or parsed (step 2): no release yet, unknown ref, ref predates the master list, network or GitHub error, invalid JSON | nothing | nothing |
| The output files cannot be written (step 5) | — | nothing |
| **No layer is kept** | both files (read the report for every drop reason) | nothing — the previous `SNT_METADATA` version stays the latest |
| Publishing fails (step 6) | both files | nothing |

**A partial result still succeeds.** When at least one layer is kept, the run publishes and finishes
as a success, even if other layers were dropped. That includes layers dropped as `CHECK_FAILED`,
which is logged at error level.

## Inputs

* **The master list** — from the repository at `repository_ref`, or `metadata_file` (see
  Parameters). Needs outbound HTTPS to `api.github.com` and `raw.githubusercontent.com`.
* **`configuration/SNT_config.json`** for **`SNT_CONFIG.COUNTRY_CODE`** and
  **`SNT_DATASET_IDENTIFIERS`** (every key named by a layer's `DATASET.NAME`).
* **Every dataset file referenced by the catalogue.** Each one is optional: a missing one drops its
  layers, it does not stop the run.

## Outputs

**Workspace filesystem** — `pipelines/snt_metadata_json_trim/output/`

* **`SNT_metadata_trimmed.json`** — the kept layers, formatted like the master list
  (4-space indent, UTF-8, non-ASCII kept as-is).
* **`SNT_metadata_trimmed_report.json`** — the run report (below).

Both are overwritten on each run.

**Published to `SNT_METADATA`** (new version on every run)

* **`SNT_metadata_trimmed.json`**
* **`SNT_metadata_trimmed_report.json`**

> **Notes for the Data Analyst:**
>
> - **Drop reasons** (`REASON` in the report, and in the run log):
>   - `COLUMN_NOT_FOUND`: the file exists but does not contain `COLUMN`.
>   - `FILE_NOT_FOUND`: the file is not in the resolved dataset version.
>   - `DATASET_NOT_FOUND`: the configured dataset does not exist in this workspace.
>   - `DATASET_NO_VERSION`: the dataset exists but has never been published to.
>   - `VERSION_NOT_FOUND`: a pinned `VERSION` matches no version name or id.
>   - `CONFIG_KEY_MISSING`: `DATASET.NAME` is not a key of `SNT_DATASET_IDENTIFIERS`.
>   - `UNSUPPORTED_FILE_TYPE`: the file is neither `.csv` nor `.parquet`.
>   - `MALFORMED_LAYER`: `SOURCE_DATA` is missing a required field.
>   - `CHECK_FAILED`: an API error, or a file that could not be read. **This does not mean the data is
>     absent.** It is logged at error level; re-run once the cause is fixed.
> - **Report fields:** `RUN_AT`, `COUNTRY_CODE`, `MASTER` (`SOURCE` = `repository` or `file`,
>   `REF_REQUESTED`, `REF_RESOLVED`, `PATH`, `SHA256`), `N_LAYERS_INPUT` / `_KEPT` /
>   `_DROPPED`, `KEPT` (layer ids), `DROPPED` (`LAYER_ID`, `REASON`, `DETAIL`, `DATASET_NAME`,
>   `FILENAME`, `COLUMN`), and `SOURCES`: one entry per file checked, including the **resolved
>   dataset version**, which records which vintage each kept layer was validated against.
> - **`MASTER.SHA256` ties the run to a release.** It is the SHA256 of the master list's bytes, the
>   same hash `release_manifest.json` records for `metadata/SNT_metadata_master.json`, so a run can
>   be matched to the release it trimmed.
> - **A kept layer is only as fresh as its dataset version.** The check is made against whatever
>   version is latest *at run time*. Re-run this pipeline after upstream pipelines publish.
> - **"Last run wins" is visible here.** Alternative pipelines that publish to the same dataset
>   (the two reporting-rate variants, quality of care `imputed` / `removed`) leave only the most
>   recent variant's file in the latest version, so the other variant's layers are dropped as
>   `FILE_NOT_FOUND`. That is correct: the Explorer would not find them either.
> - **Cost:** each referenced file is downloaded once, in full, to read its header. That is fine for
>   the ADM2-level outputs in the catalogue today; it would be slow for facility-level files.

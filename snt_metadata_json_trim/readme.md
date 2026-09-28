# SNT Metadata JSON Trim Pipeline

The **SNT Metadata JSON Trim** pipeline takes the full SNT Explorer layer catalogue
(**`SNT_metadata_all_layers.json`**) and keeps only the layers whose data actually exists in this
workspace. It publishes the trimmed catalogue, **`SNT_metadata_trimmed.json`**, and a run report to
the **`SNT_METADATA`** dataset, creating that dataset on first run.

> **Status:** a tool for NER only (`ner-snt-process`) for now. It is **not** published through
> `snt-development` and has no `push_*.yaml` workflow. Adopting it as an SNT template pipeline is a
> separate decision.

## Parameters

* **`metadata_file`** (File, Required):
  * **Name:** SNT metadata JSON (all layers)
  * **Description:** The catalogue to trim: a JSON object keyed by layer id, in the format described
    in `docs/schemas/snt_metadata_json/`. Chosen from the workspace file browser; where this file
    should live permanently is still undecided.

## Functionality Overview

1. **Configuration:** Load **`configuration/SNT_config.json`** for **`COUNTRY_CODE`** and
   **`SNT_DATASET_IDENTIFIERS`**. If it cannot be loaded, log an error and stop (nothing written).
2. **Read the catalogue:** Parse `metadata_file`. If it is not a JSON object, log an error and stop.
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

**The run never fails.** Every problem is logged and the run still finishes as a success. Nothing is
written when step 1 or 2 fails. When publishing fails, the files are still in the output folder.
When no layer at all is kept, an error-level message says so and an empty catalogue is still
published.

## Inputs

* **`metadata_file`** — required (see Parameters).
* **`configuration/SNT_config.json`** for **`SNT_CONFIG.COUNTRY_CODE`** and
  **`SNT_DATASET_IDENTIFIERS`** (every key named by a layer's `DATASET.NAME`).
* **Every dataset file referenced by the catalogue.** Each one is optional: a missing one drops its
  layers, it does not stop the run.

## Outputs

**Workspace filesystem** — `pipelines/snt_metadata_json_trim/output/`

* **`SNT_metadata_trimmed.json`** — the kept layers, formatted like the generator's output
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
> - **Report fields:** `RUN_AT`, `COUNTRY_CODE`, `INPUT_FILE`, `N_LAYERS_INPUT` / `_KEPT` /
>   `_DROPPED`, `KEPT` (layer ids), `DROPPED` (`LAYER_ID`, `REASON`, `DETAIL`, `DATASET_NAME`,
>   `FILENAME`, `COLUMN`), and `SOURCES`: one entry per file checked, including the **resolved
>   dataset version**, which records which vintage each kept layer was validated against.
> - **A kept layer is only as fresh as its dataset version.** The check is made against whatever
>   version is latest *at run time*. Re-run this pipeline after upstream pipelines publish.
> - **"Last run wins" is visible here.** Alternative pipelines that publish to the same dataset
>   (the two reporting-rate variants, quality of care `imputed` / `removed`) leave only the most
>   recent variant's file in the latest version, so the other variant's layers are dropped as
>   `FILE_NOT_FOUND`. That is correct: the Explorer would not find them either.
> - **Cost:** each referenced file is downloaded once, in full, to read its header. That is fine for
>   the ADM2-level outputs in the catalogue today; it would be slow for facility-level files.

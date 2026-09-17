# SNT Workspace Manager Pipeline

The **SNT Workspace Manager** deploys one pinned GitHub release of this repository into the
workspace it runs in. It moves both halves of the codebase on a single release tag: the R analytics
(notebooks and `.r` helpers) are copied into the workspace filesystem, and each `<name>/pipeline.py`
is registered as a new **version** of the matching OpenHEXA pipeline through the GraphQL API. It
publishes nothing to an OpenHEXA dataset — its output is the state of the workspace itself, plus a
`.snt_release` marker file at the workspace root.

It is the Phase 3/4 prototype of [`docs/wip/release_strategy.md`](../docs/wip/release_strategy.md);
the deployment mechanism is documented in
[`docs/wip/pipeline_deployment_mechanism.md`](../docs/wip/pipeline_deployment_mechanism.md).

> **Status: prototype.** Verified end to end on 2026-09-16 in `snt-development-sandbox` for
> `snt_dhis2_extract` and `snt_map_extracts` only — 2 of 20 pipelines. It replaces the per-pipeline
> `Pull scripts` toggle and the Template auto-update subscription, so do not run it against a
> country workspace until the open items in the two documents above are settled.

## Parameters

This pipeline does not read `SNT_config.json` and has no country-specific parameters. None of the
standard SNT flags (`run_report_only`, `pull_scripts`, `overwrite`) apply — `pull_scripts` in
particular is the mechanism this pipeline replaces.

* **`github_repo`** (str, Required):
  * **Name:** GitHub repository
  * **Description:** `owner/repo` to pull the release from. The release must carry a
    `release_manifest.json` asset listing the tracked files and their sha256 hashes.
  * **Default:** `BLSQ/snt_development_sandbox`.
* **`release_tag`** (str, Required):
  * **Name:** Release tag
  * **Description:** The GitHub release tag to deploy. It also becomes the **version name** of every
    pipeline version registered by the run, and the name of the backup subdirectory.
  * **Default:** `None` — the operator must supply it.
* **`api_connection`** (str, Optional):
  * **Name:** OpenHEXA API connection
  * **Description:** Slug of a **CUSTOM** connection holding a workspace API token in a secret field
    named `token`. A run's own `HEXA_TOKEN` is refused with `PERMISSION_DENIED` on `uploadPipeline`,
    so deployment cannot use the run's own credentials. Read only when `deploy_pipelines` is on.
  * **Default:** `oh`.
* **`sync_analytics`** (bool, Optional):
  * **Name:** Sync R analytics to the filesystem
  * **Description:** Copy the release's notebooks and `.r` helpers into the workspace filesystem.
  * **Default:** `True`.
* **`deploy_pipelines`** (bool, Optional):
  * **Name:** Deploy pipelines
  * **Description:** Register each release `pipeline.py` as a new version of the matching OpenHEXA
    pipeline.
  * **Default:** `True`.
* **`only_pipelines`** (str, Optional):
  * **Name:** Only these pipelines
  * **Description:** Comma-separated pipeline directory names to deploy, e.g.
    `snt_dhis2_extract, snt_map_extracts`. Hyphens are accepted and normalised to underscores. A
    name the release does not define **raises** rather than being skipped. Empty means every
    pipeline in the release. Does not affect the analytics sync.
  * **Default:** `None`.
* **`create_missing`** (bool, Optional):
  * **Name:** Create pipelines that do not exist yet
  * **Description:** Bootstrap an empty workspace by creating any pipeline the release defines but
    the workspace lacks. Off by default so a typo cannot silently create a duplicate pipeline.
  * **Default:** `False`.
* **`backup_existing`** (bool, Optional):
  * **Name:** Backup existing files
  * **Description:** Move any existing tracked file to `archive/[RELEASE_TAG]/` before overwriting
    it. Applies to the analytics sync only — pipeline versions are never overwritten, they are
    appended.
  * **Default:** `True`.
* **`dry_run`** (bool, Optional):
  * **Name:** Dry run
  * **Description:** Report what would change without writing any file, registering any version or
    writing the release marker.
  * **Default:** `False`.

## Functionality Overview

1. **Credentials:** When `deploy_pipelines` is on, read the bearer token from the custom connection
   named by `api_connection`; a missing or malformed connection aborts the run before anything is
   fetched.
2. **Resolve the release:** Fetch the GitHub release by tag, then download its
   `release_manifest.json` asset. A release without that asset aborts the run.
3. **Split the manifest:** Any entry of the form `<name>/pipeline.py` identifies a **pipeline to
   deploy**; everything else is an **analytics file to copy**. `pipeline.py` is deliberately
   excluded from the filesystem sync, because OpenHEXA runs each pipeline from its registered
   version's zip and never from the workspace bucket.
4. **Download the source tarball** once for the whole repository and extract it to a temporary
   directory (a per-file Contents API fetch would exhaust the unauthenticated rate limit).
5. **Sync analytics** (when `sync_analytics` is on): copy each tracked analytics file into
   `workspace.files_path`, archiving any existing copy first when `backup_existing` is on. A
   manifest entry missing from the tarball is logged as a **warning** and skipped, not raised.
6. **Deploy pipelines** (when `deploy_pipelines` is on): for each selected pipeline directory, parse
   its parameters with the SDK's AST-based `get_pipeline()` (no import, so the pipeline's own
   dependencies need not be installed), zip the whole directory (`.py`, `.ipynb`, `.txt`, `.md`,
   `.r`, `.sql`), and call `uploadPipeline` — or `createPipeline` with a nested version when the
   pipeline does not exist and `create_missing` is on. The OpenHEXA pipeline code is the directory
   name with `_` → `-`.
7. **Continue on failure:** one pipeline's failure is logged with an `[ERROR]` prefix and does not
   stop the others; the run raises at the end listing every failure, so re-running converges.
8. **Write the release marker:** record `{"snt_release": "<tag>"}` in `.snt_release` at the
   workspace root. Skipped on a dry run.

## Inputs

* **GitHub release `[RELEASE_TAG]` of `[GITHUB_REPO]`** — required. Read unauthenticated, so the
  repository must be public or the run fails.
  * **`release_manifest.json`** release asset — required; the list of tracked files.
  * **The release source tarball** — required; the actual file contents.
* **CUSTOM connection `[API_CONNECTION]`** with a secret field `token` — required when
  `deploy_pipelines` is on.
* **`HEXA_SERVER_URL`** from the run environment — the GraphQL endpoint.
* **No `SNT_config.json`, no OpenHEXA dataset, no country code.**

## Outputs

**Workspace filesystem**

* **Every tracked analytics file** at its repository-relative path — `pipelines/<name>/code/*.ipynb`,
  `pipelines/<name>/utils/*.r`, `code/*.r`.
* **`archive/[RELEASE_TAG]/<path>`** — the previous copy of each overwritten file, when
  `backup_existing` is on.
* **`.snt_release`** at the workspace root — the currently-deployed release tag.

**Published to a dataset**

* Nothing. This pipeline publishes no dataset and calls no `add_files_to_dataset(...)`.

**OpenHEXA object store**

* **A new version of each deployed pipeline**, named after the release tag, described as
  `Deployed by snt_workspace_manager from release <tag>`, with `externalLink` set to the release's
  GitHub page.

> **Notes for the Data Analyst:**
>
> - **Not an analytics pipeline.** It produces no data and touches no country data. It is a
>   deployment tool; the audience is whoever administers a workspace, not whoever reads results.
> - **Deployment is not a file copy.** `pipeline.py` is registered through the API, never written to
>   the workspace filesystem — a `pipeline.py` sitting in the bucket would look authoritative and
>   never execute.
> - **The zip carries the whole directory,** so `requirements.txt` and `readme.md` are deployed even
>   though the manifest tracks only `pipeline.py`. This is the manifest gap described in
>   `release_strategy.md`: verification currently covers less than what is actually deployed.
> - **Partial runs are expected to be re-run.** A failure leaves the workspace partially updated, by
>   design — every failure is named in the final error and re-running converges on the release.
> - **`default=""` breaks a deploy, not a run.** A `str` parameter with an empty-string default is
>   rejected by the SDK's AST parse, so it fails here rather than in the target pipeline. Use
>   `default=None`.
> - **`create_missing` derives the code from the pipeline name.** The run verifies the created code
>   matches the expected slug and raises if it does not; recovery is manual (delete and recreate in
>   the UI).
> - **R5 interaction.** This pipeline pushes versions **directly into the workspace that runs them**,
>   which creates no OpenHEXA template and so is a different operation from the one
>   [`CLAUDE.md`](../CLAUDE.md) **R5** governs. R5 needs rewording before this is used in production
>   — see `pipeline_deployment_mechanism.md`.

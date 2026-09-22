# SNT Workspace Check Pipeline

Reports what is actually deployed in this OpenHEXA workspace, compared against one GitHub release
of the SNT codebase. It is **read-only**: it writes a status report and changes nothing else —
installing and updating is `snt_workspace_manager`'s job. The report is a JSON file on the workspace
filesystem under **`snt_status/`**, not an OpenHEXA dataset (decision D3).

> **Phase 1 of [`docs/wip/PRODUCT_SPEC.md`](../docs/wip/PRODUCT_SPEC.md) §6.** Verification against a
> single target release, four statuses. The report shape is provisional until `schema_version: 1` is
> frozen at phase 4, and this readme is re-verified against the code then.

## Parameters

* **`github_repo`** (String, Required):
  * **Name:** GitHub repository
  * **Description:** The `owner/repo` whose releases this workspace is checked against. A parameter
    while the product is in testing; it will be hard-coded before production, so a user cannot point
    the checker at an arbitrary repository (`PRODUCT_SPEC.md` §5.4).
  * **Default:** `BLSQ/snt_development_sandbox`.
* **`release_tag`** (String, Optional):
  * **Name:** Target release tag
  * **Description:** The release to check this workspace against, e.g. `v0.2.1-test`. Left empty, the
    pipeline falls back to the tag recorded in `.snt_release` by the last `snt_workspace_manager`
    run. With neither, it stops with an explanation rather than checking against nothing —
    attribution mode (assessing a workspace with no target at all) is phase 3.
  * **Default:** none.

This pipeline takes **no credential parameter and needs no connection.** A run's own `HEXA_TOKEN`
reads pipeline version contents in full ([`HISTORY.md`](../docs/wip/HISTORY.md) §2.4), so it can run
unattended in a country workspace that holds no connection at all.

## Functionality Overview

1. Read `.snt_release` at the workspace root, if present, for the **declared** release. Its absence
   is the normal starting state for every country workspace today and is never an error.
2. Resolve the **target** release: the `release_tag` parameter, else the declared release, else stop.
   The report always records which of the two was used.
3. Fetch the release from the GitHub API and download its **`release_manifest.json`** asset. A target
   release with no manifest asset stops the run — there is nothing to check against.
4. Read the manifest's **`pipelines`** block and split the tracked paths **by directory** into two
   sources. A manifest with no such block (pre-2026-09-21) is refused rather than guessed at.
5. **Filesystem source** — hash every tracked path under `workspace.files_path` that is *not* inside
   a pipeline directory: the notebooks, the `utils/*.r` helpers and the shared `code/*.r` library.
6. **Pipeline-version source** — for each pipeline directory, read its **current registered version**
   through the OpenHEXA API, decode the stored zip and hash every file inside it. Older versions are
   not read: they are not what would run. A pipeline the release defines but the workspace has not
   deployed yields `missing` for all of its files; an API failure yields `unreadable` and does not
   stop the other pipelines.
7. Compare each file's sha256 (raw bytes, **no notebook normalisation** — that is decision D9, to be
   taken after measuring real drift in phase 2) against the target manifest, and assign one status.
8. Write the report twice, and log a summary plus one line per non-`match` file.

Each tracked path is read from **exactly one** source. Anything inside a pipeline directory comes
from the version zip, and a copy of the same file sitting on the workspace filesystem is deliberately
**not** consulted: OpenHEXA runs pipelines from the registered version and never from the bucket, so
such a copy is inert, and treating it as evidence would report a file as fine on the strength of
bytes that never execute.

### Statuses this phase can report

| Status | Meaning |
|---|---|
| `match` | Present; hash equals the target manifest's. |
| `unknown_content` | Known path, unexpected bytes — edited in place, corrupt, or simply from a different release. Phase 1 holds only the target manifest, so it cannot yet tell those apart; phase 3 splits the third case out as `mismatch_known`. |
| `missing` | In the target manifest; absent from both sources. |
| `unreadable` | Present but could not be hashed (permissions, I/O, API error). Always sets `incomplete: true`. |

`mismatch_known`, `removed_in_target`, `untracked`, `not_covered` and `position` are phase 2/3 and
are named in the report's `blind_spots` list, so it never reads as a clean bill of health for things
it did not look at.

## Inputs

| Input | Source | Required |
|---|---|---|
| `release_manifest.json` | GitHub release asset on the target tag | Yes |
| `.snt_release` | Workspace root, written by `snt_workspace_manager` | No — only used when `release_tag` is empty |
| Tracked analytics files | Workspace filesystem, at their repository-relative paths | No — absence is the `missing` finding |
| Current pipeline version zips | OpenHEXA API, `pipelineByCode.currentVersion.zipfile` | No — absence is the `missing` finding |

This pipeline does **not** read `configuration/SNT_config.json`, and takes no country code: it checks
code, not data, so nothing about it is country-specific.

## Outputs

Written to the **workspace filesystem** (no dataset, no database table):

* **`snt_status/status_<UTC timestamp>.json`** — the run's report, kept as history.
* **`snt_status/status_latest.json`** — a byte-identical copy at a stable path. This is the one the
  future status web app reads.

Nothing is published to an OpenHEXA dataset.

> **Notes for the Data Analyst:**
>
> - **`status`** and **`position`** are stable enums — a value may be added but never renamed, so a
>   consumer can switch on them.
> - **`remediation`** is display text for a human. Never parse it.
> - **`incomplete`**: `true` means at least one source could not be read. The report is then a
>   partial account, not a pass.
> - **`declared_release`** vs **`target_release`**: the declared one is what `.snt_release` says was
>   last deployed. It is written even after a partial run, so it records **intent, not verified
>   fact** — when the two disagree, the run logs a warning and checks against the target.
> - **`pipelines[].version_name_matches_content`**: a pipeline version's *name* is free text somebody
>   typed; its hash is evidence. When they disagree the hash wins and the name is flagged as
>   misleading — a workspace whose version labels have stopped meaning anything looks perfectly
>   healthy in the OpenHEXA UI, which shows only names. It is `null` for a version named after some
>   release other than the target, which phase 1 cannot verify either way.
> - **`current_version_name`** is the raw API value and **`current_version_claims_tag`** is the
>   release tag parsed out of it. They differ because OpenHEXA appends the version number:
>   `v0.1.0-test` reads back as `v0.1.0-test [v1]`. Both are reported so the parsing is visible.
> - **`pipelines[].files_in_zip_not_in_manifest`**: a count only. Per-file reporting of those is
>   phase 2. A non-zero count here is worth a look — it means something ships in a deploy zip that
>   no release describes.
> - Every field that could be absent is present as `null` rather than omitted.

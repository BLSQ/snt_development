# SNT Release Management — Product Spec

> Status: specification, work in progress. This is where the checker's **requirements** live;
> [`release_strategy.md`](release_strategy.md) keeps the *why* and the state of what is built,
> [`pipeline_deployment_mechanism.md`](pipeline_deployment_mechanism.md) the deployment *how*, and
> [`HISTORY.md`](HISTORY.md) everything that is no longer true — superseded designs, closed issues,
> dead ends and deleted fixtures. Check `HISTORY.md` before reopening any question here.
>
> Nothing specified here is live in a country workspace. Sections marked **BUILT** describe code
> that exists and has been verified in a sandbox; everything else is a requirement, not a report.

## 1. The product

A country workspace should be able to answer two questions without anyone reading a diff:

1. **What is in here?** Which release does each file in this workspace belong to, and is anything
   of unknown origin?
2. **Is it what I asked for?** Given a target release, what matches, what is behind, what is ahead,
   what is missing — and can I bring it into line, or deliberately keep what I have?

The user-facing product is eventually an OpenHEXA web app. The web app is a UI/UX layer only: the
work that cannot be done from a web app — pulling files into the workspace filesystem and
registering pipeline versions through the OpenHEXA API — is done by pipelines. **This spec covers
the pipelines. The web app is out of scope beyond the report contract it will consume (§5.5).**

### 1.1 Components

| Component | Role | State |
|---|---|---|
| `snt_workspace_manager` | **Fix / install.** Deploys one pinned release into the workspace: R analytics to the filesystem, `pipeline.py` via the API. | **BUILT**, prototype — verified for 2 of 20 pipelines |
| `snt_workspace_check` *(proposed name)* | **Check.** Read-only. Hashes what is actually in the workspace, attributes each file to a release, and writes a status report. | Not started — the subject of this spec |
| Release manifest generation | GitHub Action producing `release_manifest.json` per release | **BUILT** — covers everything the deploy zip ships |
| Status web app | Reads the checker's report; offers "fix" or "leave as is" | Deferred |

The checker and `snt_workspace_manager` are **separate pipelines** (decision D2). The checker
never writes anything except its own report; `snt_workspace_manager` is the only component that changes
workspace state.

### 1.2 In scope / out of scope

In scope for the checker: everything the release manifest tracks — `pipelines/**/code/*.ipynb`,
`pipelines/**/reporting/*.ipynb`, `pipelines/**/utils/*.r`, `code/**/*.r`, and each deployed
pipeline's registered version contents.

Out of scope, and to be stated as blind spots in the report:

* **OpenHEXA web apps** (decision D8) — there is no deployment story for them yet; see
  `release_strategy.md`.
* **`configuration/`** — managed by the Config Editor web app.
* **`data/`** — bootstrapped by the pipelines themselves.
* **Country-specific notebook variants as *overrides*** (decision D5) — in v1 they fall out as
  ordinary untracked files. Recognising them as deliberate overrides is deferred, and until it is
  done the report will look alarming in any workspace that has one.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Release tag** | A GitHub release tag, e.g. `v1.2.0`. The single source of truth for "what version is this workspace on". Tags are protected and never moved (`release_strategy.md` §"Release tags are never moved"). |
| **Release manifest** | `release_manifest.json`, attached as an asset to each release: `{version, files: {path: sha256}, pipelines: {dir: {code, zip_files}}}`. `files` is flat and covers both sources; the `pipelines` block says which of those paths ship inside which pipeline zip, and under what name once inside it (§2.1). |
| **Tracked file** | A path present in some release manifest. |
| **Target release** | The release the workspace is being compared *against*, when one is given. |
| **Declared release** | What `.snt_release` says the workspace was last deployed to. Intent, not verified fact (§3.2). |
| **Attribution** | The set of releases whose manifest contains a file's observed hash. |
| **Drift** | An observed hash that matches no manifest of any release. |

### 2.1 The manifest's `pipelines` block

`files` alone does not say *where* an entry lives in a workspace, and
the two answers are not interchangeable: `code/snt_utils.r` is a file on the filesystem, while
`snt_map_extracts/utils.py` exists **only** inside a pipeline's registered version zip and is at no
path on the filesystem at all. Without the block, every consumer has to re-derive that split by
reapplying the generator's rule, and a consumer that gets it wrong reports `missing` for files that
are deployed and correct — a false alarm in the one component whose job is to be trusted.

```json
"pipelines": {
  "snt_map_extracts": {
    "code": "snt-map-extracts",
    "zip_files": ["malariaAtlasProject/__init__.py", "pipeline.py", "readme.md", "utils.py"]
  }
}
```

* The **key** is the repository directory. Prefix it to a `zip_files` entry to get the repository
  path, which is the key into `files` and therefore the expected hash.
* `zip_files` are paths **inside the zip**, so they can be compared directly against the
  `namelist()` of what `get_pipeline` returns.
* `code` is the OpenHEXA pipeline code — the directory name with underscores replaced by hyphens.
  A consumer needs it to look the pipeline up over the API and can get it from nowhere else in the
  manifest. Verified against all 20 `push_snt_*.yaml` `--code` values and against
  `snt_workspace_manager`'s own derivation.

`files` keeps its exact previous shape, so the block is additive: a reader that ignores `pipelines`
behaves as before. Legacy manifests have no such block, and a consumer meeting one must fall back to
deriving pipeline directories from `<name>/pipeline.py` entries — `split_manifest()` in
`snt_workspace_manager` is the reference for that fallback.

**Open:** no live release lacks the block, so the fallback is unreachable in testing. Either keep a
legacy manifest as a local test fixture or delete the fallback in a PR of its own — untested
back-compat code for a case that can no longer occur is worse than either.

## 3. What the checker observes

### 3.1 Two sources of truth, hashed separately

Every entry in the report carries which source it came from.

| Source | What is hashed | Notes |
|---|---|---|
| `filesystem` | Files under `workspace.files_path` at their repository-relative paths | Where the R analytics live |
| `pipeline_version` | The files inside each OpenHEXA pipeline's **current registered version zip** | `pipeline.py` is *never* on the filesystem — a copy there is inert and misleading (`pipeline_deployment_mechanism.md`) |

Both are read from v1 (decision D7). The zip is readable via `get_pipeline`, which returns full
file contents. **Unverified:** whether reading pipeline versions works with a run's own
`HEXA_TOKEN`, or needs the `oh` connection token as *deployment* does. This must be established
early — it decides whether an unattended check needs a workspace-scoped credential (§7.3).

#### Which pipeline version is hashed, and why the version name is not evidence

An OpenHEXA pipeline is not a file — it is a registered object that accumulates **versions**, each
storing its own zipped copy of the code. A pipeline that has been deployed ten times has ten
versions; only one of them, the **current version**, is what a run actually executes. So:

* **The checker hashes the current version only.** Older versions are history. They are not what
  would run, and reporting on them would drown the report in files nobody can act on.
* **Each version has a *name*, which is free text.** `snt_workspace_manager` sets it to the release
  tag it deployed from, so in the normal case a pipeline whose current version is called
  `v0.2.1-test` really does contain `v0.2.1-test`'s code.

The name is metadata someone typed; the hash is evidence. They can disagree, and the ways they
disagree are exactly the ways the old system failed silently:

* somebody deployed by hand with the CLI, from a working tree that was not at that tag, and named
  the version after the tag anyway;
* `snt_workspace_manager` was run at a tag but a source file had been edited in the workspace
  before the zip was built;
* a version was named by hand in the UI.

So the rule is: **when the name and the contents disagree, the hash wins and the name is reported
as misleading.** This is not a per-file status — it is a per-pipeline flag in the report
(`current_version_name` alongside `version_name_matches_content: false`), because the mismatch is a
property of the deployment, not of any one file inside the zip. The point of surfacing it at all is
that a workspace where the version labels have stopped meaning anything looks perfectly healthy
from the OpenHEXA UI, which only shows the names.

### 3.2 `.snt_release`

`snt_workspace_manager` writes `{"snt_release": "<tag>"}` at the workspace root at the end of
every run. Two honest limitations, both of which the report must reflect rather than paper over:

* It records only the tag, **not the repository it came from**, while `github_repo` is still a
  parameter.
* It is written after a partial run too, so it states **intent**, not verified fact.

A workspace with no marker is the normal starting state today — every existing country workspace.
That must never be an error.

## 4. The two modes

The checker has one parameter that decides its mode: an optional target release.

### 4.1 Attribution mode — no target release given

Assess every file and report which release it belongs to, or that it belongs to none. No verdicts,
no green or red: a factual inventory plus a distribution, so the web app can show e.g. *"97% of
files are v0.1.0-test, 2.7% are v0.2.1-test, 0.3% are of unknown origin."*

This mode needs the manifest of **every** release, which is the open decision in §7.2 — it is on
this mode's critical path.

### 4.2 Verification mode — target release given

Everything attribution mode reports, plus a qualitative layer per file relative to the target.

The target release is resolved in this order, and the report always records which was used:

1. the `release_tag` parameter, if given;
2. otherwise `.snt_release`, if present → mode 4.2;
3. otherwise no target → mode 4.1.

## 5. Requirements

### 5.1 Status taxonomy

Per entry, exactly one status:

| Status | Meaning | Mode |
|---|---|---|
| `match` | Present; hash equals the target manifest's | 4.2 |
| `mismatch_known` | Present; hash differs from target but matches ≥1 other release. Carries `matching_releases` and `position` (`older` / `newer` / `both` / `unordered`) | 4.2 |
| `unknown_content` | **Known path, unknown content.** The path is tracked, but its hash matches no release. Edited by hand, or corrupted | both |
| `missing` | In the target manifest; absent from both filesystem and pipeline versions | 4.2 |
| `removed_in_target` | Present, and in an older manifest, but not in the target's | 4.2 |
| `untracked` | **Unknown path.** This path appears in no manifest of any release — the repo has never shipped a file here | both |
| `not_covered` | Deployed inside a pipeline zip, but no manifest describes it. Should not occur now the manifest mirrors the SDK's zip rule; retained as the tripwire for that rule drifting | both |
| `unreadable` | Present but could not be hashed (permissions, I/O, API error) | both |

In attribution mode, files resolve to `matching_releases` (see §5.1.2), `unknown_content`, or
`untracked`.

#### 5.1.1 `unknown_content` vs `untracked` — two different questions

They sound alike and are not. One is about the **path**, the other about the **bytes**:

| | Is this path in any manifest? | Does the content match any release? | Meaning |
|---|---|---|---|
| `unknown_content` | **yes** | no | A file we ship, whose copy here is not any version we ever released. Someone edited it, or it is corrupt. **Actionable** — this is drift. |
| `untracked` | **no** | not asked | A file the release has no opinion about: an analyst's scratch notebook, a stray export. **Not actionable** — reported for completeness only (§5.3). |

The content question is never even asked for an `untracked` file, because there is no manifest
entry to compare it against.

#### 5.1.2 Attribution is a set, not a value — and it needs collapsing

**"Belongs to release X" is not a function.** A file that did not change between `v0.1.0` and
`v0.6.0` has one hash that is correct for all seven of those releases. A single verdict only exists
relative to a target release; attribution on its own is a *set*.

Enumerating that set per file would make the report unreadable — most files are unchanged most of
the time, so most entries would carry a list of nearly every release that exists. The report must
collapse it. Proposed representation, **to be confirmed** (§7.8):

```json
"matching_releases": [{"from": "v0.1.0", "to": "v0.6.0", "count": 7}]
```

i.e. contiguous spans in `published_at` order rather than a flat list, with the common case
rendering as a single span the web app can show as "unchanged since v0.1.0". A non-contiguous set
(content introduced, changed, then reverted) yields more than one span, which is itself the signal
worth seeing.

### 5.2 Ordering: behind vs ahead

`older` / `newer` are computed from the GitHub release **`published_at`** timestamp (decision D4),
not from tag-string parsing — the sandbox's `-test` tags do not parse as semver, and tag protection
makes republication of an old release a non-concern. A release whose timestamp cannot be read is
reported `unordered` rather than guessed.

`position` describes the matching span relative to the **target**, and all four values are reachable:

| Value | When |
|---|---|
| `older` | Every release matching this content is before the target — the file is **behind**. |
| `newer` | Every match is after the target — the file is **ahead**. |
| `both` | Matches exist on both sides of the target. |
| `unordered` | At least one matching release has no usable timestamp. |

`both` is not the unchanged-across-releases case — an unchanged file that includes the target in
its span is simply `match`. It means the content was **changed in the target release and later
reverted**: `v1` has hash H, `v2` (the target) has H′, `v3` goes back to H. A workspace holding H
then matches a release older *and* a release newer than the target, while matching the target
itself not at all. Rare, but it happens whenever a change is rolled back, and calling it "behind"
would be wrong.

### 5.3 Untracked files

Reported in an `untracked` bucket with count and paths (decision D6); never acted on. One case is
sharper than the rest and must be marked as such: **an untracked file inside a pipeline directory
ships in the deployment zip**, so it is not inert the way a stray file on the filesystem is.

Excluded from the scan entirely, so the bucket stays readable: `archive/`, `papermill_outputs/`,
`reporting/outputs/`, `data/`, `configuration/`, and any checker output directory.

### 5.4 Non-negotiables

* **The checker is read-only.** It writes its report and nothing else. It never deletes, moves,
  overwrites, deploys or archives — `snt_workspace_manager` does that.
* **Nothing is ever deleted, by either component.** Superseded files are moved to
  `archive/<release_tag>/`, findable by the user, who deletes manually if they want to.
* **Verbose and explicit.** No black-box feeling: every decision that shaped the verdict is either
  logged or in the report. With 100+ tracked files, per-file logging is a summary in the log and
  the full detail in the JSON — not 300 log lines.
* **Partial results are labelled.** If a source could not be read, the report carries
  `incomplete: true` and says which, rather than implying a clean bill of health.
* **The repository is a parameter now, hard-coded later.** Testing runs against
  `BLSQ/snt_development_sandbox`; the shipped product must not let a user point it anywhere.

### 5.5 The report contract

A JSON file, written to the workspace filesystem, timestamped **and** latest (decision D3):

```
<workspace root>/snt_status/status_<UTC ISO timestamp>.json
<workspace root>/snt_status/status_latest.json      ← the stable path the web app reads
```

Shape (v1 — to be frozen at the end of phase 4, §6):

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-18T10:00:00Z",
  "checker_version": "<pipeline version tag>",
  "workspace": "<slug>",
  "repo": "BLSQ/snt_development_sandbox",
  "mode": "attribution | verification",
  "target_release": {"tag": "v0.2.1-test", "resolved_from": "parameter | marker", "published_at": "..."},
  "declared_release": {"tag": "v0.1.0-test", "source": ".snt_release"},
  "releases_considered": [{"tag": "...", "published_at": "...", "manifest_available": true}],
  "incomplete": false,
  "summary": {
    "by_status": {"match": 101, "mismatch_known": 3, "untracked": 2},
    "attribution": {"v0.1.0-test": 0.97, "v0.2.1-test": 0.027, "unknown": 0.003}
  },
  "entries": [
    {
      "path": "pipelines/snt_dhis2_incidence/code/snt_dhis2_incidence.ipynb",
      "source": "filesystem | pipeline_version",
      "pipeline": "snt_dhis2_incidence",
      "status": "mismatch_known",
      "observed_sha256": "...",
      "target_sha256": "...",
      "matching_releases": ["v0.1.0-test"],
      "position": "older",
      "remediation": "Run snt_workspace_manager at v0.2.1-test to update; the current copy is archived first."
    }
  ],
  "errors": [{"scope": "...", "message": "..."}]
}
```

Rules: `status` and `position` are **stable enums** — the web app switches on them, so a value is
added, never renamed. `remediation` is human-readable text for display, never parsed. Every field
that could be absent is present with `null` rather than omitted.

## 6. Build phases

Each phase ends with something demonstrable. Do not start a phase whose blocking decision (§7) is
still open.

| # | Deliverable | Exit criterion | Blocked by |
|---|---|---|---|
| **0** | ✅ **DONE** — manifest gap closed, sandbox reset, fixture releases cut (§6.1). Details: [`HISTORY.md`](HISTORY.md) §2.1, §4. | — | — |
| **1** | Checker skeleton: verification mode against a single target release, both sources hashed, statuses `match` / `unknown_content` / `missing` / `unreadable`, report written. | A workspace freshly deployed by `snt_workspace_manager` at tag T reports all-`match`. | 0, and the token question in §7.3 |
| **2** | Full taxonomy: `removed_in_target`, `untracked`, `not_covered`, the pipeline-directory case. Plus **measure notebook drift** on a real workspace and decide D9. | A workspace at T-1 with one hand-edited file reports exactly the expected mix. | 1 |
| **3** | Attribution mode: all releases, ordering, distribution summary. | A mixed workspace produces a correct per-release percentage breakdown. | §7.2 — **open** |
| **4** | Freeze `schema_version: 1`. Pipeline `readme.md` per [`docs/PIPELINE_README_STANDARD.md`](../PIPELINE_README_STANDARD.md); commit to the repo. | Report schema documented; readme verified against the code, not memory. | 3 |
| **5** | `snt_workspace_manager` integration: report before and after a fix; enrich `.snt_release` (§7.4). | A fix run links to the before/after reports it produced. | 4 |
| **6** | Web app. | Out of scope for this spec. | 5 |

### 6.1 Test fixtures in the sandbox

The fixture releases exist in `BLSQ/snt_development_sandbox` and are verified. Two naming rules:
earlier `-test` tags are retired and must not be reused ([`HISTORY.md`](HISTORY.md) §4.2), and do
**not** create a release named `latest` — GitHub's `/releases/latest` endpoint already resolves to
the newest non-prerelease release automatically.

| Tag | Asset | Role |
|---|---|---|
| `v0.1.0-test` | manifest, 163 files | baseline; deploy target for check run 1 |
| `v0.2.0-test` | manifest, 163 files | **dud**, byte-identical to `v0.1.0-test`; kept because tags are never deleted |
| **`v0.2.1-test`** | manifest, 163 files | **the verification target** |
| `v0.3.0-test` | manifest, 163 files | ahead of the target; deploy target for check run 2 |
| `v0.4.0-test` | **none** | `manifest_available: false`, `incomplete: true` |

The dud is harmless: its `published_at` puts it on the older side of the target, where duplicate
content changes no expected status, and it incidentally covers the case of **two releases with
identical manifests**, which attribution must not double-count.

The set produces every status at least once:

* a file **changed** between two releases → `mismatch_known` in both directions;
* a file **added** in the newer release → `missing` when checking a workspace at the older one;
* a file **removed** in the newer release → `removed_in_target`;
* a **pipeline added** and a **pipeline removed** between releases;
* a file **unchanged across all releases** → attribution to several releases at once;
* a file **changed and then reverted** → `position: both`, and the only non-contiguous
  attribution span (§5.1.2, §7.8);
* a release published with **no manifest asset** → `manifest_available: false` and
  `incomplete: true` (§5.4, §5.5), made by disabling the generator workflow for one release;
* one hand-edited file in the workspace → `unknown_content`.

Four cases are **not** release fixtures and are made in the workspace immediately before a check
run: `unknown_content`, `untracked`, `unreadable`, and the §5.3 untracked-inside-a-pipeline-zip
case. One more, `position: unordered`, is not reachable at all — GitHub always sets `published_at`
— so it is a unit test against a stubbed release list, not a fixture.

The full command-by-command plan, with the expected status for each fixture in each of three check
runs, is `ignore/SNT25-670/sandbox_fixture_plan.md` (local, not committed — it contains `git`/`gh`
commands for a human to run).

## 7. Open decisions

Blocking ones name the phase they block. None may be resolved by guessing.

### 7.1 ~~The manifest under-described what is deployed~~ — **closed**

Closed as phase 0. Kept as a numbered heading only so the §7.x references elsewhere still line up.
What the problem was, why a wider glob list was rejected, and the consumer it broke:
[`HISTORY.md`](HISTORY.md) §2.1–2.2. What the generator does now:
[`release_strategy.md`](release_strategy.md) § "Manifest generation".

Phase 1 is therefore blocked only by the token question in §7.3.

### 7.2 How to obtain every release's manifest — **blocks phase 3**

Attribution mode needs the manifest of every release. Unauthenticated GitHub API is 60 requests
per hour and this project has already hit that wall once ([`HISTORY.md`](HISTORY.md) §1). With ~20
releases and a per-workspace daily check, a naive implementation breaks. Candidate answers —
workspace-side manifest cache keyed by tag (safe, because tags never move), a cumulative index
asset published by the Action, or an authenticated token (which re-opens §7.3). **Deferred by the
user to a dedicated session (2026-09-18).**

### 7.3 Credentials in a country workspace — blocks production use of either component

Deployment requires a workspace API token from a CUSTOM connection named `oh`; a run's own
`HEXA_TOKEN` is refused. Who mints it, where it is stored and how it is rotated is undecided
(`pipeline_deployment_mechanism.md` §"Authentication"). Sub-question this spec adds: **does
*reading* a pipeline version need the same credential?** If not, the checker can run unattended in
a workspace that holds no token at all — a materially better story. Establish it in phase 1.

### 7.4 Enriching `.snt_release`

The marker should arguably record the repository, a timestamp, and whether the run completed
cleanly, so a checker can tell "deployed to T" from "attempted T, partially". Changing it means
changing `snt_workspace_manager` and handling markers written by older versions.

### 7.5 Pipelines removed in the target release

A file can be archived. An **OpenHEXA pipeline object cannot be removed without a destructive
action**, and this repo forbids agents from performing those. The only honest behaviour is
report-and-leave, with the report saying plainly that the pipeline is no longer part of the
release. Confirm this is what the team wants before phase 2.

### 7.6 Scheduling

A daily unattended check was the original motivation for splitting check from fix. Whether SNT
pipelines can be scheduled in OpenHEXA in practice — and whether a daily run is wanted per
workspace — is unconfirmed. The repo's pipelines are all launched by hand today.

### 7.7 R5

`CLAUDE.md` **R5** ("always publish from `snt-development`") was written about *template*
publication. Pushing a pipeline version into the workspace that runs it creates no template and is
a different operation. R5 needs rewording, with the team's agreement, before production use —
see `pipeline_deployment_mechanism.md`.

### 7.8 How to represent attribution without flooding the report — **think this through before phase 3**

Raised in review, 2026-09-18, and not fully solved. Most files do not change in most releases, so
a naive `matching_releases` list means nearly every entry in the report carries nearly every
release tag that exists — the report grows with release history rather than with what is wrong,
and the web app inherits the problem.

The span representation proposed in §5.1.2 (`{"from", "to", "count"}` in `published_at` order) is
the current candidate, not a settled answer. Questions it leaves open:

* Is a span enough, or does the web app need the full list somewhere for a detail view?
* What is shown when a file matches a single release — a span of one, or a plain tag?
* Do spans need to survive a release whose manifest could not be fetched (a hole in the middle of a
  span is not the same as a break in the span)?
* Should the report even carry per-file attribution for files that are `match` against the target,
  or is "matches target" all the web app needs there, with attribution reserved for the entries
  that are actually off?

The last one is probably the cheapest big win: it bounds the expensive representation to the small
set of files that are not fine. Decide before building phase 3.

### 7.9 Deferred to a later version

Country-specific variant override detection (D5), web app verification (D8), and report history
retention/pruning.

## 8. Decisions taken

Recorded so they are not re-litigated. Taken by Giulia in review, 2026-09-18 (D1–D9, D11, D12) and
2026-09-21 (D10).

| # | Decision |
|---|---|
| D1 | This spec covers the whole suite, phased: existing `snt_workspace_manager`, new checker, web app deferred. |
| D2 | Check and fix are **separate pipelines**. The checker is read-only and schedulable. |
| D3 | Report is a workspace file, timestamped plus a stable `status_latest.json`. No dataset, no DB table. |
| D4 | Release ordering uses GitHub `published_at`. |
| D5 | Country-specific variant override detection is **out of scope for v1**. |
| D6 | Untracked files are **reported** in their own bucket, never acted on. |
| D7 | Both sources — filesystem and pipeline version zips — are read from v1. |
| D8 | Web apps are out of scope for v1, and named as a blind spot in the report. |
| D9 | Notebook hash normalisation: **decide after measuring** real drift in phase 2, not up front. |
| D10 | The manifest gap is closed **before** the checker is built (phase 0). |
| D11 | Two modes, driven by whether a target release is given: attribution (factual inventory) and verification (qualitative verdicts). |
| D12 | The status formerly called `mismatch_unknown` is renamed **`unknown_content`**, to stop it reading as a synonym of `untracked`: one is about the bytes at a known path, the other about a path we never shipped (§5.1.1). |

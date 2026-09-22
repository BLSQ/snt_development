# SNT Release Management — history, closed issues and lessons learned

> **Read this before re-investigating anything in `docs/wip/`.**
>
> Nothing in this file describes the current state of the system. It is the opposite: it is where
> superseded designs, closed problems, deleted fixtures and dead ends are kept, so that a reader —
> human or agent — does not rediscover them, re-litigate a settled decision, or mistake an old
> artefact for a live one.
>
> The current state lives in:
>
> * [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) — what the checker must do (requirements).
> * [`release_strategy.md`](release_strategy.md) — why the release mechanism exists and what is built.
> * [`pipeline_deployment_mechanism.md`](pipeline_deployment_mechanism.md) — how a pipeline is
>   deployed through the OpenHEXA API.
>
> Rule of thumb when editing any of those three: if a paragraph explains what something *used to be*,
> or records a verification against something that no longer exists, it belongs here instead.

**Contents**

1. [Dead ends — things that did not work](#1-dead-ends--things-that-did-not-work)
2. [Closed issues](#2-closed-issues)
3. [Superseded artefacts](#3-superseded-artefacts)
4. [Retired sandbox state and stale verification records](#4-retired-sandbox-state-and-stale-verification-records)
5. [Timeline](#5-timeline)

---

## 1. Dead ends — things that did not work

Kept so nobody repeats the troubleshooting. Each of these cost real time.

### `.gitignore` silently swallows new workflow files

The repo has blanket `*.yml` **and** `*.yaml` ignore rules, grouped with the data-export rules. A
new workflow file does not even show as untracked — `git add` just does nothing. The existing
`push_snt_*.yaml` files are tracked only because they predate the rule, and `git add -f` is
forbidden by this repo's rules (**R1**).

Fixed with a negation mirroring the existing `!configuration/SNT_config_*.json` pattern:

```
# GitHub Actions workflows ------------------------
!.github/workflows/*.yaml
```

Two consequences worth remembering: use `.yaml`, never `.yml` (no negation exists for `.yml`), and
the fix is repo-wide — it also unblocks any future `push_<name>.yaml`.

### Per-file GitHub API fetches hit the rate limit

Unauthenticated GitHub API calls are capped at **60/hour**, and pulling a release file-by-file via
the Contents API needs ~106 requests for a single release. Fix: download the release **source
tarball** — one request — and extract.

This limit is not only a historical annoyance: it is the live constraint behind the open decision
on how attribution mode obtains every release's manifest (`PRODUCT_SPEC.md` §7.2).

### Copying `pipeline.py` into the workspace filesystem does nothing

This looked like a completed deployment and was not. OpenHEXA runs each pipeline from its registered
version's stored zip, never from `workspace/files/`. A `pipeline.py` sitting on the filesystem is
inert while looking authoritative — **the most misleading failure mode encountered in this work**,
and the reason the checker hashes pipeline version zips as a separate source. Fix: deploy through
the API, and deliberately do *not* leave a filesystem copy.

### A pipeline run's own `HEXA_TOKEN` cannot deploy pipelines

The API answers `PERMISSION_DENIED`. Same payload, same code, a workspace API token read from the
`oh` CUSTOM connection → accepted. Only the header differs, and both tokens are 95 characters, so
they are indistinguishable by shape. **If a deployment call 403s, check which token is in the header
before anything else.**

---

## 2. Closed issues

### 2.1 The manifest under-described what is deployed — closed 2026-09-21

*Was: `PRODUCT_SPEC.md` §7.1, blocking phase 1. Fixed as phase 0 (decision D10).*

**The problem.** The manifest's `*/pipeline.py` pattern tracked one file per pipeline, but
deployment zips the **whole pipeline directory**. `snt_map_extracts` was deployed with `utils.py`,
`worldpopclient.py`, the `malariaAtlasProject/` package, `readme.md` and `requirements.txt` — none of
them in the manifest, none verifiable, and a change to any of them altered no manifest hash.
`requirements.txt` was the sharpest case: it carries the two unpinned Git dependencies the repo
already worries about, ships in the zip, and was invisible to the manifest. Verifying against a
manifest that describes a third of what is deployed gives false assurance, which is worse than no
verification.

**The fix that was rejected: a wider glob list.** The obvious patch was to add patterns:

```python
patterns = [
    ..., '*/requirements.txt', '*/readme.md', '*/**/*.py',
]
```

This was not done, and should not be revived. A glob list restates the SDK's rule in a second,
drifting dialect — one pattern per suffix *per depth* — and the list above is already incomplete: it
has no `.sql` pattern at all, and its `*/requirements.txt` and `*/readme.md` only reach the pipeline
root, so an equivalent file one directory down (inside `malariaAtlasProject/`, say) would still ship
unverified. `*/**/*.py` additionally sweeps up unrelated top-level directories (`dev/`,
`deprecated/`). Every future nesting or suffix would need another line nobody remembers to add.

**The fix that was applied** — reimplementing the SDK's own selection rule, anchored on
`*/pipeline.py` — is current design and is described in
[`release_strategy.md`](release_strategy.md) § "Manifest generation".

### 2.2 Widening the manifest broke an existing consumer

Recorded because it is the general shape of the risk here: **the manifest is an interface, and
widening it changes the behaviour of everything that reads it.**

`split_manifest()` in `snt_workspace_manager` classified every entry that was not
`<name>/pipeline.py` as an analytics file and **copied it into the workspace bucket**. Under the
widened manifest that meant 49 `readme.md` / `requirements.txt` / helper-module files strewn across
the workspace filesystem, where OpenHEXA never reads them — the precise "inert and misleading copy"
failure this whole effort exists to detect (§1 above).

Fixed in the same change: the split is now by *directory*, taking the directory list from the
manifest's `pipelines` block where present, and falling back to the old `<name>/pipeline.py`
derivation for pre-phase-0 manifests. Checked against the `v0.0.1-test` and `v0.0.2-test` manifests
and the new one: **86 analytics files in all three**, so old releases deployed exactly as before.

Both of those releases were deleted in the sandbox reset (§4), so that verification stands as a
record but is no longer repeatable, and **no live release exercises the fallback path**. That leaves
one open item, carried in `PRODUCT_SPEC.md` §2.1: keep a legacy manifest as a local test fixture, or
delete the fallback in a PR of its own. Untested back-compat code for a case that can no longer
occur is worse than either.

### 2.3 `snt_workspace_manager` existed only in the sandbox workspace

It was built as a pipeline version inside `snt-development-sandbox` with no copy in the repository —
exactly the "no version-propagation story" problem this project exists to fix. Committed in
`5cb7995`, widened in `16bd149`; it now lives at [`snt_workspace_manager/`](../../snt_workspace_manager/)
with `pipeline.py`, `requirements.txt` and `readme.md`. No `push_*.yaml` workflow yet, pending the
R5 wording question (`pipeline_deployment_mechanism.md`).

---

## 3. Superseded artefacts

### 3.1 The pre-phase-0 manifest generator

The original `generate_manifest.yaml` embedded this script. It tracked 106–107 files. It is
**superseded** by the committed
[`.github/workflows/generate_manifest.yaml`](../../.github/workflows/generate_manifest.yaml), which
tracks 156 and emits a `pipelines` block. Kept only so a manifest found in the wild can be dated:
a manifest with no `pipelines` key and ~106 entries came from this.

```python
def main():
    version = os.environ.get('GITHUB_REF_NAME', 'unknown')
    manifest = {"version": version, "files": {}}

    # Directories to track based on our strategy
    patterns = [
        'pipelines/**/code/*.ipynb',
        'pipelines/**/reporting/*.ipynb',
        'pipelines/**/utils/*.r',
        'code/**/*.r',
        '*/pipeline.py',
    ]

    tracked_files = []
    for pattern in patterns:
        tracked_files.extend(glob.glob(pattern, recursive=True))
    tracked_files = list(set(tracked_files))  # Deduplicate

    for fpath in tracked_files:
        manifest["files"][fpath.replace('\\', '/')] = hash_file(fpath)

    with open('release_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
```

Verified 2026-09-16 against `v0.0.1-test`: `version` read the tag correctly (not `"unknown"`), 106
files tracked, every hash a valid 64-char sha256, all 20 `pipeline.py` files matched a local `find`
excluding `deprecated/`.

### 3.2 The original product-spec draft — 2026-09-18

`product_spec_draft.md` was the user's free-form statement of intent. It was consolidated into
[`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) the same day and the file removed. Its substance is reproduced
here because it is the only record of what was asked for before the requirements were formalised —
useful if a requirement in the spec ever looks arbitrary.

> Final product vision: an OH webapp that allows the user to install a specific release of the SNT
> Stratification suite (SNT pipelines collection + associated webapps), and/or to check what is the
> status of the workspace content relative to a specific release (check that all expected files are
> there and if they are of the correct version for a given release, else flag anything that is off)
> and then decide to either fix things (if files are missing or are of the wrong version: install and
> update to match a target release) or leave as is (it is possible that some changes were made
> manually and the user wants to keep them).
>
> This webapp is built as a nice UI/UX layer on top of an OH pipeline. This is because the pipeline
> is needed to do things that are a bit too much for a web app: namely pulling files (saving to OH ws
> file system) and deploying pipelines in OH, from the GitHub repo. So for now I want to focus on the
> pipeline.
>
> What I want the pipeline to do: check the status of the workspace — look at all files present in
> the workspace and extract their SHA; compare against the `release_manifest.json` for each release
> tag of the reference GitHub repo, and derive which release each file belongs to. Options: belongs
> to target release = correct; does not = "behind" (older), "ahead" (newer) or "unknown" (belongs to
> no release, probably edited manually or corrupted); plus files that are missing — defined in the
> manifest but absent from the file system or the pipelines database. It could output a summary file
> with the status of all relevant files, conceived to be readable by the future webapp.
>
> The GitHub repo will be public. The user eventually will not be able to choose the repo (so it
> should be hard coded, so we are in control), but for initial stages let's make it a parameter for
> ease of testing. The user should be allowed to choose the release version though, for
> reproducibility (a country may run an analysis now and in a year want the exact same analysis with
> newer data). The pipeline should be very verbose, explicit and clear, to avoid any "blackbox"
> feeling. Older or stale files should never be deleted, but moved to an "archive" location findable
> by the user.

The draft's four open questions, and where each landed:

| Draft question | Resolution |
|---|---|
| One pipeline for check + fix, or two? | Two — decision **D1**/**D2**, `PRODUCT_SPEC.md` §8. |
| A mechanism to import every release's manifest | Still open — `PRODUCT_SPEC.md` §7.2, blocks phase 3. |
| What to do with files not in the manifest ("ignore"?) | Not ignored: **reported** in an `untracked` bucket, never acted on — decision **D6**. |
| Different releases having different file lists | Covered by the `missing` / `removed_in_target` statuses, `PRODUCT_SPEC.md` §5.1. |
| Test repo needs a `latest` release | **Rejected.** GitHub's `/releases/latest` endpoint already resolves to the newest non-prerelease release; a release *named* `latest` would collide with it. |

---

## 4. Retired sandbox state and stale verification records

### 4.1 Sandbox reset — 2026-09-21

The sandbox repository had accumulated two branches, two manifest generations and a fixture set
built in stages; disentangling it was worth less than restarting. `BLSQ/snt_development_sandbox` was
**deleted and recreated** under the same name, then seeded with a single parentless commit carrying
the tree of `snt_development` @ `551ddd8` — minus the 20 `push_snt_*.yaml` deployment workflows,
which target the **real** `snt-development` workspace via `secrets.OH_TOKEN` and must never fire
from a sandbox.

Deleting the repository also deleted its tag-protection ruleset, which had to be recreated. That is
also the reason a sandbox cannot be cleaned up *in place*: *Restrict deletions* with an empty bypass
list stops an admin deleting the very tags they want gone.

The OpenHEXA workspace `snt-development-sandbox` was **not** reset. It still holds pipelines and a
`.snt_release` marker naming a tag that no longer exists — harmless, and itself a usable test of how
the checker handles an unresolvable declared release.

Procedure: `ignore/SNT25-670/sandbox_reset_runbook.md` (local, not committed).

### 4.2 `v0.0.1-test` and `v0.0.2-test` are retired names

Both releases were deleted in the reset. **The numbers are deliberately not reused**: they are
attached in writing — in this file and in the git history — to a 106-file legacy manifest, and
reusing them would make a tag mean two things, which is the exact failure the tag-protection
convention exists to prevent. The fixture series restarts at `v0.1.0-test`.

Verifications performed against those tags, which stand as a record but **cannot be re-run**:

| What was verified | When | Result |
|---|---|---|
| Manifest generation (legacy generator) against `v0.0.1-test` | 2026-09-16 | 106 files, tag read correctly, all hashes valid |
| `snt_workspace_manager` full run against `v0.0.1-test` | 2026-09-16 | 60s; `.snt_release`, the three shared `code/*.r` files and pipeline code all confirmed |
| API deployment of `snt_dhis2_extract` + `snt_map_extracts` from `v0.0.1-test` | 2026-09-16 | Correct codes; parameters round-tripped through `Parameter.to_dict()` including the `dhis2_connection` connection-typed parameter; deployed `pipeline.py` read back at sha256 `44290bd9…d77755cd`, byte-identical to the manifest hash |
| `split_manifest()` old and new paths against both legacy manifests | 2026-09-21 | 86 analytics files in both, unchanged (§2.2) |

`backup_existing` was **never** exercised by any of these — the verified run was against an empty
workspace, so there was nothing to archive. That gap is still open.

---

## 5. Timeline

| Date | Event |
|---|---|
| 2026-09-16 | Legacy manifest generator verified; `snt_workspace_manager` v3 proven end to end against `v0.0.1-test`; API deployment mechanism written up. |
| 2026-09-18 | `product_spec_draft.md` reviewed with Giulia; decisions D1–D9, D11, D12 taken; `PRODUCT_SPEC.md` written. §7.2 (obtaining every manifest) deferred to a dedicated session. Tag-protection ruleset created on the sandbox. |
| 2026-09-21 | Phase 0: manifest generator rewritten to mirror the SDK's zip rule (107 → 156 files) and given a `pipelines` block; `split_manifest()` fixed in the same change. Sandbox repo reset; fixture releases `v0.1.0-test` … `v0.4.0-test` cut. |
| 2026-09-22 | `docs/wip/` split: current state in the three live documents, history consolidated here. |

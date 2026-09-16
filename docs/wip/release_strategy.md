# SNT Stratification Release Strategy & Implementation Plan

## The Core Challenge

Currently, OpenHEXA distributes Python orchestration code (`pipeline.py`) via its native Template
system, but the R analytics and helpers live in the workspace filesystem. This leads to version
drift, as different parts of the system travel by different routes.

*(Note: There are also two OpenHEXA web apps in the ecosystem. Deployment strategies for these web
apps are currently deferred and remain an open question for the OpenHEXA developers. We will focus
solely on the Python/R codebase for now.)*

## The Roadmap (5 phases)

This is the plan end to end. Each phase's design detail lives in the sections below; the
**Sandbox Setup** section is where we're currently validating phase 2 in isolation before touching
the real repo.

1. **Sandbox environment.** An independent copy of this repo to experiment in safely — deliberately
   *not* a GitHub fork (see [Why not a fork?](#why-not-a-fork) below).
2. **Cut a release + generate a manifest.** A GitHub Release on the sandbox triggers a GitHub
   Action that hashes the tracked files and publishes a `release_manifest.json` (file path → sha256)
   alongside the release.
3. **Pull mechanism for R/notebook files.** An OpenHEXA pipeline ("Workspace Manager") that fetches
   the R/notebook half of a given release into the workspace filesystem.
4. **Pull mechanism for `pipeline.py` files.** The same Workspace Manager also fetches `pipeline.py`
   for every pipeline from the release — see [why this replaces the Template
   mechanism](#opting-out-of-openhexas-template-auto-update).
5. **Verification pipeline.** An OpenHEXA pipeline that hashes what's actually in the workspace
   filesystem, compares it against a release manifest, and reports which release (or "modified" /
   "unknown") each tracked file currently matches.

---

## The Strategy: "Code-Only Delivery" via Workspace Manager

Shift away from individual pipelines updating themselves (deprecating the `Pull scripts` toggle)
and introduce a centralized **Workspace Manager** pipeline (or integrate this into the new
Orchestrator Web App later).

This manager is backed by a **Release Manifest** generated via GitHub Actions whenever a new
release is cut. The source of truth for versions (e.g., `v1.2.0`) will be the GitHub Release Tag.

### 1. What the Workspace Manager actually syncs

The deployment mechanism drops the **entire codebase** — both the R "engine" and the Python
orchestration — into the workspace, all pinned to the same release tag. It manages:

* `pipelines/**/code/*.ipynb`
* `pipelines/**/reporting/*.ipynb`
* `pipelines/**/utils/*.r`
* `code/**/*.r`
* `**/pipeline.py` (see below — this is new relative to the original plan)

It explicitly **ignores**:
* `data/` directory (bootstrapped by pipelines).
* `configuration/` directory (created/managed by the Config Editor web app).
* Web app deployments (for now).

#### Opting out of OpenHEXA's Template auto-update

Today, `pipeline.py` travels a completely different route from everything else: CI pushes it to
the `snt-development` workspace on every merge to `main`, which publishes a new version of the
OpenHEXA **Template**, and every country workspace subscribed to that template updates
automatically — independent of, and usually on a different cadence than, whenever an operator
next runs a pipeline with `Pull scripts` = ON to fetch the R side. **That mismatch is the root
cause of the drift** described in "The Core Challenge" above.

This plan deliberately removes that second, uncoordinated update path. Once the Workspace Manager
also pulls `pipeline.py`, country workspaces stop being subscribed to OpenHEXA's native Template
auto-update entirely. Python and R can then only ever move together, on the same release tag,
through the same manifest-driven mechanism. This is a full replacement, not an additional parallel
path — a workspace should not be receiving `pipeline.py` from both the Template system and the
Workspace Manager at once.

### 2. The Workflow

* **Initialize (Empty Workspace):**
  The Workspace Manager pulls the specified release's full codebase (R + Python) into the
  workspace. The user can then proceed with configuring the workspace to unblock the rest of the
  pipelines.
* **Diff / Check (Existing Workspace):**
  The script hashes the `pipelines/`, `code/` folders and every `pipeline.py`, comparing them
  against the release manifest. It flags manually edited files while safely ignoring data and
  configs. It explicitly ignores but flags country-specific notebook variants (e.g.,
  `snt_seasonality_rainfall_NER.ipynb`) as deliberate overrides.
* **Update / Downgrade:**
  The script backs up any modified files into a `~/workspace/archive/` folder, then overwrites the
  core analytics and orchestration folders with the target release.

### 3. Discussion Points for Technical Team & OH Devs

1. **Web App Deployment Context:** How can we programmatically deploy OH webapps by pulling code
   from GitHub repositories? Currently, it's done manually via the OH webapp creating UI or via
   agent. Can we pull webapp code directly into the workspace and trigger an update via API?
2. **Retiring both auto-update paths:** Deprecate the `Pull scripts` boolean in individual R-driven
   pipelines, *and* stop pushing `pipeline.py` to the OpenHEXA Template system from CI (per this
   repo's current `R5` / "Always publish from `snt-development`" convention) — the Workspace
   Manager becomes the single update path for the whole analytical suite, Python included.
3. **Version Number Alignment:** The GitHub Release Tag (e.g., `v2.0.0`) becomes the single source
   of truth. The Workspace Manager writes `{ "snt_release": "v2.0.0" }` into a hidden file in the
   workspace root.

---

## Sandbox Setup

Steps to validate phase 2 (release + manifest generation) in isolation, without touching the real
`BLSQ/snt_development` repo or its CI.

### Why not a fork?

A GitHub **fork** exists to let outside contributors send pull requests back to the upstream repo,
and it carries two behaviors we don't need here:

* GitHub disables Actions by default on a freshly-forked repo — the manifest workflow would
  silently never run until that's flipped on once in the fork's Settings → Actions.
* A fork keeps an upstream-comparison link (ahead/behind banner) that's only useful if we intend
  to PR changes back — we don't; this is a disposable sandbox.

**Confirmed via the GitHub API:** `BLSQ/snt_development_test-release` came back with `"fork":
true` — it really was a GitHub fork. GitHub has no "detach from fork network" button, so we
created a genuinely independent repo instead: `BLSQ/snt_development_sandbox`, created empty
through the GitHub UI (not the Fork button). The local remote is named `sandbox` (renamed from
the earlier `testfork` placeholder). The old fork is left as-is on GitHub for now — to be deleted
by hand later (Settings → General → Danger Zone), not by an agent.

### Step 1: Create the sandbox repo — done

Created `https://github.com/BLSQ/snt_development_sandbox` empty via the GitHub UI (no README/
`.gitignore`/license, so pushing existing history into it doesn't conflict). Since it's a plain
new repo, not a fork, Actions are enabled by default — no toggle needed.

### Step 2: Link the sandbox repo to your local clone — done

```bash
# Repoint the existing "testfork" remote at the new, genuinely independent repo
git remote rename testfork sandbox
git remote set-url sandbox https://github.com/BLSQ/snt_development_sandbox.git

# Branch for the experiment, cut from main (not from SNT25-670, which carries unrelated work)
git checkout -b feature/release-manifest-test main

# Push — since the repo is empty, this also becomes its default branch
git push sandbox feature/release-manifest-test
```

### Step 3: Create the GitHub Action YAML — done, with two gitignore gotchas found

1. Create a file at `.github/workflows/generate_manifest.yaml` — **note `.yaml`, not `.yml`**.
   `.gitignore` has a blanket `*.yml` rule (grouped with the data-export rules like `*.csv`), so a
   `.yml` file is silently swallowed on `git add` and never even shows as untracked. Every existing
   workflow in this repo already uses `.yaml` — following that convention avoids this trap.
2. `.gitignore` also has a blanket `*.yaml` rule with **no exception for `.github/workflows/`**.
   The existing `push_snt_*.yaml` files are tracked only because they were committed before that
   rule existed — git keeps tracking already-tracked files regardless of later ignore rules. Any
   *new* workflow file (ours, or any future `push_<name>.yaml` per the "Adding or changing a
   pipeline" checklist) hits the same silent-ignore trap, and this repo's rules forbid `git add
   -f` as the workaround. Fixed by adding a negation to `.gitignore`, mirroring the existing
   `!configuration/SNT_config_*.json` pattern:
   ```
   # GitHub Actions workflows ------------------------
   !.github/workflows/*.yaml
   ```
   This is a repo-wide fix, not sandbox-only — worth calling out in the eventual PR description
   since it silently unblocks something unrelated to the release-strategy work itself.
3. Paste the following YAML into it:

```yaml
name: Generate Release Manifest

on:
  release:
    types: [published]
  workflow_dispatch: # Allows manual triggering from the UI without creating a release

jobs:
  generate-manifest:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          ref: ${{ github.event.release.tag_name || github.ref }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Generate Manifest Script
        run: |
          cat << 'EOF' > generate_manifest.py
          import os
          import json
          import hashlib
          import glob

          def hash_file(filepath):
              hasher = hashlib.sha256()
              with open(filepath, 'rb') as f:
                  while chunk := f.read(8192):
                      hasher.update(chunk)
              return hasher.hexdigest()

          def main():
              version = os.environ.get('GITHUB_REF_NAME', 'unknown')
              manifest = {
                  "version": version,
                  "files": {}
              }

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

              tracked_files = list(set(tracked_files)) # Deduplicate

              for fpath in tracked_files:
                  standard_path = fpath.replace('\\', '/')
                  manifest["files"][standard_path] = hash_file(fpath)

              with open('release_manifest.json', 'w') as f:
                  json.dump(manifest, f, indent=2)

              print(f"Generated manifest with {len(manifest['files'])} files.")

          if __name__ == '__main__':
              main()
          EOF

      - name: Run Script
        run: python generate_manifest.py

      - name: Upload Manifest to Release (If published release)
        if: github.event_name == 'release'
        uses: softprops/action-gh-release@v1
        with:
          files: release_manifest.json
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Upload Artifact (If triggered manually)
        if: github.event_name == 'workflow_dispatch'
        uses: actions/upload-artifact@v4
        with:
          name: release-manifest
          path: release_manifest.json
```

> Note: the `patterns` list above now also includes `*/pipeline.py`, matching the
> [phase 4 decision](#opting-out-of-openhexas-template-auto-update) to track Python orchestration
> files in the same manifest as the R engine.

### Step 4: Commit and push to the sandbox

```bash
# 1. Stage the files (note: docs/wip/release_strategy.md is part of the real SNT25-670 ticket
#    work and lives on that branch too — the .gitignore fix and the workflow file are sandbox-
#    only until they're deliberately ported to a real PR against origin/main)
git add .gitignore .github/workflows/generate_manifest.yaml docs/wip/release_strategy.md

# 2. Commit them
git commit -m "Add release strategy and manifest generation workflow"

# 3. Push this branch specifically to the "sandbox" remote — never to origin
git push sandbox feature/release-manifest-test
```

### Step 5: Cut a test release and verify the manifest

1. On `https://github.com/BLSQ/snt_development_sandbox`, create a GitHub Release from the
   `feature/release-manifest-test` branch (e.g. tag `v0.0.1-test`).
2. Confirm the Action runs (Actions tab) and that `release_manifest.json` is attached to the
   release.
3. Spot-check the manifest content: does it list a `pipeline.py` per pipeline, and the expected
   `pipelines/**/code/*.ipynb` / `code/**/*.r` files, with plausible sha256 hashes?

This closes the loop on phase 2 of the roadmap. Phases 3–5 (the OpenHEXA-side pull and
verification pipelines) are not yet started.

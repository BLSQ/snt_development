# MIGRATION PLAN — moving SNT stratification into one public repo, `snt-stratification`

> **This is a plan, not a description of the current system.** Everything else in `docs/` states how
> things *are*; this file states what we *intend* to do and how far we have got. Do not cite it as
> fact about the running pipelines.
>
> **Living document.** It is public so the whole team can follow the migration. Update the checkboxes,
> the Decision log and the Session log at the end of every working session. Status markers:
> `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped.
> New repo: **`BLSQ/snt-stratification`** (D-6). Owner: Giulia.
>
> **Where it lives:** `docs/MIGRATION_PLAN.md` in `snt_development` until the new repo exists. In
> Phase 1 it moves to the new repo (same path), and this file becomes a one-line pointer to it.
> **For agents:** read it only when working on the migration. It is not part of the routine reading
> path.
>
> **Working rule:** project knowledge lives **in the repo** (CLAUDE.md, docs, ADRs, BACKLOG), never
> only in someone's local Claude memory.

---

## 1. Context

Today, "SNT stratification on OpenHEXA" lives in 4–5 repos that are tightly coupled but versioned
and documented separately:

| Source repo | What | Maturity |
|---|---|---|
| `BLSQ/snt_development` | ~20 SNT pipelines (`pipeline.py` at the root + R notebooks under `pipelines/<name>/`), shared R `code/`, `catalogs/`, `contracts/`, the release-manifest workflow | Docs: high (Register R1–R22, ADRs 0001–0009, BACKLOG, 3 skills). Tests: none |
| `BLSQ/snt_webapp_orchestrator` | Orchestrator static webapp + the `create_pipeline_cards` pipeline | Docs: high (ADRs 0001–0016, R1–R12, BACKLOG). Already a fresh repo built from a snapshot (ADR 0013) |
| `BLSQ/snt_workspace_bootstrap` | `snt_workspace_deployer`, `snt_pipeline_deployer`, `snt_workspace_checker` | Docs: high (D1–D28, contracts, roadmap). pytest exists. No CI |
| `BLSQ/openhexa-webapps-edit-snt-config-json` | Config-editor webapp (ticket SNT25-661) | **Spec stage only**: `docs/PRODUCT_SPEC.md` + schema analysis docs + a throwaway `prototype/index.html`; `app/` is empty. Its own copy of `SNT_config.skeleton.json`. Its spec §12 lists every upstream dependency |
| `BLSQ/snt_utils` (`snt_lib`) | Python helpers: config loading, `pull_scripts_from_repository`, `save_pipeline_parameters` | **External and pinned in the short term** (managed by a colleague); **moves into the monorepo in the long term** (D-3, §10) |

**Problems this solves:**
- Every cross-repo contract (release manifest R22, pipeline ids ↔ orchestrator map, config skeleton
  ↔ editor) needs paired PRs, and the docs and rules (R17/R18/R19 are copied three times) drift apart.
- The release mechanism (tag → `release_manifest.json` → deployer) is designed for one source repo.
  The multi-source roadmap (bootstrap `roadmap.md` §1) disappears once there is one repo.
- A clean start lets us make breaking changes held back to protect live workspaces (config split,
  dataset-key naming, config as a dataset, report-only semantics, retiring Pull scripts).

**Hard constraints:**
- Existing country workspaces keep working **unchanged**: the old repo, the old template CI and the
  old templates published from `snt-development` all stay in place (D-4).
- The new repo starts as a clean snapshot without history. The old repos stay as backups and are
  archived later (orchestrator ADR 0013 is the precedent).
- Public repo: no country data, no secrets.

**Success:**
- A fresh test workspace can be bootstrapped end to end from a GitHub release of the new repo:
  pipelines, R half, webapps, with a version marker, and the checker reports `match`.
- A second fresh workspace can be set up from the **new template set**. Its pipeline codes equal
  `snt-<name>`, so the deployer and checker can take it over later.
- One CLAUDE.md router, one Register, one BACKLOG and one ADR sequence for everything.

---

## 2. Decision log (agreed)

| ID | Decision | Date |
|---|---|---|
| D-1 | One new public monorepo, built as a clean snapshot (no history). Old repos stay as backups and are archived later | 2026-11-02 |
| D-2 | **One release train.** One tag `vX.Y.Z` = one tested combination of pipelines + webapps + bootstrap. Webapps get a `webapps` section in the manifest and a `version.json` stamped at release time. A webapp-only fix = a patch release (the deployer skips unchanged pipelines, D22) | 2026-11-02 |
| D-3 | **Short term:** `snt_lib` (`BLSQ/snt_utils`, managed by a colleague) stays **external, pinned to a real tag** in every `requirements.txt` (closes DEP-1 for it); changes to it are coordinated cross-repo PRs. **Long term:** it moves into `snt-stratification` as `packages/snt_lib/`, with a companion R package next to it (§10) | 2026-11-02 |
| D-4 | Old `snt_development` goes **maintenance-only**: critical fixes for live countries only, ported by hand; all new work happens in the new repo. Archive once no workspace depends on its template CI or Pull scripts | 2026-11-02 |
| D-5 | Co-locate each pipeline's R half with its `pipeline.py` **only if spike S1 passes**. Otherwise use the fallback layout (§4.3). For S2, the preferred outcome is **self-sync** (option b). **Condition met 2026-11-03:** S1 and S2(b) both passed → co-locate + self-sync | 2026-11-02 (confirmed 2026-11-03) |
| D-6 | Repo name **`snt-stratification`** (dash, matching most BLSQ repos). The conda env follows the same name | 2026-11-02 |
| D-7 | The config editor **is included** from the first snapshot, as `webapps/config_editor/`. Its throwaway prototype goes to `webapps/config_editor/design/`; its `ignore/` (NER draft configs) is **not** carried over | 2026-11-02 |
| D-8 | Tag numbering starts from scratch (e.g. `v0.1.0`; first production release `v1.0.0`). The old repo's tags are tests only, so no live workspace's marker can collide | 2026-11-02 |
| D-9 | **Repo path = workspace path** for every file that lands on the workspace filesystem. No mapping layer. The shared R library is renamed `code/` → **`shared/r/`**, in the repo and in the workspace (§4.2) | 2026-11-02 |
| D-10 | **Docs placement rule:** a unit's own docs live with the unit (`webapps/<name>/{docs,design}/`, beside the deployable `app/`; a pipeline's `readme.md`); docs spanning several units live in `docs/`; contract prose lives in `contracts/<name>/` (§4.1) | 2026-11-02 |
| D-11 | **One config skeleton:** the `snt_development` copy becomes the single copy (`contracts/snt_config/`), and every empty single value is written as **`null`**, the pipelines' convention. The editor's `""` for empty strings changes to `null` | 2026-11-03 |
| D-12 | **Report-only fix** (was OD-7): the orchestrator reads the run's own `config.run_report_only` through the proxy (`PIPELINES_READ`, proven by S4) and counts only full runs as "processed". No `RUN_MODE` in `{CC}_parameters.json` | 2026-11-03 |
| D-13 | **Public-import scope:** Jira links and Bluesquare dev-workspace slugs (`snt-development`, `snt-development-sandbox`, `snt-testing`, `snt-app-dev`) are **kept**. **Country-specific notebooks (`*_<CC>.ipynb`, plus the un-suffixed `NER_pyramid_format.ipynb`) are left out.** They stay findable in the old `snt_development` repo at the commit SHA in the provenance table (it stays readable after archiving), and in the country workspaces themselves. How they are versioned in the future is OD-10 | 2026-11-03 |

## 3. Open decisions (owner: Giulia unless stated)

**When do they need an answer?** Only when the phase in the "Blocks" column starts. None of them
blocks Phase 0 or Phase 1. Settle each one at the start of the session that begins the phase it
blocks, and move it to the Decision log.

| ID | Question | Recommendation / current leaning | Blocks |
|---|---|---|---|
| OD-4 | How do the bootstrap pipelines (deployer, checker, single-pipeline deployer) reach a workspace? | **Leaning (Giulia):** as regular OH **template pipelines**, kept in this repo as `kind: bootstrap` folders in `pipelines/`, with their **own CI** that pushes every merge to `main` into the new templates workspace (not tied to the release train). Implications to keep in mind: (a) a deployer on `main` must stay able to read every manifest schema version it may be asked to deploy, or refuse old ones with a clear `[ERROR]`; (b) they are then **outside** the release manifest's `pipelines` (or listed with `kind: bootstrap` for provenance only, never deployed by the deployer); (c) this supersedes bootstrap D19 ("no deploy workflow") and solves roadmap §3 (chicken-and-egg). Not decided | Phase 3 (CI) / Phase 4 |
| OD-5 | Import ADRs: one renumbered sequence, or per-origin folders? | One sequence with an "Originally: `<repo>` ADR NNNN / D<n>" line and a mapping table in `decisions/README.md`. Bootstrap D-entries that are only implementation detail stay in `docs/bootstrap/decisions.md` | Phase 2 |
| OD-6 | Testing releases: keep a sandbox repo, or use pre-releases? | Pre-release tags (`v0.x.y`, marked pre-release) on the new repo + a test workspace. Drop `snt_development_sandbox`. The deployer already ignores pre-releases unless asked (D27) | Phase 4 |
| OD-8 | CODEOWNERS (ADR 0005 said no) | Revisit: three areas with different reviewers is the case where it pays off | Phase 2 |
| OD-9 | Retiring Pull scripts | **S2b passed (2026-11-03):** self-sync works, so Pull scripts can go for both routes. Still open: the S2b sub-questions (§4.3), in particular the per-run escape hatch | Phase 6 |
| OD-10 | **How to version country-specific notebooks, and analyst-edited ones** (raised 2026-11-03). Today nothing versions `_<CC>` variants, and no route delivers them to their country. Analysts also edit notebooks in their workspace, and updates (templates, deployer, self-sync) must not silently overwrite those edits. We need one system that keeps good practice (versioned, reviewable, no country data in the public repo) and still lets users run what they edited | No leaning yet. Constraints so far: country variants are not in the public repo (D-13); self-sync skips `_<CC>` files and archives what it replaces (S2b). Inputs: OD-9's escape hatch, and possibly a private per-country repo or dataset | None (not scheduled). Must be settled before Pull scripts is retired (OD-9) |

---

## 4. Target structure of the new repo

### 4.1 Tree

```
snt-stratification/
├── CLAUDE.md                 ← router: guardrails, Register (merged), Traps, "Where to find things". ≤15 KB
├── README.md                 ← human overview + newcomer path
├── LICENSE                   ← confirm licence (DOC-5)
├── pyproject.toml            ← one ruff + pytest config for all Python (line-length 110, numpydoc)
├── .gitignore                ← allowlist for JSON (see Traps §7)
├── .claude/
│   ├── settings.json         ← R19 hook + permissions.deny (merged from the 3 repos)
│   ├── hooks/                ← block-destructive-git (one implementation)
│   ├── README.md             ← how the guardrail works
│   └── skills/               ← handover, write-pipeline-readme, add-explorer-layer (+ new: release, deploy-webapp)
├── .github/
│   ├── workflows/
│   │   ├── ci.yaml                     ← PR: ruff, pytest, schema validation, link check (TOOL-2/3/5)
│   │   ├── release.yaml                ← on release: build manifest v2, slim release tarball, stamped webapp zips
│   │   ├── push_templates.yaml         ← SNT + support pipelines → NEW templates workspace; ONE matrix job (replaces 20 copies)
│   │   └── push_bootstrap.yaml         ← bootstrap pipelines → templates on every merge to main (OD-4 leaning)
│   └── pull_request_template.md
│
├── pipelines/                ← flat: one folder per pipeline, folder name = @pipeline id (ADR 0006 join key)
│   ├── CLAUDE.md             ← pipeline + notebook rules (lazy-loaded), R11/R12, country variants
│   ├── registry.json         ← name → kind (snt | support | bootstrap) + deploy route. CI fails on an
│   │                           unregistered folder or a registered folder that is missing
│   ├── snt_dhis2_extract/    ← kind: snt (all ~19 SNT pipelines alike)
│   │   ├── pipeline.py   requirements.txt   readme.md
│   │   ├── code/*.ipynb        ┐ co-located R half (D-5, if S1 passes).
│   │   ├── reporting/*.ipynb   │ SAME path in the workspace (D-9):
│   │   └── utils/*.r           ┘ <ws>/pipelines/snt_dhis2_extract/code/…  (= today's workspace layout)
│   ├── create_pipeline_cards/    ← kind: support (also snt_metadata_json_trim, future utility pipelines)
│   ├── snt_workspace_deployer/   ← kind: bootstrap (also snt_pipeline_deployer, snt_workspace_checker)
│   └── …
│
├── shared/
│   └── r/                    ← shared R library (today's code/*.r). SAME path in the workspace: <ws>/shared/r/
│
├── packages/                 ← LONG TERM only (§10), not created now: snt_lib (Python), companion R package
│
├── webapps/
│   ├── CLAUDE.md             ← webapp-wide rules (static bundle, no secrets, proxy scopes)
│   ├── orchestrator/
│   │   ├── CLAUDE.md         ← app-code rules (bundle, i18n, escaping)
│   │   ├── app/              ← THE deployable bundle; release.yaml zips only this folder (+ stamped version.json)
│   │   ├── docs/             ← PRODUCT_SPEC, DEPLOYMENT, research/ (if it passes the public audit)
│   │   └── design/           ← wireframes, grid editor, map preview (not deployed)
│   └── config_editor/
│       ├── CLAUDE.md         ← the editor's rules (from its current CLAUDE.md)
│       ├── app/              ← empty until the editor is built
│       ├── docs/             ← PRODUCT_SPEC, EXTERNAL_CHECKS
│       └── design/           ← the throwaway prototype (prototype/index.html)
│
├── contracts/                ← every cross-unit schema, one README index
│   ├── release_manifest.schema.json   (v2)   status_report.schema.json   snt_release_marker.md
│   ├── snt_config/           ← THE skeleton (one copy, ends the editor's vendored copy) + snt_config.schema.json (CFG-3)
│   │   │                       + the contract prose: config_analysis_notes.md, config_field_matrix.md (from the
│   │   │                       editor repo; they describe what pipelines AND the editor must agree on)
│   │   └── fixtures/         ← conformance cases (valid + one per blocking rule) run by BOTH the editor's JS
│   │                           validator and snt_lib's validate_config() — editor spec §12.2 option 2
│   └── orchestrator/         ← pipeline_map, descriptions, cards schemas
├── catalogs/                 ← universal references shipped per release: SNT_metadata_master.json,
│                               later snt_datasets.json (fixed dataset ids, CFG-1) and the other fixed config parts
│
├── docs/                     ← ONLY docs that span several units (see the placement rule below)
│   ├── ARCHITECTURE.md       ← NEW system map: units, contracts, deploy routes, data flow (1 page + links),
│   │                           incl. "where docs live" (the placement rule)
│   ├── DATA_ARCHITECTURE.md  CONVENTIONS.md  GLOSSARY.md  OWNERSHIP.md  PIPELINE_README_STANDARD.md
│   ├── RELEASES.md           ← release train, tag rules, manifest, how to cut a release, rollback
│   ├── OPENHEXA_WEBAPP_RUNTIME.md ← platform facts used by both webapps AND the bootstrap pipelines
│   ├── BACKLOG.md            ← ONE list, area-prefixed IDs (see §5 Phase 2)
│   ├── MIGRATION_PLAN.md     ← this plan, until the migration is done
│   ├── decisions/            ← ONE ADR sequence + mapping table of imported ADRs
│   └── bootstrap/            ← deployer.md, checker.md, decisions.md (D1–D28 detail): shared by the three
│                               bootstrap pipelines, so not one pipeline's readme
├── reference/openhexa/       ← schema.generated.graphql (used by webapps AND bootstrap; grep it, don't read it whole)
├── tools/                    ← build_manifest.py (from the yaml heredoc), check_links.py, notebook_drift.py
├── tests/                    ← pytest: contracts, manifest builder, deployer/checker helpers
├── dev/                      ← environment.yml (ONE conda env: snt-stratification), setup README
└── ignore/                   ← git-ignored: plans, scratch, spikes
```

**Where docs live (placement rule):** *a unit's own docs live with the unit; docs that span several
units live in `docs/`.*
- **Webapp:** `webapps/<name>/docs/` and `design/`, beside its `app/` bundle. Only `app/` is deployed,
  so docs can never leak into the bundle.
- **Pipeline:** its own doc is `readme.md` (the operator contract, deployed with the zip, ADR 0016).
  Pipelines get **no `docs/` subfolder**: the pipeline zip includes every `.md`, so longer design
  notes go in `docs/<area>/` when they cover several pipelines (as `docs/bootstrap/` does).
- **Contract:** the prose sits next to its schema in `contracts/<name>/`.
- **Several units, or the whole system:** `docs/`.
- No top-level `design/`. The design material of each webapp lives with that webapp.

**Not carried over:** `deprecated/` from every repo (links to it are pinned to the source commit,
like ADR 0013), empty `__pycache__`-only folders (`snt_workspace_check/`, `snt_workspace_manager/`),
and `snt_assemble_results`.

### 4.2 Repo path = workspace path (D-9)

The deployer already copies each file to its repo-relative path, and the checker's model is "workspace
path = repo path". We **keep that model** and choose the repo layout so it holds:

| Repo path | Workspace path | Change vs today's workspaces |
|---|---|---|
| `pipelines/<name>/{code,reporting,utils}/…` | `pipelines/<name>/{code,reporting,utils}/…` | none |
| `shared/r/*.r` | `shared/r/*.r` | **renamed** from `code/` — every notebook's `source(…/code/…)` changes once |
| `pipelines/<name>/` (zip) | pipeline code `<name>` with `_`→`-` | none (ADR 0006 join key kept) |
| `webapps/<name>/*` | not on the filesystem: webapp files (manifest `webapps` section, Phase 7) | — |

That is why `pipelines/` is **flat**: kind folders (`pipelines/snt/<name>/`) would push the kind into
every workspace path and change the layout analysts know. The kind lives in `pipelines/registry.json`
and in the manifest instead.

**Manifest v2** stays close to v1: `schema_version: 2`, `version`, `repo`,
`pipelines: {<name>: {path: "pipelines/<name>", kind, code, zip_files[]}}`, `files: {<path>: sha256}`
(unchanged meaning: path = repo path = workspace path), `webapps: {<name>: {path, files{}, version}}`,
`repository_files`. The deployer's `split_manifest()` and the checker's `parts[0]` checks move to
"under `pipelines/<name>/`". The schema, the builder (`tools/build_manifest.py`), the deployer and the
checker change **in the same PR**, which is the monorepo's main benefit.

**Watch for:** anything that hardcodes `code/` as the shared-R path. That includes the notebooks, the
`.r` utils, `snt_lib` (`pull_scripts_from_repository`, and possibly path helpers) and the checker's
scan exclusions. Grep for all of these in Phase 1. If `snt_lib` hardcodes it, a pinned `snt_lib`
release is a prerequisite of the rename (D-3). → **Checked 2026-11-05:** `snt_lib` does not, so no
release is needed. The list of references is under Phase 1, "Facts checked".

### 4.3 R-half layout — spikes first (D-5)

- **S1 — does `openhexa pipelines push` handle a pipeline folder with `code/*.ipynb` + `utils/*.r`
  subfolders?** The bootstrap docs say the SDK's `generate_zip_file()` recurses and keeps
  `.ipynb`/`.r`. Check: push a toy pipeline to a test workspace, then read the version zip (the
  checker already reads `currentVersion.zipfile`). Also check zip size and push time with the real
  notebooks.
- **S2 — can a running pipeline use notebooks from its own bundle?**
  - (a) Does papermill find the R kernel when the notebook path is relative to `__file__`?
  - **(b) ★ PREFERRED** — can `pipeline.py` *self-sync*? On start it copies bundled notebooks and
    `.r` files to the workspace filesystem when they are missing or older (archiving the replaced
    copy, as the deployer does), and never touches `_<CC>` variants.
  - (b) would retire Pull scripts for **both** routes and still leave notebooks on the filesystem for
    the OH UI. Test (b) first; test (a) only if (b) fails.
  - Open sub-questions for (b): how "older" is decided (hash vs the version's own manifest, never
    mtime); whether `shared/r/` travels inside every pipeline zip (duplicated) or is synced by one
    pipeline only; and a per-run escape hatch for an analyst who edited a notebook on purpose.
- **Outcomes:**
  - S1 fails → **fallback layout**: `pipelines/<name>/` holds the Python only, and the R half lives
    in a parallel tree. The parallel tree must still sit at a path equal to its workspace path (D-9),
    which means today's split layout under a new root. Nothing else in this plan changes.
  - S1 passes and S2(b) fails → co-locate. The bundled copy is inert, and the filesystem copy is
    delivered by the deployer (or by Pull scripts for template users).
  - Both pass → co-locate plus self-sync (OD-9).

---

## 5. Phases

Each phase ends in reviewable PRs. Nothing touches a live country workspace. All OpenHEXA work happens
in **new test workspaces** (names TBD, e.g. `snt-mono-test`, `snt-mono-templates`).

### Phase 0 — Read, spike, decide (no new repo yet)
- [x] Read the config editor's docs (2026-11-02): spec stage, no app code. It vendors the skeleton,
      reimplements `validate_config()` in JS, and targets `configuration/SNT_config.json` with
      drafts in `configuration/snt_config_history/`, with a `DatasetStore` (`SNT_CONFIGURATION`) as
      the long-term goal. → see Phase 5 sequencing note.
- [x] Diff the editor's skeleton against `snt_development/contracts/snt_config_json/SNT_config.skeleton.json`
      (`""`/`null` placeholder differences are known). Decide which one becomes the single copy.
      → 2026-11-03, D-11; see [Phase 0 findings](#phase-0-findings).
- [x] Spike S1 (push a nested toy pipeline, inspect the zip). → **passed** 2026-11-03.
- [x] Spike S2: self-sync (b, preferred) first, bundle execution (a) only as a fallback. → **both
      passed** 2026-11-03.
- [x] Spike S3 — **template code derivation.** In a scratch workspace, publish a template, create a
      pipeline from it, and record how the code is derived (from the template name? from the pipeline
      name?). Goal: the new templates produce code `snt-map-extracts`, not `b-2-map-extracts`.
- [x] Spike S4 — can the orchestrator read `PipelineRun.config` (the run's `run_report_only` value)
      through the webapp proxy with `PIPELINES_READ`? Grep `reference/openhexa/schema.generated.graphql`
      first. → **yes**, 2026-11-03.
- [ ] Spike S5 (optional, Phase 7 input) — `updateWebapp(files:)` with a CUSTOM-connection token.
- [~] Pre-publication audit list (first pass done 2026-11-03; see findings): which files in the source trees are unfit for a public repo? Known
      candidates: the persona questionnaire (`docs/research/personas/…`, answers from real people),
      country-specific `_<CC>` notebooks with DHIS2 UIDs, workspace slugs, Jira references,
      `ignore/` content (incl. the editor's NER draft configs).

#### Phase 0 findings

Spike code lives in `snt_development/ignore/spikes/` (git-ignored, throwaway); only the results are
recorded here. Test workspaces: `snt-testing` (runs) and `snt-development-sandbox` (template publishing).

**Skeleton diff (2026-11-03).** The editor's copy (`docs/SNT_config.skeleton.json`, 2,180 B) is
**older** than `contracts/snt_config_json/SNT_config.skeleton.json` (2,630 B). Every structural
difference is resolved in favour of the `snt_development` copy by the pipeline code:

| Path | `snt_development` | editor | Which is right |
|---|---|---|---|
| `SNT_DATASET_IDENTIFIERS.SNT_POPULATION_USER_PROVIDED` | present | missing | dev — read by `snt_user_population` |
| `SNT_DATASET_IDENTIFIERS.SNT_RESULTS` | missing | present | dev — retired with `snt_assemble_results` (ADR 0006) |
| `DHIS2_REPORTING_RATES.REPORTING_INDICATORS` | `{ACTUAL_REPORTS, EXPECTED_REPORTS}: null` | missing | dev — read by `snt_dhis2_extract` (`pipeline.py` ~l.375) |
| `DHIS2_REPORTING_RATES.REPORTING_DATASETS` | one example entry (`DATASET: null`, `METRICS`) | `[]` | either; dev's shows the entry shape |
| `SNT_CONFIG.{COUNTRY_CODE, COUNTRY_NAME, DHIS2_ADMINISTRATION_1/2}` | `null` | `""` | known placeholder difference — **convention to settle** |

Key order is otherwise identical. **Decided (D-11):** the `snt_development` copy becomes the single
copy at `contracts/snt_config/` (Phase 1), with `null` for every empty single value. The editor's
"emit `""` for empty strings" rule then changes to `null` (its rule 4 already treats `null` as empty).
The conformance fixtures (Phase 3) confirm `validate_config()` accepts it.

**S1, local half (2026-11-03).** `openhexa` SDK 2.22.7, `openhexa/cli/api.py::generate_zip_file()`:
it recurses (`glob("**/*")`) and keeps relative paths, so `code/`, `reporting/`, `utils/` and deeper
folders all go into the zip. Only files with these extensions are kept: **`.py .ipynb .txt .md .r .sql`**.
Consequences:
- **`.json` never travels in a pipeline zip.** A bundled hash manifest or config for self-sync (S2b)
  must be `.txt`/`.py`, or be computed at run time.
- Only `<dir>/workspace` and virtualenvs are excluded. **`.gitignore` is not respected**: a local
  `ignore/` subfolder with executed notebooks ships if it exists. Example:
  `pipelines/snt_dhis2_formatting/reporting/ignore/` (19 MB locally) made a 12.6 MB zip. CI pushes
  from a clean checkout so it is not affected, but anyone pushing by hand from a working tree is.
- Size, with every real pipeline's R half copied into its folder (clean tree, no `ignore/`): every
  zip is 10–56 KB. Size is not a concern.

**S1 + S2, on OpenHEXA (2026-11-03, `snt-testing`, toy `spike_monorepo_layout`, 4 runs).**
- **S1 passed.** The runner unpacks the whole nested zip at `/home/jovyan/pipeline` (= the folder of
  `__file__`): `code/`, `reporting/`, `utils/` and `_shared/r/`, all 7 files. The cwd is
  `/home/jovyan`, the workspace is `/home/hexa/workspace`, and the bundle folder is writable.
- **S2b (self-sync) passed.**
  - First run: copied the 4 missing files, then ran the **synced** notebook with the `ir` kernel
    (R 4.5.3). The notebook sourced both the pipeline's own `utils/` and the `shared/r` stand-in.
  - Second run, after a simulated analyst edit and a `spike_BFA.ipynb` variant: replaced exactly the
    edited file (hash differed), archived the old copy to `_archive/<stamp>/`, left 3 files unchanged,
    and **left the `_BFA` variant untouched**.
  - The archive stamp is in UTC (container clock).
- **S2a (bundle execution) passed as well.** Papermill ran `/home/jovyan/pipeline/code/spike.ipynb`
  directly, and it sourced the `.r` files from the bundle.
- **Trap for both:** R's working directory is `/home/jovyan`, *not* the notebook's folder. Notebooks
  must build every path from an injected root (as the SNT notebooks already do from
  `SNT_ROOT_PATH`). A `source("utils/x.r")` relative to the notebook's own folder will fail.
- **Still open (S2b sub-questions, settle in Phase 1/6):**
  - "older" was decided by sha256 against the bundle, which works. Consequence: any edit an analyst
    makes to a *generic* notebook is archived and overwritten on the next run, so the per-run escape
    hatch is required. It is implemented (`skip_sync`) but has not been run.
  - `shared/r/`: duplicate it in every zip (as tested) or have one pipeline sync it.

**S4 passed (2026-11-03, orchestrator in `snt-testing`).** The query
`pipelines{runs{id status config}}` through the webapp proxy returns each run's `config`, including
`run_report_only`. `PIPELINES_READ` is enough → OD-7 needs no `RUN_MODE`.

**S3 (2026-11-03, published from `snt-development-sandbox`, installed in `snt-testing`).** The publish
UI has **no code field**. A template named `SNT Spike Template` produced pipeline code
**`spike-monorepo-layout`**. So the code comes from the **source pipeline** (its code
`spike-monorepo-layout`, which here equals the `@pipeline("spike_monorepo_layout")` id), **not from
the template name**. If that code already exists, the new pipeline gets a random suffix:
`spike-monorepo-layout-599458`, `-2ab2b6`, …
- **For the new templates (Phase 6):** the source pipelines in the publishing workspace must
  already have the code `snt-<name>`. The template's display name can be anything.
- **For the deployer/checker:** a template-created pipeline can have the code `snt-<name>-<6 hex>`,
  so matching on the code must accept that suffix (or report it).
- The old `b-2-map-extracts` codes are explained if the source pipelines in `snt-development` carry
  those codes. Not verified.
- Not separated by this test: whether the code comes from the source pipeline's code or from its
  `@pipeline` id. Both were the same here.

**Pre-publication audit (2026-11-03, first pass).** A text scan of the four local source trees.
It found no credentials, `.env` files or raw country data files. The UID scan only catches 11-character
tokens that mix upper case, lower case and digits, so a second pass is needed after the snapshot copy.
**The import is a file copy, so `.gitignore` does not protect it**: the copy step itself must leave out
`ignore/`, `deprecated/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/` and `*:Zone.Identifier`.

| Verdict | File(s) | Why |
|---|---|---|
| **leave out** | `snt_development/pipelines/snt_dhis2_extract/code/NER_pyramid_format.ipynb` | NER notebook *without* the `_NER` suffix: 14 org-unit-group UIDs + a Jira link |
| **leave out** | `…/snt_dhis2_formatting/country_specific/snt_dhis2_formatting_pyramid_BDI.ipynb` | Facility UIDs next to facility names |
| **leave out** | `…/pipelines/snt_dhis2_outliers_{detection,removal_imputation}/code/*:Zone.Identifier` | Windows metadata with a personal email in notebook URLs |
| **leave out** | orchestrator `docs/research/personas/persona_questionnaire_answered-20260713.md` | Real people's answers on internal strategy |
| redact | `…/snt_dhis2_extract/utils/snt_dhis2_extract_report.r` | Shared helper with a NER-only fix: 7 facility UIDs + real coordinates → move to a NER-only file that is not imported |
| **leave out** (D-13) | Every `*_<CC>.ipynb`, e.g. `…/snt_dhis2_incidence/country_specific/snt_dhis2_incidence_NER.ipynb`, `…/snt_seasonality_rainfall/code/snt_seasonality_rainfall_NER.ipynb` | Country variants are not imported. These two also held Jira refs, a Google Sheets link, a live workspace URL (`ner-snt-process`) and 1 UID |
| done | `…/snt_dhs_indicators/code/snt_dhs_vaccination_computation.ipynb` | Had saved outputs in 3 cells (dataset version name, COD shapes size). Stripped on `main`, 2026-11-03 |
| redact | 2 notebooks (`reporting_rate_dataset`, `outliers_imputation_path`), 1 UID each in a comment | Replace with a placeholder |
| redact | editor `docs/{config_field_matrix,config_analysis_notes,PRODUCT_SPEC}.md`, `prototype/index.html` | Real UIDs from country configs used as examples (18 in total), bucket name `snt-ner-files` |
| redact | Live country workspace slugs: `ner-snt-process` (`snt_metadata_json_trim/readme.md`), `cmr-snt-process` (orchestrator `BACKLOG.md`, a wireframe) | Live country workspaces |
| ok (D-13) | Jira refs (`bluesquare.atlassian.net/browse/SNT25-*`), about 30 across all four repos | Kept |
| ok (D-13) | Bluesquare dev-workspace slugs (`snt-development`, `snt-development-sandbox`, `snt-testing`, `snt-app-dev`), about 25 mentions | Kept |
| ok (public by choice) | Team-glossary Google Sheets link, shared as "anyone with the link" (`docs/GLOSSARY.md`, `snt_dhis2_incidence.ipynb`) | A public reader can open it. Kept: the sheet may be public (Giulia, 2026-11-03) |
| ok | Owner names (Giulia / `@sPuntinG`) in OWNERSHIP, ADRs; `secrets.OH_TOKEN` references in workflows; dummy `mario.rossi@example.org` | — |

Checked 2026-11-03 (`git ls-files`): the `Zone.Identifier` files are **not tracked** (they exist only
locally). `snt_dhs_vaccination_computation.ipynb` was **tracked with its outputs** (R2 broken on
`main`). It was stripped with `nbstripout` and merged to `main` on 2026-11-03 (branch
`fix/strip-dhs-vaccination-outputs`).

### Phase 1 — Snapshot import + layout (behaviour-neutral)

**Outcome:** `BLSQ/snt-stratification` holds every source tree in the §4.1 layout, landed through
**one PR** (branch `phase-1/snapshot-import`), plus one small PR in `snt_development` that turns this
file into a pointer. No pipeline, workflow or workspace behaviour changes. The runbook below was
written on 2026-11-05. Follow it in order and tick each step. **Agents:** run no `git`/`gh` command
yourself (R19). Give Giulia the exact commands.

**Rules for this phase**
- **Export with `git archive <sha>`**, never by copying a working tree. It exports tracked files only,
  so `ignore/`, `__pycache__/`, `*:Zone.Identifier` and executed notebooks under `ignore/` stay out.
  It is read-only: it changes nothing locally or on GitHub (it is unrelated to archiving a repo).
- **Remove everything that must not be public *before* the import commit**: leave-outs, redactions,
  notebook outputs. History is permanent. A later "remove" commit does not unpublish anything.
- **Import no `.github/` folder.** The old `push_snt_*.yaml` deploy to `snt-development`. Phase 3 reads
  `generate_manifest.yaml` at its provenance SHA.
- **Merge the PR with "Create a merge commit"**, not squash. The import commit (files exactly as at
  the source SHA, at their new path) must stay separate from the edits made after it.
- Keep source file names. Renames beyond the mapping below wait for Phase 2.
- If a source repo's `main` moves after P1-5, redo P1-5 → P1-7.

**Facts checked 2026-11-05** (they settle §4.2 "Watch for"):
- `snt_lib` 0.23.0 does **not** hardcode the shared root `code/`. Its only `"code"` is the
  per-pipeline `pipelines/<name>/code/` (`pull_scripts_from_repository`), which does not move. So
  **the rename needs no `snt_lib` release**.
- Shared-R references to change (commit 5):
  - About 33 notebooks and `.r` files build `file.path(ROOT_PATH | SNT_ROOT_PATH, "code", …)`, either
    in `source(… "snt_utils.r" | "snt_report.r" | "snt_palettes.r")` or in `CODE_PATH <- …`.
  - `snt_dhis2_extract/pipeline.py` creates the workspace folder `"code"` (`folders_to_create`,
    ~l.1465).
  - `snt_pipeline_deployer` mentions `code/` in docstrings and log messages. Its logic
    (`parts[0] != "pipelines"`) changes in Phase 4.
- The old `snt_development/.gitignore` ignores `**/[Tt]est*`, which would hide `tests/`. It also
  ignores every `*.json`, `*.yaml` and `*.yml`.
- `pipelines/snt_dhis2_outliers_{detection,removal_imputation}/` and the root `snt_workspace_check/`
  and `snt_workspace_manager/` hold only untracked files, so `git archive` drops them.
- Bootstrap tests find the code through `Path(__file__).parents[1] / "snt_<…>"`, so they break after
  the move. Fixed in commit 8.
- LICENSE: all four repos use MIT (Bluesquare). Only the year differs (2026 in the orchestrator).
  DOC-5 is still open.

**Source → target mapping** (implemented by the staging script, P1-6)

| Source (at its SHA) | Target |
|---|---|
| ***snt_development*** | |
| `<name>/` (root folder with `pipeline.py`) + `pipelines/<name>/` | `pipelines/<name>/` (merged) |
| `pipelines/CLAUDE.md` | `pipelines/CLAUDE.md` |
| `code/*.r` | `shared/r/` |
| `contracts/snt_config_json/` | `contracts/snt_config/` |
| rest of `contracts/`, `catalogs/`, `docs/`, `dev/`, `pyproject.toml`, `LICENSE`, `README.md`, `CLAUDE.md` | same path (root `CLAUDE.md` is interim until Phase 2) |
| `.claude/` | already on `main` (P1-3) |
| `.gitignore` | base of commit 1, not imported as is |
| `deprecated/`, `.github/`, `snt_workspace_check/`, `snt_workspace_manager/` | **left out** |
| ***snt_webapp_orchestrator*** | |
| `webapps/orchestrator/CLAUDE.md` | `webapps/orchestrator/CLAUDE.md` |
| rest of `webapps/orchestrator/` (`index.html`, `app.js`, `styles.css`, `*.json`, …) | `webapps/orchestrator/app/` |
| `docs/OPENHEXA_WEBAPP_RUNTIME.md` | `docs/` |
| rest of `docs/` (incl. `decisions/`; `research/` minus the persona questionnaire) | `webapps/orchestrator/docs/` (ADRs and BACKLOG merged in Phase 2) |
| `design/` | `webapps/orchestrator/design/` |
| `contracts/*.schema.json` | `contracts/orchestrator/` |
| `reference/` | `reference/` |
| `pipelines/create_pipeline_cards/` | `pipelines/create_pipeline_cards/` |
| `CLAUDE.md` (repo rules R1–R12) | `webapps/CLAUDE.md` (interim) |
| `README.md` | `webapps/orchestrator/README.md` |
| `.claude/`, `.github/`, `.gitignore`, `LICENSE` | left out (`.gitignore` rules go into commit 1, `.claude/` is merged in Phase 2) |
| ***snt_workspace_bootstrap*** | |
| `snt_workspace_deployer/`, `snt_pipeline_deployer/`, `snt_workspace_checker/` | `pipelines/<same name>/` |
| `docs/contracts/*` | `contracts/` |
| rest of `docs/` (incl. `context/`) | `docs/bootstrap/` |
| `CLAUDE.md` | `docs/bootstrap/CLAUDE.md` (interim) |
| `tests/`, `tools/` | `tests/`, `tools/` |
| `pyproject.toml`, `dev/environment.yml` | merged in commit 4, not imported as is |
| `.claude/`, `.gitignore`, `LICENSE` | left out (as above) |
| ***openhexa-webapps-edit-snt-config-json*** | |
| `CLAUDE.md` | `webapps/config_editor/CLAUDE.md` |
| `docs/PRODUCT_SPEC.md`, `docs/EXTERNAL_CHECKS.md` | `webapps/config_editor/docs/` |
| `docs/config_analysis_notes.md`, `docs/config_field_matrix.md` | `contracts/snt_config/` |
| `prototype/index.html` | `webapps/config_editor/design/index.html` |
| `docs/SNT_config.skeleton.json`, `ignore/`, `LICENSE` | left out (D-11, D-7) |

**Steps**
- [ ] **P1-0 Close Phase 0.** Merge `plan/snt-single-repo` into `main`. The local spike code stays in
      `ignore/spikes/` (git-ignored). Optional, not blocking: delete the spike pipelines and the
      template in `snt-testing` and `snt-development-sandbox`.
- [ ] **P1-1 Land this runbook** in `snt_development` (branch `plan/phase-1-runbook`, commit
      `plan: Phase 1 runbook`).
- [ ] **P1-2 Create the repo** (GitHub UI, Giulia). Name `BLSQ/snt-stratification`, with **no**
      README, `.gitignore` or licence. Recommended visibility: **private until P1-10**, so that the
      second audit pass (P1-7) and DOC-5 come before anything is public. Clone it to
      `~/Bluesquare/snt-stratification`.
- [ ] **P1-3 Seed `main`** by a direct push, before protection exists:
  - a `README.md` stub;
  - `.claude/` from `snt_development` (`settings.json`, `hooks/block-destructive-git.py`,
    `README.md`, `skills/`; no `__pycache__/`).

  Test the guardrail as `.claude/README.md` describes. Commit:
  `chore: initialise snt-stratification with the R19 git guardrail`.
- [ ] **P1-4 Protect** (GitHub settings):
  - Ruleset on `main`: require a pull request (0 approvals, so Giulia can merge her own PRs), block
    force pushes, block deletion.
  - Tag ruleset on `v*`: restrict updates and deletions, with an empty bypass list.
- [ ] **P1-5 Record the source SHAs.** For each of the four source repos, confirm that `main` is clean,
      pushed and holds everything wanted. Then run `git -C ~/Bluesquare/<repo> fetch origin` and
      `git -C ~/Bluesquare/<repo> rev-parse origin/main`. Write the four SHAs into the Session log.
- [ ] **P1-6 Stage the export** into `~/Bluesquare/_import_staging/` (outside every repo), using a
      throwaway script `snt_development/ignore/phase1/stage_import.sh` that implements the mapping:
  1. For each repo, `git -C <repo> archive <sha>`, extracted into `src/<repo>/`.
  2. Copy into `tree/` according to the mapping.
  3. Delete the persona questionnaire and the 4 country notebooks (D-13), plus any `country_specific/`
     folder left empty:
     - `snt_seasonality_rainfall/code/snt_seasonality_rainfall_NER.ipynb`
     - `snt_dhis2_extract/code/NER_pyramid_format.ipynb`
     - `snt_dhis2_incidence/country_specific/snt_dhis2_incidence_NER.ipynb`
     - `snt_dhis2_formatting/country_specific/snt_dhis2_formatting_pyramid_BDI.ipynb`
  4. Redact each `redact` row of the Phase 0 audit table. Placeholders: `<DHIS2_UID>`,
     `<country-workspace>`. The NER-only block of `snt_dhis2_extract_report.r` is removed (NER does
     not run from this repo, D-13).
  5. Run `nbstripout` on every `.ipynb` in `tree/`.
  6. Write `import_manifest.tsv` with target path, source repo, source path and sha256, and mark
     every redacted file `modified`.
- [ ] **P1-7 Second audit pass** on `tree/`. Grep for:
  - 11-character UID-shaped tokens (`\b[A-Za-z][A-Za-z0-9]{10}\b` that contain a digit; review the
    hits by hand);
  - `-snt-process`;
  - e-mail addresses;
  - `token|secret|password`;
  - long decimal coordinates.

  Fix anything found in the script and rerun P1-6, never by hand. Record the result in the Phase 0
  audit table and close its `[~]` item.
- [ ] **P1-8 Build the PR** on `phase-1/snapshot-import`, one commit per row, in this order:

| # | Content | Commit message |
|---|---|---|
| 1 | `.gitignore`: the rules of the three repos, made valid for the new tree. `**/[Tt]est*` dropped. Allowlist: `pipelines/registry.json`, `webapps/*/app/**/*.json`, `tests/fixtures/*.json`, `contracts/**/*.json`, `catalogs/SNT_metadata_master.json`, `.claude/settings.json`, `dev/environment.yml`, `.github/workflows/*.yaml` | `chore: .gitignore allowlist for the monorepo layout` |
| 2 | `tree/` copied in, `git add -A`, nothing else | `import: snapshot of snt_development, snt_webapp_orchestrator, snt_workspace_bootstrap and the config editor` |
| 3 | `README.md` provenance table (repo, SHA, files left out or redacted); `pipelines/registry.json` (`kind`: `snt`; `support` for `create_pipeline_cards` and `snt_metadata_json_trim`; `bootstrap` for the 3 bootstrap pipelines); `webapps/config_editor/app/.gitkeep` | `docs: provenance table and pipelines/registry.json` |
| 4 | One `pyproject.toml` (ruff from `snt_development`, pytest from bootstrap); one `dev/environment.yml` named `snt-stratification` (union of both envs) | `chore: one pyproject.toml and one conda env, snt-stratification` |
| 5 | Every shared-library `"code"` → `"shared", "r"`; `folders_to_create`; deployer wording. Per-pipeline `pipelines/<name>/code/` untouched | `refactor: shared R library path code/ → shared/r/ (D-9)` |
| 6 | Orchestrator `CLAUDE.md` + `docs/DEPLOYMENT.md`: the bundle is `app/`, and drag-and-drop uploads its contents. Editor `CLAUDE.md` links. Any other link broken by the moves | `docs(webapps): paths and links for the app/ + docs/ split (D-10)` |
| 7 | `ruff format .` and `ruff check --fix .`, no manual edits | `style: ruff format and autofix` |
| 8 | Bootstrap test paths (`pipelines/…`, `contracts/`), remaining path breakage, `pytest` green | `fix: test paths and path breakage after the import` |
| 9 | This file: header ("Where it lives"), Phase 1 checkboxes, Session log | `plan: Phase 1 results` |

- [ ] **P1-9 Verify** and paste the results into the PR description:
  - `git ls-files --others --ignored --exclude-standard` is empty, or lists only files that should
    be ignored.
  - Each file from commit 2 matches `import_manifest.tsv` (same sha256, or marked `modified`).
  - No notebook has outputs.
  - No `ROOT_PATH, "code"` left in any `.ipynb` or `.r`.
  - `ruff check .` is clean and `pytest` is green.
  - State that nothing was run on OpenHEXA. That happens in Phase 4.
- [ ] **P1-10 Merge** the PR with a merge commit. Then make the repo public if P1-7 passed and DOC-5
      is settled.
- [ ] **P1-11 Pointer PR in `snt_development`** (branch `docs/migration-plan-moved`):
  - This file becomes a one-line pointer to the new repo's copy.
  - Drop the MIGRATION_PLAN row from CLAUDE.md "Where to find things", and the README and BACKLOG
    pointers.
  - Commit: `docs: MIGRATION_PLAN moved to BLSQ/snt-stratification`.

  From then on, plan updates happen in the new repo, and `snt_development` is maintenance-only (D-4).

### Phase 2 — Docs and agent setup
- [ ] Root `CLAUDE.md` router (≤15 KB) with the **merged Register**. Rules shared by all three repos
      (git guardrail, `[ERROR]`/`[WARNING]` prefixes, Python style) appear once. Webapp-only rules move
      to `webapps/CLAUDE.md` (prefix `W`); notebook rules stay in `pipelines/CLAUDE.md`.
- [ ] `docs/ARCHITECTURE.md` (new): one-page system map. Units, the contracts between them, the
      deploy routes (deployer, new templates, old templates), and the release train.
- [ ] Import the ADRs per OD-5. Write the first new ADRs: monorepo (supersedes orchestrator REPO-1),
      release train (D-2), repo path = workspace path (D-9, §4.2), docs placement rule (D-10), templates kept alongside the deployer
      (**supersedes bootstrap D20**), publishing workspace for the new templates (**amends R5 / ADR 0004**).
- [ ] Merge the backlogs into one `docs/BACKLOG.md`. IDs collide (`TOOL-1`, `DOC-2`, `OH-1`… exist
      in several repos), so give each an area prefix (`PIPE-`, `APP-`, `BOOT-`, `CFG-`, `REL-`, `TOOL-`)
      and add an "old ID" column. Drop items this migration resolves: REPO-1/2, the multi-source
      roadmap §1, the CI path-filter typos.
- [ ] Update the skills for the new paths. Add a `release` skill: checklist, cutting an rc,
      verifying in the test workspace, then promoting.
- [ ] Sweep the local Claude memory and the `ignore/` plans of all source repos. Anything still true
      and useful goes **into the repo** (CLAUDE.md, ADR, BACKLOG, docs). Nothing may rely on local memory.

### Phase 3 — Tooling and CI
- [ ] `tools/build_manifest.py` (from the `generate_manifest.yaml` heredoc) emitting **manifest v2**,
      plus pytest coverage and fixtures.
- [ ] `contracts/release_manifest.schema.json` v2 (R22 becomes an in-repo rule: schema + builder +
      consumers in one PR).
- [ ] `ci.yaml`: ruff, pytest, contract validation, metadata validator (`validate_metadata.py`), link check.
- [ ] `release.yaml`: on `release: published`, build and attach `release_manifest.json`, a **slim
      release tarball** (only manifest-listed files, so the deployer doesn't download design PNGs and
      docs), and `webapp-<name>-<tag>.zip` with a stamped `version.json`.
- [ ] `push_templates.yaml`: one matrix job **generated from `pipelines/registry.json`** (kinds `snt` + `support`),
      so typo'd per-pipeline path filters (`readme.txt`, `worlpopclient.py`) can't come back.
      Target = the new templates workspace. Keep it **disabled** until Phase 6.
- [ ] `push_bootstrap.yaml` (per OD-4 once settled): registry `kind: bootstrap` pipelines → templates on every
      merge to `main`.
- [ ] Config conformance fixtures (`contracts/snt_config/fixtures/`) + a Python runner against the
      **pinned** `snt_lib` `validate_config()` in `ci.yaml`. The JS side runs the same folder when the
      editor's validator exists.

### Phase 4 — Bootstrap adapted to the monorepo
- [ ] Deployer and checker read manifest v2 (pipelines under `pipelines/<name>/`, `kind`, ignore
      `webapps` for now; the checker's root exclusions are updated for `shared/r/`).
- [ ] `github_repo` default = the new repo (no longer `snt_development_sandbox`).
- [ ] `.snt_release` marker records `repo` + tag (closes roadmap §7.4 for this case).
- [ ] Checker: name pipelines with legacy template codes (`b-2-map-extracts`, …) through a mapping
      table, instead of reporting them as plain `not_in_any_release`.
- [ ] **Release `v0.1.0` (pre-release) = pure move, no behaviour change.** Bootstrap a fresh test workspace,
      run the SNT chain for one country (copy a config), compare outputs with an old-repo workspace,
      and confirm the checker reports `match`. This isolates "did the move break anything" from the
      bold changes.

### Phase 5 — Breaking-change window (before the first production tag)
Batch the held-back breaking changes before any country adopts the new repo; later they would need a
major version. One PR each, each verified on a `v0.x` pre-release.

**Sequencing with the config editor:** the editor has no code yet, and its storage design (spec §6–7)
assumes `configuration/SNT_config.json` plus the current full shape. Settle the config split and the
`SNT_CONFIGURATION` dataset **before** building the editor's write step (`WorkspaceStore`), so it is
built once against the final contract. The form and validation (`LocalStore`, download only) can
proceed in parallel against the new schema.
- [ ] **CFG-1 dataset keys:** key = dataset slug in uppercase (`SNT_DHIS2_EXTRACTS: snt-dhis2-extracts`).
      Move the fixed identifiers out of `SNT_config.json` into `catalogs/snt_datasets.json`. NER's extra
      keys need a deliberate home.
- [ ] **Config split:** `SNT_config.json` keeps only the country-specific parts (`SNT_CONFIG`,
      `DHIS2_DATA_DEFINITIONS`). Write `snt_config.schema.json` (CFG-3) in `contracts/`, used by both
      the config editor and pipelines.
- [ ] **Config as dataset `SNT_CONFIGURATION`** (CFG-4), with a fixed slug by convention (it can't
      be looked up in itself). The editor writes through a validating pipeline. Needs a pinned
      `snt_lib` release (D-3), so sequence this together with CFG-1, since both touch config loading.
- [ ] **Report-only semantics** (OD-7): S4 first; otherwise `RUN_MODE` in `{CC}_parameters.json`;
      the orchestrator only counts full runs as "processed".
- [ ] Clear the `⚠ exceptions` the old repo couldn't touch: R11 (seasonality lowercase globals,
      PIPE-3), R14 (`run_reports_only`), R9 (ERA5 name, healthcare_access `File`), PIPE-5 (pregnant women).
- [ ] Orchestrator: fix `ohName` drift (APP-9) and the `create_pipeline_cards` README raw-URL docs.
      Consider generating `pipeline_map.json` ids from the manifest, so a pipeline can't be added
      without the map knowing.
- [ ] The editor follows each change: its rules 3 and 5 (emit all 15 dataset ids, keep the
      `POP_PREGNANT_WOMEN`/`_PREGNANT_WOMAN` mismatch) **must be reversed** in the same PR as CFG-1 and
      PIPE-5 respectively.
- [ ] Cut **`v1.0.0`** (first production release).

### Phase 6 — New template set (for teams not using the deployer)
- [ ] Create a dedicated publishing workspace for the new templates (it does **not** replace
      `snt-development`, which keeps publishing the old templates from the old repo, D-4).
- [ ] Name the templates according to the S3 result, so template-created codes = `snt-<name>`.
- [ ] Enable `push_templates.yaml`. Decide whether it publishes on every merge to `main` or **only
      on release** (recommended: on release, so templates match a tag).
- [ ] Choose the R-half route for template users per OD-9: self-sync, or Pull scripts with `snt_lib`
      accepting repo + ref + path mapping (pinned `snt_lib` release).
- [ ] Bootstrap pipelines as templates (OD-4): enable `push_bootstrap.yaml`. That solves the
      chicken-and-egg problem (bootstrap roadmap §3; supersedes D19).
- [ ] Verify: a fresh workspace built from the new templates, then the checker run against the
      matching tag, gives `match`. Then switch it to the deployer and check it adopts the pipelines
      without duplicates.

### Phase 7 — Webapps on the release train (D-2)
- [ ] `version.json` (tag, commit, date) is stamped into the release zips. The app reads it and shows
      it in the footer (REL-1).
- [ ] Manual deploy route documented in `docs/RELEASES.md`: download `webapp-<name>-<tag>.zip` from
      the release and upload it. Never deploy from `main`.
- [ ] Checker: compare the deployed webapp files with the manifest hashes (needs the files to be
      readable; S5).
- [ ] Deployer webapp support (REL-2), gated on S5 / OpenHEXA OH-1. Scopes stay manual (they aren't
      part of the bundle).
- [ ] Keep the runtime contracts unchanged: webapp slug `snt-pipelines-orchestrator-cockpit`, bucket key
      `utils_pipelines/…` (REPO-3), template name `Create pipeline_cards.json`, `GENERATOR_KEY_MATCH`.

### Phase 8 — Adoption and sunset
- [ ] New country workspaces start from the new repo (deployer or new templates).
- [ ] Optional migration guide for existing workspaces: it is a re-bootstrap into a **new** workspace,
      not an in-place change.
- [ ] Old repos: README banner "Moved → new repo", then `snt_development` goes maintenance-only (D-4).
      `snt_webapp_orchestrator`, `snt_workspace_bootstrap` and the config editor freeze right after
      Phase 1 (they have no live-country CI, apart from `create_pipeline_cards` publishing from
      `snt-development`; keep that workflow alive until Phase 6).
- [ ] Archive criteria for `snt_development`: no workspace subscribed to the old templates **and**
      none using Pull scripts against it. An archived public repo stays readable, so Pull scripts keeps
      working even after archiving. Confirm the repo's visibility first.

---

## 6. Risks of a single repo for pipelines + webapps (and mitigations)

| Risk | Mitigation |
|---|---|
| A webapp fix has to wait for or ship with pipeline changes (one train) | Patch releases are cheap: the deployer skips unchanged pipelines (D22). Release from `main` only when `main` is green on an rc |
| CI blast radius: one merge triggers unrelated deploys | Path-scoped jobs. Templates publish on release, not on every merge |
| Mixed toolchains (Python, R notebooks, plain JS, pytest) | One conda env, one `pyproject.toml`. JS stays no-build (ADR 0007), so it needs no toolchain |
| Agent context bloat in a bigger repo | Root CLAUDE.md is a router only. Nested CLAUDE.md per area (`pipelines/`, `webapps/`, `webapps/<app>/`) loads lazily |
| Public exposure grows (more docs, research, workspace names) | Phase 0 audit. R1 stays enforced. Public webapps already must not hold secrets (ADR 0013) |
| Deployer downloads the whole repo tarball | Slim release tarball asset (Phase 3) |
| Everyone with write access can change everything | CODEOWNERS per area (OD-8), plus branch protection |
| `.gitignore` `*.json` trap silently drops webapp/contract JSON | Allowlist + `git check-ignore` check in Phase 1 |
| Webapp deployment still can't be automated (no CLI/SDK route, OH-1) | Not caused or fixed by the monorepo. Stamped zips make the manual route reproducible |
| Two live systems (old repo/templates + new) for a long period | D-4 maintenance-only + explicit archive criteria |
| `snt_lib` stays external for now: config, Pull-scripts and path changes still need cross-repo PRs | Pin a tag (D-3). Plan `snt_lib` releases as named steps in Phases 1, 5 and 6. Long term: bring it in (§10) |

## 7. Further issues found in the existing docs

- **The deployer can't see template-created pipelines** (it looks up by code only). Covered by S3 and
  the checker mapping.
- **CI drift in the old repo:** `readme.txt` filters, the `worlpopclient.py` typo, and
  `snt_metadata_json_trim` with no push workflow. Discovery-based CI removes this class of bug.
- **Workflow typos aside**, `generate_manifest.yaml`'s depth-1 `*/pipeline.py` anchor was the only
  thing excluding `dev/` and `deprecated/`. The new builder needs explicit roots.
- **ID collisions** across the three backlogs and rule registers. Handled in Phase 2.
- **Webapp scopes are metadata, not bundle files.** Any deploy route has to set them separately.
- **No repo in the release marker**, so tags from different repos would collide. Harmless today (D-8),
  but Phase 4 still adds `repo` to the marker.
- **Config editor ↔ pipelines duplication:** `validate_config()` lives in Python (`snt_lib`) and in
  the editor's JS, and nothing keeps them in agreement. Handled by the conformance fixtures (Phase 3).
  Server-side validation (editor spec §12.2 option 4) arrives naturally with CFG-4's validating pipeline.
- **The editor writes to the workspace, but the orchestrator only has `FILES_READ`.** The write
  mechanism and scopes are unverified (editor EXTERNAL_CHECKS item 4). Fold this into S5.
- **`snt_lib@release-v1` is unpinned** (tag or branch is unverified). D-3 pins it.
- **The tag ruleset is not yet on `BLSQ/snt_development`.** Set it on the new repo from day one.
- **The licence is provisional** (DOC-5). Settle it before going public.

---

## 8. Verification (end-to-end, per milestone)

- **Every PR:** `ruff check .`, `pytest`, schema validation, link check (CI from Phase 3; run them by
  hand before that).
- **Phase 1:** `git check-ignore` audit. Every imported file traced to the provenance table. Zero
  notebook outputs (`nbstripout --verify`-style check).
- **Phase 4 rc:** fresh test workspace via the deployer, then the checker reports `match` on all
  pipelines and files. One-country SNT chain outputs match an old-repo workspace (same filenames, same
  columns, same row counts).
- **Phase 5:** each breaking change on its own rc, re-running the affected pipelines plus the
  orchestrator in the test workspace.
- **Phase 6:** template-built workspace has codes = `snt-<name>`. The checker reports `match`. The
  deployer takes it over without duplicates.
- **Phase 7:** the footer shows the tag. The checker's webapp hash comparison matches.

## 9. Session log

| Date | Session summary | Next step |
|---|---|---|
| 2026-11-02 | Read md docs of snt_development, orchestrator, bootstrap, config editor. Agreed D-1…D-10 (D-10 = docs placement rule); plan made public in `docs/`; OD-4 has a leaning, not decided; S2(b) self-sync preferred | Phase 0: skeleton diff, spikes S1–S4, public-exposure audit |
| 2026-11-03 | Phase 0, desk half: skeleton diff (dev copy wins; `null` vs `""` to settle), S1 read from the SDK source (recurses; extension allowlist excludes `.json`; ignores `.gitignore`), S3 and S4 schema checks, first-pass public audit. Toy pipeline `spike_monorepo_layout` written for S1/S2. Then on OpenHEXA: **S1, S2a, S2b and S4 passed**. S3 showed that the code comes from the source pipeline, not from the template name, with a random suffix on collision. Decided D-11 (`null`), D-12 (report-only, was OD-7), D-13 (keep Jira links + dev slugs, leave out country notebooks); D-5 confirmed; OD-10 opened (versioning country/analyst-edited notebooks) | Glossary link kept (public by choice); DHS notebook stripped on `main`. Left: delete the spike pipelines + template; second audit pass at the start of Phase 1 |
| 2026-11-05 | Phase 1 prepared: checked the four source trees and `snt_lib` (no hardcoded shared `code/`, so the rename needs no release); wrote the Phase 1 runbook (mapping, steps P1-0…P1-11, commit sequence). Local spike code kept in `ignore/spikes/` | P1-0, then P1-1 (land the runbook), then P1-2 (create the repo) |

---

## 10. Long term: bring `snt_lib` (and a companion R library) into the monorepo

Not scheduled. Start only once Phases 0–5 are done **and** the colleague who maintains
`BLSQ/snt_utils` agrees. Until then D-3 applies (external, pinned).

**Why:** `snt_lib` holds config loading, `validate_config()`, `pull_scripts_from_repository` and
`save_pipeline_parameters`, which are exactly the code the bold changes touch (config split, config
as dataset, dataset-key naming, Pull-scripts retirement, provenance/`RUN_MODE`). As an external repo,
each of those becomes a two-repo, two-release change. Inside the monorepo it is one PR, versioned by
the same release tag as the pipelines that call it.

**Target:**
```
packages/
├── snt_lib/            ← Python package (pyproject.toml, src/snt_lib/, tests/)
│                          installed by pipelines as
│                          snt_lib @ git+https://github.com/BLSQ/snt-stratification@<tag>#subdirectory=packages/snt_lib
└── snt_r/              ← companion R library replacing shared/r/*.r (and duplicated pipelines/*/utils/*.r
                           helpers where they are really shared), e.g. an R package installable with
                           remotes::install_github("BLSQ/snt-stratification", subdir = "packages/snt_r", ref = "<tag>")
```

**Steps:**
1. Agree ownership and review with the `snt_utils` maintainer (record it in `docs/OWNERSHIP.md`).
   List its other consumers outside SNT, if any. They keep using the old repo or switch to the
   subdirectory install.
2. Freeze `BLSQ/snt_utils` at a final tag. Import that snapshot into `packages/snt_lib/`
   (provenance recorded, no history, same as D-1).
3. Add `packages/snt_lib` tests to `ci.yaml`. The conformance fixtures (`contracts/snt_config/fixtures/`)
   then test the real `validate_config()` in-repo.
4. Switch every `requirements.txt` to the subdirectory install at the release tag. The release builder
   must **stamp the tag** into requirements at release time (or the pipelines pin `@main` by
   mistake). Add a CI check for that.
5. Banner + archive `BLSQ/snt_utils`.
6. R companion (separate, later). Decide the delivery route first: a filesystem-synced folder (as
   `shared/r/` is today) vs an R package installed at runtime vs baked into the workspace image
   (OH-3 / DEP-2). Only then migrate `shared/r/` into `packages/snt_r/`. Each notebook's
   `source(...)` becomes `library(snt_r)` in one sweep.

**Risk:** a pip install from a monorepo subdirectory downloads the whole repo archive. Keep the repo
lean (no large binaries) or publish wheels as release assets.

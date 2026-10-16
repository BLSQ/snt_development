---
name: handover
description: Run the CLAUDE.md handover checklist on the current change in snt_development and write the handover note (what was and was not verified, Pull scripts = ON, push command, test-run plan). Use before calling any change done, before opening a PR, or when asked to "hand over", "wrap up" or "check my change".
---

# Handover

Executes the [Handover checklist](../../../CLAUDE.md#handover-checklist) for the change in front of
you. The checklist is the source of truth; this file only says *how* to check each item. Report
every item as **pass / fail / not applicable / not verified** — never skip one silently, and never
report a check you did not run.

All commands run from the repo root, in the `snt_development` conda env.

## 1. Find the change

Git is gated (**R19**): **propose** these read-only commands and run them only once approved —

```bash
git status --short
git diff --name-only main...HEAD      # committed on this branch
git diff --cached --name-only         # staged
```

From the file list, note: changed pipeline dirs (`<name>/pipeline.py`), changed notebooks and `.r`
files (`pipelines/<name>/…`, `code/*.r`), changed `readme.md`s, and `metadata/SNT_metadata_master.json`.

## 2. Run the checks

| # | Item (rule) | How |
|---|---|---|
| 1 | Lint (R18) | `ruff check <each changed dir>` — report the output verbatim if not clean. Pre-existing hits in files you did not touch: say so, don't fix them in passing. |
| 2 | Notebook outputs stripped (R2) | Any non-empty `outputs` in a changed notebook is a **fail**. Do **not** use `nbstripout --verify` as the gate: it flags every notebook in the repo for harmless cell metadata. Check with the snippet below. A stray `execution_count` with no output is a note, not a fail. |
| 3 | No data staged (R1) | In the staged list: any `.csv`, `.xlsx`, `.parquet`, `.rds`, `.geojson`, `*.json` outside the `.gitignore` negations, or `*:Zone.Identifier` → **fail**. Never suggest `git add -f`. |
| 4 | Injected parameters ↔ fallbacks (R11, R12) | `python3 .claude/skills/handover/check_notebook_params.py <changed pipeline dirs>` — also run it when only a notebook changed (pass that pipeline). ERROR = fail. Resolve every UNVERIFIED line by hand (open the dict and the notebook's `exists()` cell) and say you did. |
| 5 | New outputs published and documented (R4, R16) | For each `add_files_to_dataset(...)` added or changed: the file is listed in that pipeline's `readme.md` *Outputs → Published* and in [`docs/DATA_ARCHITECTURE.md`](../../../docs/DATA_ARCHITECTURE.md) §3. |
| 6 | Explorer metadata | A published column added or renamed → its layer in `metadata/SNT_metadata_master.json` added/updated in the same PR (use the `add-explorer-layer` skill). |
| 7 | New scripts registered (R6) | Every new notebook filename appears in that pipeline's `pull_scripts_from_repository(code_scripts=[...], report_scripts=[...])` call. Country variants (`_<CC>`) must **not** be registered. For a new `utils/*.r` file: no pipeline lists `.r` files in that call today, and how they reach a workspace (inside `snt_lib`) is unconfirmed — report it as *not verified*, don't invent a registration. |
| 8 | Country variants | For each changed generic notebook `X.ipynb`, list `X_<CC>.ipynb` siblings in the same folder (`code/` or `reporting/`; `country_specific/` folders are inert archives). Name those countries in the note: the change does not reach them. `main` is not a complete record of which variants exist in workspaces — say so. |
| 9 | Readme re-verified (R16) | `pipeline.py` parameters, outputs or publication changed → `readme.md` changed too, and checked against the code with the `write-pipeline-readme` skill's §3 table, not patched by memory. |
| 10 | New pipeline (R5) | Workflow file added under `.github/workflows/`, name updated in every place, `workspace:` left as `snt-development`. |
| 11 | Manifest shape (R22) | `generate_manifest.yaml` changed in a way that alters the manifest's shape → a paired schema PR in `snt_workspace_bootstrap` is open. |

Outputs check for item 2:

```bash
python3 -c "
import json, sys
for f in sys.argv[1:]:
    cells = [c for c in json.load(open(f))['cells'] if c['cell_type'] == 'code']
    n = sum(bool(c.get('outputs')) for c in cells)
    e = sum(c.get('execution_count') is not None for c in cells)
    print(('FAIL ' if n else 'ok   ') + f, f'outputs in {n} cell(s), execution_count in {e}')
" <changed .ipynb files>
```

Fix with `nbstripout <file>` (it rewrites the file — tell the user before running it).

## 3. Write the handover note

Stop at **"ready to push"**. Never deploy, never run a pipeline, never use the OpenHEXA MCP tools to
do either — the user pushes and runs. The note contains:

1. **Summary** — what changed, one or two lines per pipeline.
2. **Checklist results** — the table from §2, one line per item.
3. **Not verified** — everything you could not check (there is no test suite; R notebooks were not
   executed). Say which country/workspace it was tested in, or that it was not tested.
4. **Pull scripts** — if any notebook or `.r` file changed: *"Run with `Pull scripts` = ON, or the
   workspace keeps the old R code."* Plus the countries from item 8 that need a manual fix.
5. **Push command** — only for changed `pipeline.py` / `requirements.txt` (CI deploys these on merge
   to `main`; the command is for testing before merge, from `snt-development` only — R5):
   `openhexa pipelines push <pipeline_name> --yes`
6. **Test-run plan** — per pipeline: the parameter values to use, and the expected outcome (which
   files appear in which dataset, row/column checks, which report sections, what a skip looks like
   — missing inputs skip rather than fail in this repo).

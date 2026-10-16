# `dev/` — local development

Everything needed to work on this repo from your own machine. **Optional**: you can read the repo,
edit notebooks and commit without any of it. None of these tools runs on OpenHEXA.

## What works locally today

Local development is a **known pain point**, honestly stated:

| Kind | Local story |
|---|---|
| **Python-only pipelines** (`snt_dhis2_extract`, `snt_map_extracts`, `snt_worldpop_extract`, `snt_era5_climate_data`) | The best-supported path. Editable and lintable locally; still needs a workspace to actually run. |
| **Notebook-driven pipelines** (15 of them) | Editable locally and **executable against the remote workspace kernel** — see [`../pipelines/CLAUDE.md`](../pipelines/CLAUDE.md#editing-r-notebooks-the-vs-code-remote-kernel-loop). There is no fully offline loop. |

## Getting set up

Install by tier — you do not need all of it. Everything below is cross-platform.

| Tier | Install | Why | Effort |
|---|---|---|---|
| **1 — do this** | `conda env create -f dev/environment.yml` | Gets you `ruff` (the repo's only automated quality gate, and the only check you can run before opening a PR), plus `nbstripout` and `nbdime`. | ~3 min |
| **2 — if you edit R notebooks** | VS Code + the Jupyter extension | Lets you edit locally and execute on the workspace kernel. | ~5 min |
| **3 — rarely** | `openhexa` CLI | Already in `environment.yml`. CI deploys for you; only needed for a manual push — and read [R5](../docs/CONVENTIONS.md#r5) before you do. | — |

```bash
conda env create -f dev/environment.yml
conda activate snt_development

# after someone edits dev/environment.yml
conda env update -f dev/environment.yml --prune

ruff check .          # the repo's only automated code check
ruff check --fix .
nbstripout pipelines/<name>/code/<notebook>.ipynb    # before every notebook commit (R1, R2)
```

**These are desk tools only.** What a workspace installs is declared per pipeline in
`<pipeline_name>/requirements.txt`, a separate list that never meets this one. So a version drift
here cannot affect a pipeline run.

**You do not need a local R installation.** R code runs on the workspace kernel, so installing R
locally buys you syntax checking at best and a subtly different environment at worst — the
workspace runs a specific image, documented in
[`DATA_ARCHITECTURE.md` §7.4](../docs/DATA_ARCHITECTURE.md#74-the-workspace-runtime-image).

**Nothing extra is needed for the agent guardrail** (R19), but it uses a *system* `python3`, not the
one in this environment — see [`../.claude/README.md`](../.claude/README.md#which-python-the-hook-uses).

## The two files

| File | What it is |
|---|---|
| [`environment.yml`](environment.yml) | The shopping list — which tools to install. Conda, matching team convention, and the path this repo supports. |
| [`../pyproject.toml`](../pyproject.toml) | The style guide — `ruff`'s rulebook (line length, which mistakes to flag) — plus a `[project]` block: PEP 621 metadata, `requires-python`, and a short dependency list for anyone setting up with `pip install -e .` or `uv sync`. |

**The two dependency lists overlap on purpose.** `pyproject.toml`'s `dependencies` /
`[dependency-groups]` repeat part of `environment.yml`. That duplication is accepted for now, until
the team settles on a single local-workflow strategy
([ADR 0009](../docs/decisions/0009-conda-and-pep621-side-by-side.md)). If you add a tool, add it to
both; if they ever disagree, `environment.yml` is the one describing what the team actually runs.
Neither file reaches OpenHEXA — a workspace installs `<pipeline_name>/requirements.txt` and nothing
else.

`requires-python = ">=3.11"` in that block is also where `ruff` gets its target Python version (it
drives the `UP` and `FA` rules), so don't drop it without putting `target-version = "py311"` under
`[tool.ruff]` instead.

**Why `pyproject.toml` is not in this folder.** It has to sit at the repo root. `ruff` finds its
rules by starting at the file it is checking and walking *up* the folders until it finds one; from
`snt_dhis2_extract/pipeline.py` that search reaches the root and stops. A copy in `dev/` would only
govern `dev/` itself, and everywhere else `ruff` would quietly fall back to its own defaults —
wrong line length, most of the repo's rules switched off, and the same wrong squiggles in your
editor. It is the one piece that cannot be grouped here.

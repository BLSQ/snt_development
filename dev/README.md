# `dev/` — local Python development

Everything needed to edit this repo's Python on your own machine. **Optional**: you can read the
repo, edit notebooks and commit without any of it. None of these tools runs on OpenHEXA.

```bash
conda env create -f dev/environment.yml
conda activate snt_development

ruff check .          # the repo's only automated code check
ruff check --fix .
nbstripout pipelines/<name>/code/<notebook>.ipynb    # before every notebook commit (R1, R2)
```

| File | What it is |
|---|---|
| [`environment.yml`](environment.yml) | The shopping list — which tools to install. Conda, matching team convention. |
| [`../pyproject.toml`](../pyproject.toml) | The style guide — `ruff`'s rulebook (line length, which mistakes to flag). |

**Why `pyproject.toml` is not in this folder.** It has to sit at the repo root. `ruff` finds its
rules by starting at the file it is checking and walking *up* the folders until it finds one; from
`snt_dhis2_extract/pipeline.py` that search reaches the root and stops. A copy in `dev/` would only
govern `dev/` itself, and everywhere else `ruff` would quietly fall back to its own defaults —
wrong line length, most of the repo's rules switched off, and the same wrong squiggles in your
editor. It is the one piece that cannot be grouped here.

Full context: [`../CLAUDE.md`](../CLAUDE.md) → *Local development — current state*.

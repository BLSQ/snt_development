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
| [`../.gitattributes`](../.gitattributes) | Tells git to treat notebooks as notebooks. Needs the one-time setup below to take effect. |

**Why `pyproject.toml` is not in this folder.** It has to sit at the repo root. `ruff` finds its
rules by starting at the file it is checking and walking *up* the folders until it finds one; from
`snt_dhis2_extract/pipeline.py` that search reaches the root and stops. A copy in `dev/` would only
govern `dev/` itself, and everywhere else `ruff` would quietly fall back to its own defaults —
wrong line length, most of the repo's rules switched off, and the same wrong squiggles in your
editor. It is the one piece that cannot be grouped here.

---

## One-time git setup for notebooks

**Do this once per clone.** It takes about ten seconds and it prevents the two worst
things that happen to `.ipynb` files in git: leaked country data, and silently broken
merges.

```bash
conda activate snt_development

nbstripout --install        # strip notebook outputs on the way into git
nbdime config-git --enable  # diff and merge notebooks by cell, not by line
```

Check it worked:

```bash
nbstripout --status         # → "nbstripout is installed in repository ..."
git config --get filter.nbstripout.clean
git config --get merge.jupyternotebook.driver
```

### What these actually do, and why you should care

A `.ipynb` file is not code — it is a JSON document with the code, the execution
counts and the **cell outputs** all stored together. Git does not know that. Two
consequences:

| Without the setup | With it |
|---|---|
| Every committed notebook carries its outputs — the executed results, i.e. **real district-level health data**, straight into a public repository (breaks **R1**). | Outputs are stripped as the file is staged. Your local notebook keeps them; git never sees them. |
| A notebook diff is thousands of lines of unreadable JSON, so nobody genuinely reviews it. | `git diff` shows "cell 4 changed, this line of R differs". |
| Git merges the JSON **line by line**. It can report success while leaving you a notebook holding cells from two different versions, or quietly reinstating old code. This is the main source of dirty merges here. | Merges happen cell by cell. Real conflicts are still reported as conflicts — but git stops inventing wrong answers. |

Two things to know:

- **[`../.gitattributes`](../.gitattributes) is committed, but the tools are not.**
  The committed file only says *which* tool handles notebooks; the tools themselves
  live in your `.git/config`, which is local and cannot be shared. If you skip the
  setup, git falls back to plain text behaviour **with no error and no warning** — you
  simply lose the protection. So don't assume a colleague is set up; ask.
- **The CI check is your backstop, not your first line.**
  [`../.github/workflows/pr-checks.yaml`](../.github/workflows/pr-checks.yaml) fails a
  PR that contains data files or unstripped notebooks, precisely because the filter is
  per-clone. Better to catch it before you push.

### If a notebook conflict happens anyway

**Never hand-edit the JSON to resolve it.** That is how cells get silently orphaned or
duplicated. Take one side whole, then redo your change in the notebook editor:

```bash
git checkout --ours  path/to/notebook.ipynb   # keep your version, then re-apply theirs
git checkout --theirs path/to/notebook.ipynb  # keep their version, then re-apply yours
nbdime diff <a>.ipynb <b>.ipynb               # see what actually differs, by cell
nbdime mergetool                              # resolve visually, cell by cell
```

Run these yourself: `git checkout --<file>` discards changes, so the agent guardrail
(**R19**) blocks an AI assistant from running them. That is expected behaviour.

---

Full context: [`../CLAUDE.md`](../CLAUDE.md) → *Local development — current state*.

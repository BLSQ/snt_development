# `pipelines/` — rules for R notebooks and helpers

Loaded by Claude Code only when files under `pipelines/` are read; the root
[`../CLAUDE.md`](../CLAUDE.md) still applies in full. Rules referenced by ID are in its Register,
with the reasoning in [`../docs/CONVENTIONS.md`](../docs/CONVENTIONS.md).

> **Rule 2:** committing a notebook or `.r` change does *not* update any workspace. Operators must
> run the pipeline with **`Pull scripts` = ON** to pick it up. Say so in every handover.

## Notebook hygiene

- **Strip outputs before every commit** (`nbstripout`) (**R2**). Executed notebooks belong in
  `papermill_outputs/` and `reporting/outputs/` inside the workspace, never in git — they leak
  country data and produce unreviewable diffs.
- Windows `*.ipynb:Zone.Identifier` sidecars are ignored — do not commit them; several are already
  tracked by mistake under `snt_dhis2_formatting/reporting/`.
- **A notebook change needs a domain reviewer**, not just any reviewer: a change to
  `pipelines/*/code/`, `pipelines/*/utils/` or `code/` should be reviewed by the person responsible
  for that pipeline — [`docs/OWNERSHIP.md`](../docs/OWNERSHIP.md#2-pipeline-responsibilities).
  Convention only: nothing blocks an unreviewed merge
  ([ADR 0005](../docs/decisions/0005-no-codeowners.md)).

## Parameters injected from `pipeline.py` (R11, R12)

- Every parameter injected from `pipeline.py` needs an `if (!exists("X")) X <- <default>` fallback
  cell, so the notebook stays interactively runnable. Change both sides together, spelled
  identically — **case included**. A mismatch is silent: the notebook quietly runs on its
  hardcoded default instead of the operator's choice.
- **Notebook parameter globals are UPPERCASE** — `ROUTINE_DATA_CHOICE`, `SNT_ROOT_PATH`,
  `DEVIATION_IQR`. This distinguishes an injected pipeline parameter from an ordinary R local at a
  glance, and matches the UPPERCASE column convention.
- **Two pipelines predate the rule** and inject lowercase globals: `snt_seasonality_cases` and
  `snt_seasonality_rainfall`. Each is internally self-consistent, so it works — but read the
  pipeline's injected dict before writing the notebook's `exists()` cell. Not a permitted variant:
  logged for migration as [BACKLOG PIPE-3](../docs/BACKLOG.md#pipe-3). New parameters are UPPERCASE
  even when added to one of those two, *unless* that would leave a single notebook mixing both —
  converting a violator is an all-at-once change, not a drive-by. A notebook mixing
  `threshold_for_seasonality` and `THRESHOLD_FOR_SEASONALITY` is worse than either.
- Beware the near-miss in `snt_healthcare_access`: it injects UPPERCASE (`INPUT_FOSA_FILE`,
  `WORLDPOP_YEAR`) into the notebook but records lowercase keys in its parameters JSON. Both are
  intentional; only the notebook side is governed by R11.

## Editing R notebooks: the VS Code remote-kernel loop

The working loop today:

1. Work in a local clone of this repo, with the `.ipynb` open in VS Code.
2. Connect the Jupyter extension to the OpenHEXA workspace kernel. Code executes **on the OH
   server**, so you get the workspace's files, datasets, connections and globals — exactly as if
   you were in JupyterLab — while keeping VS Code's syntax highlighting, completions, git
   integration and AI tooling.
3. Commit from the local clone as normal.

**The gap — `utils/*.r` helpers.** The notebook `source()`s its helpers from the *workspace*
filesystem, not from your local clone. So changing a helper function means editing it in the OH
workspace, then copying the change back into the local clone to version it — by hand, in that
order, every time. It is the main friction in the R loop and the reason a helper change is easy to
lose. Until it is solved: **make the helper edit in the workspace, test it, then immediately copy
the file back and commit** — do not batch several helper edits before copying back.

Setup for this loop: [`../dev/README.md`](../dev/README.md).

## Country-specific notebook variants

**The mechanism.** A notebook whose filename is the generic name plus an underscore and a country
code — `snt_seasonality_rainfall_NER.ipynb` beside `snt_seasonality_rainfall.ipynb` — is executed
**instead of** the generic notebook, but only in the workspace whose `SNT_CONFIG.COUNTRY_CODE`
matches that suffix. Everywhere else the generic notebook runs and the variant is inert.

`pipeline.py` never names the variant. It passes `country_code=` to `run_notebook()` /
`run_report_notebook()` alongside the *generic* `nb_path`, and the substitution happens inside
`snt_lib` (external repo — do not guess its resolution rules beyond this):

```python
run_notebook(
    nb_path=pipeline_path / "code" / "snt_seasonality_rainfall.ipynb",  # generic name only
    country_code=country_code,                                          # ← enables the swap
    ...
)
```

**Why it exists.** It lets a country keep a bespoke analysis without that file being clobbered on
the next `Pull scripts` run ([ADR 0003](../docs/decisions/0003-country-variants-outside-pull-scripts.md)).
Variants are deliberately **not** listed in `pull_scripts_from_repository(code_scripts=[...])` —
only generic names are. That single omission is what makes them safe, and it cuts both ways:

| | `Pull scripts` = ON |
|---|---|
| Generic notebook | overwritten from `main` |
| `_<CC>` variant | **untouched — neither overwritten nor delivered** |

**The consequence — there is no version-propagation story.** A variant has to be created and edited
**in the workspace**. Committing it to `main` is *archival only*: nothing brings it into any
workspace, ever. A committed variant is a copy of what the workspace had at some past moment, with
no mechanism keeping the two in step and nothing that reports the drift. Treat the committed file
as a backup, not as the running code.

**Working rules:**

- **Prefer a config-driven branch inside the generic notebook** over creating a variant. A variant
  is a permanent fork that no tooling maintains.
- When a variant is unavoidable, record the reason and the ticket in a markdown cell inside it.
- After editing a variant in the workspace, **copy it back and commit it immediately** — same
  discipline as `utils/*.r` helpers, and for the same reason.
- **Before changing a generic notebook, check for `_<CC>` siblings.** A fix applied to
  `snt_seasonality_rainfall.ipynb` does **not** reach Niger if `..._NER.ipynb` exists there. State
  in the handover which countries need the change applied to their variant by hand.
- **Location matters.** A variant only takes effect when it sits in the same folder the generic
  notebook is loaded from (`code/`, or `reporting/`). Files parked in a `country_specific/`
  folder — `snt_dhis2_incidence/country_specific/snt_dhis2_incidence_NER.ipynb`,
  `snt_dhis2_formatting/country_specific/snt_dhis2_formatting_pyramid_BDI.ipynb` — are **not** on
  any execution path; they are archived copies. Only
  `snt_seasonality_rainfall/code/snt_seasonality_rainfall_NER.ipynb` is committed in an active
  location.

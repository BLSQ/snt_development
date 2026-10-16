# 0003. Country-specific notebook variants live outside `Pull scripts`

Date: not recorded
Status: Accepted

## Context

Some countries need a bespoke analysis. Notebooks reach a workspace through `Pull scripts`, which
**overwrites** the workspace copy from `main`. A country's custom edits to a generic notebook would
be clobbered on the next run with `Pull scripts` = ON.

## Decision

A notebook named `<generic_name>_<CC>.ipynb`, in the same folder as the generic notebook (`code/` or
`reporting/`), runs **instead of** the generic one in the workspace whose `SNT_CONFIG.COUNTRY_CODE`
matches `<CC>`. `pipeline.py` passes only the generic path plus `country_code=` to `run_notebook()`;
the substitution happens inside `snt_lib`. Variants are deliberately **not** listed in
`pull_scripts_from_repository(code_scripts=[...])`.

## Consequences

- `Pull scripts` never overwrites a variant. It also never **delivers** one: a variant must be
  created and edited in the workspace.
- Committing a variant to `main` is archival only. Nothing keeps the committed copy and the running
  copy in step, and nothing reports drift. `main` is not a reliable record of which variants exist.
- A fix to a generic notebook silently misses every country with a variant. Check for `_<CC>`
  siblings before changing a generic notebook, and say in the handover which countries need the fix
  by hand.
- A variant is a permanent fork that no tooling maintains, so a config-driven branch inside the
  generic notebook is preferred where possible.
- Files parked in a `country_specific/` folder are on no execution path.

Source: `CLAUDE.md` *Country-specific notebook variants*.

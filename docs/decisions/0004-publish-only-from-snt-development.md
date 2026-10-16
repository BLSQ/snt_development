# 0004. Publish SNT pipelines only from `snt-development`

Date: not recorded
Status: Accepted

## Context

Country workspaces install the SNT pipelines as OpenHEXA **template pipelines** and can subscribe to
automatic updates. OpenHEXA ties template publication to the workspace a pipeline is pushed from:

- a push from the workspace that owns the template publishes a **new version** of it, which flows to
  every subscribed country workspace;
- a push of the same pipeline from **any other** workspace creates a **separate, new template**: a
  duplicate nobody is subscribed to, which competes with the real one in the template list.

## Decision

`snt-development` is the single publication point for SNT pipelines. All 20
`.github/workflows/push_snt_*.yaml` workflows push there (`workspace: "snt-development"`). Never
change that value, and never run `openhexa pipelines push` for an SNT pipeline from a country or
personal workspace.

## Consequences

- Merging a `pipeline.py` or `requirements.txt` change to `main` is enough to reach every subscribed
  country.
- Each workflow stamps the published version with `--description` (the commit message) and `--link`
  (the commit URL), so the OpenHEXA version list doubles as deployment history. Write commit messages
  that read well there.
- This covers the Python half only. Notebooks and `.r` files travel by `Pull scripts`
  ([BACKLOG OH-1](../BACKLOG.md#oh-1)).
- Nothing enforces the rule except review of the workflow files.

Sources: `CLAUDE.md` rule R5; [`DATA_ARCHITECTURE.md` §2.2](../DATA_ARCHITECTURE.md#22-how-a-pipeline-version-reaches-a-country-workspace--the-template-mechanism).

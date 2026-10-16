# 0008. The SNT Explorer master list is edited by hand; no generator

Date: 2026-10-12
Status: Accepted

## Context

`metadata/SNT_metadata_master.json` lists every SNT Explorer layer the pipelines can publish. It was
bootstrapped on 2026-09-28 by a one-off generator, `build_snt_metadata_all_layers.py`, which traced
each `SOURCE_DATA` pointer from the code and wrote placeholder text. After that, the file started to
receive **curated** content: labels, descriptions, scales. Re-running the generator would overwrite
that content.

## Decision

The JSON file is the source of truth. Edit it by hand, reviewed in a PR. The generator was deleted on
2026-10-12 so it can never overwrite curated text. It remains in git history at commit `062cf0c`; do
not resurrect it over curated content.

## Consequences

- When a pipeline adds or renames a published column, its master-list entry must be updated in the
  same PR. Nothing does it automatically.
- Checking the file against the pipelines is manual for now. Scripted checks are
  [BACKLOG META-11](../BACKLOG.md#meta-11). Static extraction of columns from R code was judged not
  worth building.
- Two kinds of field get two kinds of review: the objective `SOURCE_DATA` against the code, and the
  curated fields by someone with domain knowledge.

Sources: [`metadata/README.md`](../../metadata/README.md); `contracts/snt_metadata_json/README.md`
(master-list section, before 2026-10-16).

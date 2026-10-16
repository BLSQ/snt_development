# `docs/decisions/` — architecture decision records

One file per decision that shapes how this repository works, so the *why* survives after the people
who made it move on. Format after Michael Nygard's
[ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

- An ADR records a decision **and its reasoning**. The rule itself lives where people apply it
  (`CLAUDE.md`, a contract README); the ADR is what that rule links to for "why".
- **Never edit an accepted ADR to change the decision.** Write a new one and mark the old one
  `Superseded by NNNN`. Fixing a typo or a link is fine.
- Open questions do not belong here: they go to [`../BACKLOG.md`](../BACKLOG.md). When a backlog
  item is settled by a decision, delete it there and write the ADR here.
- Where the original date or reason was never written down, the ADR says so. Do not fill the gap
  with a plausible guess.

## Index

| # | Title | Status | Date |
|---|---|---|---|
| [0001](0001-no-dag-engine.md) | No DAG engine or scheduler: pipelines are launched by hand | Accepted | not recorded |
| [0002](0002-alternative-methods-last-run-wins.md) | Alternative methods overwrite each other: last run wins | Accepted | not recorded |
| [0003](0003-country-variants-outside-pull-scripts.md) | Country-specific notebook variants live outside `Pull scripts` | Accepted | not recorded |
| [0004](0004-publish-only-from-snt-development.md) | Publish SNT pipelines only from `snt-development` | Accepted | not recorded |
| [0005](0005-no-codeowners.md) | No `CODEOWNERS` file: review by convention | Accepted | not recorded |
| [0006](0006-retire-snt-assemble-results.md) | Retire `snt_assemble_results`; the Explorer reads datasets directly | Accepted | 2026-10 |
| [0007](0007-snt-metadata-json-format.md) | `SNT_metadata.json`: a pointer per layer, keyed on config keys, time via `YEAR` | Proposed | 2026-09-08 |
| [0008](0008-master-list-maintained-by-hand.md) | The SNT Explorer master list is edited by hand; no generator | Accepted | 2026-10-12 |
| [0009](0009-conda-and-pep621-side-by-side.md) | conda for desk tools, with an accepted PEP 621 duplicate | Accepted | not recorded |

## Template

Copy this into `NNNN-short-title.md`, with the next free number. Keep it short (about 40 lines).

```markdown
# NNNN. Title

Date: YYYY-MM-DD
Status: Proposed | Accepted | Superseded by NNNN | Deprecated

## Context

What forced a decision, and the constraints at the time.

## Decision

What was decided, stated as a rule.

## Consequences

What follows — the good, the costs, and what to watch for.
```

Then add a row to the index above.

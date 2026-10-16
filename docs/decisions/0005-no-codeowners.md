# 0005. No `CODEOWNERS` file: review by convention

Date: not recorded
Status: Accepted

## Context

Changes to the analytics (`pipelines/*/code/`, `pipelines/*/utils/`, `code/`) compute
epidemiological indicators that decide where malaria interventions are deployed. A change that shifts
a number does not fail, does not warn, and is caught by no test. So the right reviewer matters more
than in most repositories. A `CODEOWNERS` file was drafted to route reviews automatically.

`[TODO: Giulia]` The reason the team decided against it was not written down. Add it here.

## Decision

No `CODEOWNERS` file. Every change to `main` goes through a PR and should be reviewed by a team
member; a change to the analytics should be reviewed by the person responsible for that pipeline, as
listed in [`OWNERSHIP.md`](../OWNERSHIP.md#2-pipeline-responsibilities).

## Consequences

- Nothing in GitHub blocks an unreviewed merge. The convention holds only because people follow it.
- **The burden is on the author** to request the right reviewer.
- **Silence is not approval.** A change to the analytics that nobody with context has read has not
  been reviewed, whatever the merge button showed.
- The convention depends on `OWNERSHIP.md` naming people, which it does not yet
  ([BACKLOG DOC-4](../BACKLOG.md#doc-4)).

Sources: [`OWNERSHIP.md` §1](../OWNERSHIP.md#1-the-review-convention); `CLAUDE.md` *Notebook hygiene*.

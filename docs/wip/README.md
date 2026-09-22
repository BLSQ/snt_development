# `docs/wip/` — SNT release management

Working documents for the release / workspace-versioning effort (SNT25-670). Nothing described here
is live in a country workspace yet.

**Four files, and they do not overlap.** Each states only what is *currently* true; anything that
stopped being true moved to `HISTORY.md`.

| File | Answers | Read it when |
|---|---|---|
| [`release_strategy.md`](release_strategy.md) | **Why** the release mechanism exists, what it delivers, and what is built today. | You need the shape of the whole thing, or the state of the manifest generator / Workspace Manager. |
| [`PRODUCT_SPEC.md`](PRODUCT_SPEC.md) | **What** the workspace checker must do — statuses, report contract, build phases, open decisions. | You are building or reviewing the checker. |
| [`pipeline_deployment_mechanism.md`](pipeline_deployment_mechanism.md) | **How** a pipeline is deployed into a workspace through the OpenHEXA API. | You are touching deployment, tokens or the GraphQL calls. |
| [`HISTORY.md`](HISTORY.md) | **What is no longer true** — superseded designs, closed issues, dead ends, deleted fixtures, the original spec draft. | **Before** investigating anything that smells already-solved, or before reopening a decision. |

## Rules for keeping these useful

1. **Current state only, in the three live documents.** If a paragraph explains what something *used
   to be*, or records a verification against a fixture that no longer exists, move it to
   `HISTORY.md` and leave a link.
2. **Nothing is deleted, it is relocated.** A closed problem keeps its write-up — the point of
   `HISTORY.md` is that the next reader does not pay to rediscover it.
3. **Decisions live in `PRODUCT_SPEC.md` §8**, numbered `D<n>`, so they can be cited rather than
   re-argued.

Local, uncommitted companions under `ignore/SNT25-670/` (runbooks containing `git`/`gh` commands for
a human to run) are referenced by name where relevant.

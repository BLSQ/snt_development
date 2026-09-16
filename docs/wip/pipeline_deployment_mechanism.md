# How OpenHEXA pipelines are actually deployed — findings

Companion to [`release_strategy.md`](release_strategy.md). Written for a fresh agent session
picking up phases 3–4 of the release roadmap.

**Status: investigation in progress, one blocking question still open.** Read
[Open question](#open-question--can-a-run-push-a-pipeline-version) before building anything.

---

## 1. The problem this document exists to solve

Phase 3 of the roadmap built a "Workspace Manager" pipeline that downloads a GitHub release
tarball and copies the tracked files into `workspace.files_path`. For the **R half** that is
sufficient and correct — notebooks `source()` their helpers from the workspace filesystem, so a
file landing there *is* the deployment.

For the **Python half it is not sufficient.** Writing `snt_dhis2_extract/pipeline.py` into the
workspace filesystem produces a *file*, not a runnable pipeline. OpenHEXA pipelines are not read
from the filesystem at run time: each is a registered object in the OpenHEXA database with
versions, and each version stores its own zipped copy of the code (`type: "zipFile"`). The
runner downloads that zip (see `Downloading pipeline...` in any run log), not anything from the
workspace bucket.

So phase 4 needs a real deployment call, not a file copy.

---

## 2. The mechanism, as implemented by `openhexa pipelines push`

Reference implementation: `openhexa/cli/api.py`, function `upload_pipeline()` (~line 690) and
`_build_pipeline_version_input()` (~line 298). Read it in an installed SDK:

```
/home/gpuntin/miniconda3/envs/snt_development/lib/python3.13/site-packages/openhexa/
```

Three steps:

### Step 1 — parse the pipeline's parameters

```python
from openhexa.sdk.pipelines.runtime import get_pipeline
parsed = get_pipeline(Path("snt_dhis2_extract"))   # directory, not file
parsed.parameters   # list[Parameter]; .to_dict() matches the GraphQL ParameterInput exactly
parsed.timeout
```

**This is AST-based, not import-based** (`openhexa/sdk/pipelines/runtime.py:225`, uses
`ast.parse`). That is the single most important finding in this document: a Workspace Manager
can parse all 20 `pipeline.py` files **without importing them**, so none of their dependencies
(`snt_lib`, `openhexa.toolbox`, …) need to be installed in the manager's own environment. The
obvious blocker for this whole approach turns out not to exist.

`Parameter.to_dict()` (`openhexa/sdk/pipelines/parameter/decorator.py:118`) emits exactly the
keys the API's `ParameterInput` accepts. Do not hand-build this dict — the backend rejects the
whole mutation on any unknown input field (the SDK source cites HEXA-1687 about precisely that).

### Step 2 — zip the pipeline directory

Only these suffixes are included, everything else is skipped:

```
.py  .ipynb  .txt  .md  .r  .sql
```

Paths inside the zip are relative to the pipeline directory, so `pipeline.py` sits at the zip
root. The zip is then base64-encoded into the `zipfile` input field.

### Step 3 — call the GraphQL mutation

Endpoint: `{HEXA_SERVER_URL}/graphql/`, header `Authorization: Bearer {HEXA_TOKEN}`.

| Mutation | Purpose | Input type |
|---|---|---|
| `createPipeline` | create a pipeline that does not exist yet | `CreatePipelineInput` |
| `uploadPipeline` | push a **new version** of an existing pipeline | `UploadPipelineInput` |

`UploadPipelineInput` (verified against `openhexa/graphql/graphql_client/input_types.py:919`):

```python
{
    "workspaceSlug": workspace.slug,   # required
    "code": "snt-dhis2-extract",       # kebab-case pipeline code, the deploy target
    "name": "v1.2.0",                  # version name
    "description": "...",              # shown in the version list
    "externalLink": None,              # link to the commit; what CI uses for traceability
    "zipfile": "<base64>",
    "parameters": [p.to_dict() for p in parsed.parameters],
    "timeout": parsed.timeout,
}
```

`CreatePipelineInput` takes `{workspaceSlug, name}` and optionally a nested `version`
(`CreatePipelineVersionInput`, same fields as above minus `workspaceSlug`/`code`).

### Credentials are already present inside a pipeline run

`HEXA_SERVER_URL` and `HEXA_TOKEN` are injected into every cloud pipeline run
(`openhexa/sdk/utils.py:90`, `openhexa/sdk/workspaces/current_workspace.py:45`). So a running
pipeline can call the API as itself, with **no CLI, no Docker, no stored API key** — subject to
the permission question below.

---

## 3. Open question — can a run push a pipeline version?

**This is the blocker. Do not design around an assumed answer.**

What is established:

| Capability | Result | Evidence |
|---|---|---|
| **READ** (`pipelines`, `pipelineByCode`) | **Allowed** | Probe logged the workspace pipeline count and successfully looked up a pipeline by code. |
| **CREATE** (`createPipeline`) | **Fails** | HTTP 200 with `{'message': 'An unknown error occurred.', 'path': ['createPipeline']}` — an unhandled server-side exception, *not* a clean `success: false` denial. Reproduced with input byte-identical to what the CLI sends (`{workspaceSlug, name}` only). |
| **UPLOAD** (`uploadPipeline`) | **Unverified — probably failed** | See below. |

On UPLOAD: probe v4 ran to completion (status `success`, 2026-09-16 13:47 UTC), but the target
pipeline `zz-deploy-target` was **still at version 1** afterwards. Since v4 wraps each check in a
non-fatal `run_check()`, a failed upload would be swallowed into a warning and the run would still
report success — which matches what we see. **But the run's `current_run` messages were never
read**, so the actual error is not yet known. That is the first thing to do next.

To read them: `get_pipeline(workspace_slug="snt-development-sandbox", pipeline_code=
"snt-deploy-probe", runs_per_page=1)` to get the run id, then `get_pipeline_run(run_id=...)`,
which returns the `messages` list. Note the container stdout shown in the OH run log does **not**
include `current_run.log_*` output — those are separate, and reading only stdout is what caused
confusion in this session.

### The fallback, already prepared

The user created a **CUSTOM connection named `oh`** in `snt-development-sandbox`
(slug `oh`, one secret field `token`, description "Token for pipelines to write pipelines"). If
the run token turns out to lack write scope, the Workspace Manager uses a workspace API key
instead:

```python
conn = workspace.custom_connection("oh")
headers = {"Authorization": f"Bearer {conn.token}"}
```

This is the same class of credential CI already uses for `openhexa pipelines push`, so it is not
a new trust assumption — just an explicit one.

### Why UPLOAD matters far more than CREATE

In a real country workspace all 20 SNT pipelines **already exist**. The Workspace Manager's job
is to push *new versions* of them — that is `uploadPipeline`. `createPipeline` is only needed to
bootstrap a pipeline that is not there yet, which is a once-per-pipeline event a human can do
from the UI. **If UPLOAD works and CREATE does not, the strategy still stands**, with a documented
manual bootstrap step. So establish UPLOAD first.

The `createPipeline` 500 is worth raising with the OpenHEXA devs regardless of the outcome: a
permission check should return a named error, not an unhandled exception. We cannot tell from the
client side whether it is a permission issue or a backend bug — that needs their server logs.

---

## 4. Gotchas that cost runs in this session

1. **`pipelines` is a root-level query, not a field on `workspace`.** `query { workspace(slug:) {
   pipelines } }` returns a bare HTTP 400 with no message. Correct form:
   `pipelines(workspaceSlug: $slug, page: 1, perPage: 1) { totalItems }`.
2. **The SDK's `graphql()` helper hides the reason for failures.** `openhexa.sdk.utils.graphql`
   calls `raise_for_status()` and discards the response body — which is exactly where GraphQL puts
   the error. Always POST manually and log `response.text` before raising. The probe's
   `call_graphql()` does this and should be reused.
3. **Do not pass `code` to `createPipeline`.** It is declared `Optional[str]` in the generated
   schema types, but the CLI never sends it and the resolver appears not to handle it. (Removing
   it did not fix the 500, so it was not *the* cause — but it is still a deviation from the
   reference implementation and should stay removed.)
4. **A pipeline run reporting `success` does not mean the work happened.** Always verify the
   effect independently — here, by re-reading the target pipeline's `currentVersion.versionNumber`.
5. **`@task` is not used anywhere in this repo.** The MCP `create_pipeline` tool's generic
   cheat-sheet suggests `@<pipeline_name>.task`; this codebase uses plain helper functions called
   from the `@pipeline` function. Follow the repo, not the tool hint (and see CLAUDE.md **R3**).

---

## 5. Artefacts left in the `snt-development-sandbox` OpenHEXA workspace

All disposable. None of this is in git yet.

| Object | Code / slug | State |
|---|---|---|
| Workspace Manager (phase 3/4 pull) | `snt-workspace-manager` | v1, file-sync only, **no deployment step yet**. Verified working for R + file copy. |
| Deployment probe | `snt-deploy-probe` | v4 `v4-capability-matrix`. Diagnostic only — delete once the question is settled. |
| Probe target | `zz-deploy-target` | v1. Exists only so the probe can attempt an upload against it. Delete with the probe. |
| Connection | `oh` (CUSTOM, secret field `token`) | Fallback credential. Keep. |

The probe's full source is not in git; the current version can be recovered with
`get_pipeline(workspace_slug="snt-development-sandbox", pipeline_code="snt-deploy-probe")`.

---

## 6. Suggested next steps, in order

1. **Read the v4 probe run's messages** and establish the UPLOAD answer (see §3). One tool call.
2. If UPLOAD is denied with the run token, re-test using the `oh` connection's token. That
   isolates *permission scope* from *mechanism* — if the connection token works, the mechanism is
   proven and only the credential source changes.
3. Once UPLOAD is proven by either credential, extend `snt_workspace_manager` with a deployment
   step: for each `<name>/pipeline.py` in the extracted release tarball, AST-parse it, zip the
   directory, and `uploadPipeline` under code `<name-with-hyphens>`. Use the release tag as the
   version `name` and the GitHub release URL as `externalLink`, so the OpenHEXA version list
   becomes a readable deployment history (the same reasoning as CLAUDE.md's note on
   `--description` / `--link`).
4. **Decide whether `pipeline.py` should still be copied into the workspace filesystem at all.**
   Once pipelines are deployed via `uploadPipeline`, the filesystem copy is redundant and
   arguably harmful: it looks authoritative but is not what runs. Options: stop copying it; or
   keep it read-only for diffing in phase 5. This needs a decision before phase 5 is designed,
   because it changes what the verification pipeline should hash.
5. Clean up the probe and target pipelines.

### One design consequence worth flagging early

This approach pushes pipeline **versions directly into each country workspace**, bypassing
OpenHEXA's Template system entirely — which is exactly the intent recorded in
[`release_strategy.md` §"Opting out of OpenHEXA's Template auto-update"](release_strategy.md).

It does, however, sit oddly beside CLAUDE.md **R5** ("Always publish from `snt-development`").
R5 exists because pushing *a template* from the wrong workspace creates a competing duplicate
template. Pushing a *pipeline version into the workspace that will run it* is a different
operation and does not create a template at all. R5 should be reworded rather than treated as
violated — but that is a real edit to CLAUDE.md and needs the team's agreement, not a drive-by
change.

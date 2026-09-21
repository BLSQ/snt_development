# Deploying an OpenHEXA pipeline from inside a pipeline run

Companion to [`release_strategy.md`](release_strategy.md), which explains *why* the Workspace
Manager deploys `pipeline.py` instead of copying it. This document is the *how*.

> Verified end to end 2026-09-16, implemented in `snt_workspace_manager` v3 in the
> `snt-development-sandbox` workspace.

## Why a file copy is not a deployment

Writing `snt_dhis2_extract/pipeline.py` into `workspace.files_path` produces a *file*, not a
runnable pipeline. OpenHEXA never reads pipelines from the filesystem at run time: each pipeline is
a registered object in the OpenHEXA database with versions, and **each version stores its own
zipped copy of the code** (`type: "zipFile"`). The runner downloads that zip — you can see
`Downloading pipeline...` in any run log — and ignores the workspace bucket entirely.

So deployment needs a real API call. No CLI and no Docker are required; it is three steps.

---

## The mechanism

Reference implementation: `openhexa/cli/api.py` → `upload_pipeline()` (~line 690) and
`_build_pipeline_version_input()` (~line 298), readable in any installed SDK under
`site-packages/openhexa/`.

### Step 1 — parse the pipeline's parameters

```python
from openhexa.sdk.pipelines.runtime import get_pipeline
parsed = get_pipeline(Path("snt_dhis2_extract"))   # directory, not file
parsed.parameters   # list[Parameter]; .to_dict() matches the GraphQL ParameterInput exactly
parsed.timeout
```

**This is AST-based, not import-based** (`openhexa/sdk/pipelines/runtime.py:225` uses `ast.parse`).
That is what makes the whole approach viable: the Workspace Manager parses all 20 `pipeline.py`
files **without importing them**, so none of their dependencies (`snt_lib`, `openhexa.toolbox`, …)
need to be installed in the manager's own environment.

Use `Parameter.to_dict()` (`openhexa/sdk/pipelines/parameter/decorator.py:118`) — it emits exactly
the keys `ParameterInput` accepts. Do not hand-build that dict: the backend rejects the entire
mutation on any unknown input field (the SDK source cites HEXA-1687 about precisely this).

### Step 2 — zip the pipeline directory

Only these suffixes are included; everything else is skipped:

```
.py  .ipynb  .txt  .md  .r  .sql
```

Paths inside the zip are relative to the pipeline directory, so `pipeline.py` sits at the zip root.
The zip is base64-encoded into the `zipfile` input field.

Note the zip carries the **whole directory**, not just `pipeline.py` — `snt_map_extracts` deploys
with `utils.py`, `worldpopclient.py`, the `malariaAtlasProject/` package, `readme.md` and
`requirements.txt`. This is the origin of the manifest gap described in
[`release_strategy.md`](release_strategy.md).

### Step 3 — call the GraphQL mutation

Endpoint `{HEXA_SERVER_URL}/graphql/`, header `Authorization: Bearer <token>` (which token matters
— see below).

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
(`CreatePipelineVersionInput` — same fields as above minus `workspaceSlug` / `code`). The nested
form creates a pipeline **and** its first version atomically, so bootstrapping an empty workspace
needs no manual UI step.

The **pipeline code** is the directory name with `_` → `-`. Cross-checked against all 20
`push_snt_*.yaml` workflows: 20/20 match the `--code` slug CI already uses.

---

## Authentication: use the `oh` connection token, not the run's token

`HEXA_SERVER_URL` and `HEXA_TOKEN` are injected into every cloud pipeline run
(`openhexa/sdk/utils.py:90`, `openhexa/sdk/workspaces/current_workspace.py:45`), so it is tempting
to call the API as the run itself. **That does not work for deployment.**

| Credential | `uploadPipeline` | `createPipeline` |
|---|---|---|
| The run's own `HEXA_TOKEN` | `PERMISSION_DENIED` | opaque HTTP 500 |
| Workspace API token from the `oh` connection | allowed | allowed (bare and nested) |

It is a credential-*scope* problem, not a mechanism problem: the identical payload succeeds or
fails purely on the header. Both tokens are 95 characters, so they are indistinguishable by shape.

In practice, a **CUSTOM connection named `oh`** (slug `oh`, one secret field `token`):

```python
token = workspace.custom_connection("oh").token
headers = {"Authorization": f"Bearer {token}"}
```

This is the same class of credential CI already uses for `openhexa pipelines push` — not a new
trust assumption, just an explicit one. But it is **workspace-scoped**: deploying into a country
workspace means that workspace holding such a connection. Who mints it, where it is stored and how
it is rotated is the one genuinely new operational requirement this design adds, and it is still
undecided.

---

## Gotchas

1. **`pipelines` is a root-level query, not a field on `workspace`.** `query { workspace(slug:) {
   pipelines } }` returns a bare HTTP 400 with no message. Correct form:
   `pipelines(workspaceSlug: $slug, page: 1, perPage: 1) { totalItems }`.
2. **The SDK's `graphql()` helper hides the reason for failures.** `openhexa.sdk.utils.graphql`
   calls `raise_for_status()` and discards the response body — which is exactly where GraphQL puts
   the error. POST manually and log `response.text` before raising.
3. **Do not pass `code` to `createPipeline`.** It is declared `Optional[str]` in the generated
   schema types, but the CLI never sends it and the resolver appears not to handle it.
4. **`createPipeline` returns an opaque 500 on a permission failure**, where `uploadPipeline`
   returns a clean `PERMISSION_DENIED`. If you get `{'message': 'An unknown error occurred.'}`,
   suspect the token before anything else. (Worth reporting to the OpenHEXA devs.)
5. **`default=""` is rejected by the SDK.** A `str` parameter with an empty-string default raises
   `ParameterValueError("Empty values are not accepted.")` at parse time — meaning the *deploy*
   fails, not the run. Use `default=None`. Catch it locally by running `get_pipeline(Path(...))`
   before pushing; that is the same AST parse the backend performs, so it is worth doing after any
   parameter edit.
6. **A pipeline run reporting `success` does not mean the work happened.** Verify the effect
   independently — re-read the target pipeline's `currentVersion.versionNumber`.
7. **`@task` is not used anywhere in this repo.** The MCP `create_pipeline` tool's generic
   cheat-sheet suggests `@<pipeline_name>.task`; this codebase uses plain helper functions called
   from the `@pipeline` function. Follow the repo, not the tool hint (CLAUDE.md **R3**).
8. **A version's `files` list is the ground truth for what was deployed.** `get_pipeline` returns
   the full zip contents, so you can hash the deployed `pipeline.py` and compare it to the release
   manifest — that is how byte-identity was confirmed, and it is the basis for the verification
   pipeline.

---

## What has been proven, and what has not

Verified: `snt_workspace_manager` v3 bootstrapped `snt_dhis2_extract` and `snt_map_extracts` into
`snt-development-sandbox` from release `v0.0.1-test`. Correct codes; parameters round-tripped
through `Parameter.to_dict()` including the `dhis2_connection` connection-typed parameter; the
deployed `snt_dhis2_extract/pipeline.py` read back at sha256 `44290bd9…d77755cd`, **byte-identical
to the release manifest's hash.**

> `v0.0.1-test` was deleted in the 2026-09-21 sandbox reset (`release_strategy.md` §"Sandbox
> reset"). The result above stands as a record; re-running it means re-running against the new
> fixture series, which starts at `v0.1.0-test`.

Not verified:

* **The remaining 18 pipelines.** Only 2 of 20 have been through the deployer.
* **Whether `externalLink` is stored.** It is sent in the payload, but the MCP `get_pipeline` query
  does not select that field, so its absence from the response proves nothing either way. Check the
  OpenHEXA UI's version list.

---

## Open items

1. **Commit `snt_workspace_manager` to this repo.** It currently exists *only* as a version in the
   sandbox workspace — exactly the "no version-propagation story" problem this project is meant to
   fix. It needs `snt_workspace_manager/pipeline.py` + `requirements.txt` + `readme.md` per the
   "Adding or changing a pipeline" checklist, minus a `push_*.yaml` workflow until item 4 is
   settled. **This is the main outstanding gap.**
2. **Run the full 20-pipeline bootstrap** once, to confirm nothing in the other 18 trips the
   deployer.
3. **Close the manifest gap** — see [`release_strategy.md`](release_strategy.md). Until the manifest
   covers `requirements.txt`, `readme.md` and helper modules, verification cannot check most of
   what is actually deployed.
4. **Decide where the `oh` token comes from in a country workspace** (minting, storage, rotation).

### R5 needs rewording, not violating

This approach pushes pipeline **versions directly into each country workspace**, bypassing
OpenHEXA's Template system — which is the stated intent of the release strategy.

That sits oddly beside CLAUDE.md **R5** ("Always publish from `snt-development`"). R5 exists
because pushing *a template* from the wrong workspace creates a competing duplicate template.
Pushing a *pipeline version into the workspace that will run it* is a different operation and
creates no template at all. R5 should be reworded rather than treated as violated — a real edit to
CLAUDE.md, needing the team's agreement, not a drive-by change.

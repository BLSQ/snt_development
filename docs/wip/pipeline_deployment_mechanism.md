# How OpenHEXA pipelines are actually deployed — findings

Companion to [`release_strategy.md`](release_strategy.md). Written for a fresh agent session
picking up phases 3–4 of the release roadmap.

**Status: resolved and proven end to end, 2026-09-16.** The blocking question is answered — a run's
own token cannot deploy, a workspace API token supplied through a connection can. See
[The credential answer](#3-the-credential-answer--settled). The mechanism is implemented in
`snt_workspace_manager` v3 and verified against two real SNT pipelines.

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

## 3. The credential answer — settled

**It is a credential-scope question, not a mechanism question.** Probe v5 ran the *identical*
upload twice in one run, changing only the `Authorization` header:

| Credential | `uploadPipeline` | Evidence |
|---|---|---|
| Run's own `HEXA_TOKEN` | **DENIED** | `errors: ['PERMISSION_DENIED']` — a clean, named refusal. |
| Workspace API token from the `oh` connection | **ALLOWED** | Registered version 2 of `zz-deploy-target`; independently confirmed by re-reading the pipeline. |

Both tokens are 95 characters, so they are indistinguishable by shape — only by scope.

Probe v6 then retested `createPipeline` with the connection token, since the earlier opaque 500
had only ever been seen with the run token:

| Form | Result |
|---|---|
| `createPipeline {workspaceSlug, name}` | **ALLOWED** |
| `createPipeline {workspaceSlug, name, version {...}}` | **ALLOWED** — pipeline and its first version in one atomic call |

So the earlier `{'message': 'An unknown error occurred.'}` **was the permission failure surfacing
as an unhandled server-side exception.** Worth reporting to the OpenHEXA devs — a permission check
should return `PERMISSION_DENIED` the way `uploadPipeline` does, not a 500 — but it is no longer a
blocker for us.

**Consequence: empty-workspace bootstrap can be fully automated.** The manual once-per-pipeline UI
step anticipated in the earlier draft of this document is not needed.

### The credential, in practice

A **CUSTOM connection named `oh`** in `snt-development-sandbox` (slug `oh`, one secret field
`token`). The Workspace Manager reads it as:

```python
token = workspace.custom_connection("oh").token
headers = {"Authorization": f"Bearer {token}"}
```

This is the same class of credential CI already uses for `openhexa pipelines push`, so it is not a
new trust assumption — just an explicit one. Note it is a **workspace-scoped** token: deploying to
a country workspace means that workspace holding such a connection, which is a real operational
question for the rollout (who mints it, where it is stored, how it is rotated).

### Verified end to end

`snt_workspace_manager` v3 bootstrapped `snt_dhis2_extract` and `snt_map_extracts` into
`snt-development-sandbox` from release `v0.0.1-test`:

* Both pipelines were created with codes `snt-dhis2-extract` / `snt-map-extracts`, matching the
  `--code` slugs this repo's CI uses. The slug rule (`_` → `-`) was cross-checked against all 20
  `push_snt_*.yaml` workflows: **20/20 match**.
* Parameters round-tripped through `Parameter.to_dict()`, including the `dhis2_connection`
  connection-typed parameter.
* The deployed `snt_dhis2_extract/pipeline.py` read back with sha256
  `44290bd9…d77755cd` — **byte-identical to the release manifest's recorded hash.**
* The zip carried the **whole pipeline directory**, not just `pipeline.py`: `snt_map_extracts`
  arrived with `utils.py`, `worldpopclient.py`, the `malariaAtlasProject/` package, `readme.md`
  and `requirements.txt`. This matters — see the manifest gap in
  [`release_strategy.md`](release_strategy.md).

**Not verified:** whether `externalLink` is stored. It is sent in the payload, but the MCP
`get_pipeline` query does not select that field, so its absence from the response is not evidence
either way. Check it in the OpenHEXA UI's version list.

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
6. **`default=""` is rejected by the SDK.** A `str` parameter with an empty-string default raises
   `ParameterValueError("Empty values are not accepted.")` at parse time, which in a workspace
   means the deploy fails rather than the run. Use `default=None`. Caught locally by running
   `get_pipeline(Path(...))` before pushing — worth doing for every pipeline edit, since it is the
   same AST parse the backend performs.
7. **A version's `files` list is the ground truth for what was deployed.** `get_pipeline` returns
   the full zip contents, so you can hash the deployed `pipeline.py` and compare it to the release
   manifest. That is how the byte-identity check above was done, and it is the obvious basis for
   the phase 5 verification pipeline.

---

## 5. Artefacts left in the `snt-development-sandbox` OpenHEXA workspace

All disposable. None of this is in git yet.

| Object | Code / slug | State |
|---|---|---|
| Workspace Manager | `snt-workspace-manager` | **v3 `v3-scoped-deploy`** — analytics sync + pipeline deployment + dry run + `only_pipelines`. Keep. |
| Connection | `oh` (CUSTOM, secret field `token`) | The deployment credential. Keep. |
| Deployment probe | `snt-deploy-probe` | v6. Diagnostic only, question now settled — **delete.** |
| Probe target | `zz-deploy-target` | v2. **Delete** with the probe. |
| Probe leftovers | `zz-created-bare`, `zz-created-nested` | Created by probe v6 to test bootstrap. **Delete.** |
| Bootstrap test output | `snt-dhis2-extract`, `snt-map-extracts` | Real pipelines at `v0.0.1-test`, created by the scoped bootstrap test. Keep or delete depending on whether the sandbox is kept. |

Deleting pipelines is a UI action — this repo's rules keep agents away from destructive operations,
so do it by hand.

Neither probe's source is in git; any version can be recovered with
`get_pipeline(workspace_slug="snt-development-sandbox", pipeline_code="snt-deploy-probe")`.
The Workspace Manager's source is not in git either — **that is the main outstanding gap.**

---

## 6. Next steps, in order

Steps 1–4 of the earlier list are **done**: the credential question is settled, the deployment step
is built and proven, and the filesystem-copy decision is made (see below). What remains:

1. **Commit `snt_workspace_manager` to this repo.** It exists only as a version in the sandbox
   workspace, which is exactly the "no version-propagation story" problem this whole project is
   meant to fix. It needs a home (`snt_workspace_manager/pipeline.py` + `requirements.txt` +
   `readme.md` per the "Adding or changing a pipeline" checklist), minus a `push_*.yaml` workflow
   until the R5 question below is settled.
2. **Run the full 20-pipeline bootstrap** once, to confirm nothing in the remaining 18 trips the
   deployer. The scoped test covered 2 of 20.
3. **Close the manifest gap** (see [`release_strategy.md`](release_strategy.md)): the manifest
   tracks only `<name>/pipeline.py`, but deployment ships the whole directory. Until the manifest
   covers `requirements.txt`, `readme.md` and helper modules, phase 5 cannot verify most of what
   is actually deployed.
4. **Decide where the `oh` token comes from in a country workspace** — who mints it, how it is
   stored and rotated. This is the one genuinely new operational requirement the design adds.
5. **Clean up the probe and throwaway pipelines** (§5).

### Decided: `pipeline.py` is no longer copied into the workspace filesystem

Confirmed with Giulia, 2026-09-16. Deployment happens through the API; the filesystem copy is
inert, looks authoritative, and is precisely the confusion that made phase 3 look finished when it
was not. `split_manifest()` in the Workspace Manager routes `<name>/pipeline.py` entries to the
deployer and everything else to the filesystem sync.

For phase 5 this means the verification pipeline has **two sources to hash, not one**:

| What runs | Where it lives | How phase 5 checks it |
|---|---|---|
| R analytics | workspace filesystem | hash the file, compare to the manifest |
| Python orchestration | the pipeline version's stored zip | read it back via `get_pipeline`, hash `pipeline.py`, compare to the manifest |

The second is proven to work — the byte-identity check in §3 is exactly that comparison.

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

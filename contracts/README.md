# `contracts/` — the files this repo exchanges with other tools

A contract is a file whose shape another tool depends on. One subfolder per contract kept in this
repo. Each holds that file's JSON Schema (the source of truth for its shape) or a skeleton, a
canonical valid example, any reference instances worth keeping, and a `README.md` carrying the
reasoning, the conventions and the open questions.

| Contract | What | Producer | Consumer(s) | Schema / reference | Status |
|---|---|---|---|---|---|
| `SNT_config.json` | Per-workspace configuration, one country | Operator, by hand (config webapp in future) | Every pipeline | [`snt_config_json/`](snt_config_json/README.md) — skeleton; `snt_config.schema.json` `[TODO]` | Skeleton current; schema not written |
| `SNT_metadata.json` | SNT Explorer data-layer catalogue, one country | [`metadata/SNT_metadata_master.json`](../metadata/README.md) → `snt_metadata_json_trim` | SNT Explorer (IASO) | [`snt_metadata_json/`](snt_metadata_json/README.md) | **Work in progress**, being validated with the IASO developers |
| `release_manifest.json` | Files and hashes shipped by a release | [`.github/workflows/generate_manifest.yaml`](../.github/workflows/generate_manifest.yaml) | `snt_workspace_bootstrap` deployer and checker | Owned by that repo: [`docs/contracts/release_manifest.schema.json`](https://github.com/BLSQ/snt_workspace_bootstrap/blob/main/docs/contracts/release_manifest.schema.json) | Stable (rule R22) |
| Per-dataset outputs | Files, columns and `YEAR` per OpenHEXA dataset | Each `pipeline.py` | Downstream pipelines, SNT Explorer, master list | — | **Planned**, not started |

**These schemas describe target shapes, not necessarily what is deployed.** Read the subfolder's
`README.md` before assuming a field's meaning. Older files that nothing reads any more are archived
in [`deprecated/configuration/`](../deprecated/configuration/README.md).

**A new subfolder needs nothing in `.gitignore`.** The repo ignores `*.json` wholesale; the negation
`!contracts/**/*.json` already reaches into any subfolder here. Do not add a per-folder negation,
and do not weaken it to a single `*` — that would stop matching subfolders and silently untrack every
schema below this one.

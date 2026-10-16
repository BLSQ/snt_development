# 0006. Retire `snt_assemble_results`; the Explorer reads datasets directly

Date: 2026-10 (PR #144)
Status: Accepted

## Context

`snt_assemble_results` (stage E) flattened every stage's output into one row per ADM2 and published
it to `SNT_RESULTS` for the SNT Explorer. That step:

- was **lossy**: it aggregated over time and dropped `YEAR`, so the Explorer could not show a layer
  over time;
- **silently dropped** any column not declared in the old-format `SNT_metadata.json`, which it used as
  a gate, and wrote `PERIOD` back into that file at runtime;
- duplicated what already existed: the per-stage OpenHEXA datasets were already the contract
  ([0001](0001-no-dag-engine.md)).

## Decision

Retire `snt_assemble_results`. The SNT Explorer imports each data layer **directly from its per-stage
OpenHEXA dataset**. The layers are listed in `catalogs/SNT_metadata_master.json` and trimmed per
workspace by `snt_metadata_json_trim`, in the format of [0007](0007-snt-metadata-json-format.md).

## Consequences

- The code is archived in `deprecated/DEPRECATED_snt_assemble_results/`, CI no longer deploys it, it
  ships in no release, and its OpenHEXA template is deleted.
- `SNT_RESULTS` is removed from `SNT_config.skeleton.json`. It remains in the archived per-country
  configs and in live workspace configs, where nothing reads it. The dataset survives in workspaces
  that ran the pipeline, frozen at its last run.
- Every output table now needs a `YEAR` column to be shown over time (rule R21).
- The old-format `SNT_metadata.json` lost its pipeline reader; the repo copy is archived in
  `deprecated/configuration/`.

Source: [`DATA_ARCHITECTURE.md` §3.2 Stage E](../DATA_ARCHITECTURE.md#32-stages).

---
name: write-pipeline-readme
description: Write, update or audit a pipeline's readme.md in snt_development against docs/PIPELINE_README_STANDARD.md, verifying every claim against the code. Use when a pipeline.py's parameters, outputs or publication change, when asked to write/fix/check a pipeline readme, or when the handover skill reports one may be stale.
---

# Write (or audit) a pipeline readme

The standard is [`docs/PIPELINE_README_STANDARD.md`](../../../docs/PIPELINE_README_STANDARD.md).
**Read it in full first** — this skill is only the order of work. Do not copy its rules here.

## Steps

1. **Read the code, in the standard's §0 order** — `<name>/pipeline.py`, then
   `pipelines/<name>/code/*.ipynb`, then `pipelines/<name>/utils/*.r`, plus any shared helper from
   `code/*.r` the notebook calls (e.g. `load_dataset_file`, `init_snt_workspace` in
   `code/snt_utils.r`). The report notebook and `utils/*_report.r` do not define the published
   contract — read them only to verify filesystem outputs the readme attributes to reporting. The
   existing readme is supplementary: it may enrich wording, never serve as evidence.
2. **Collect the facts** (in your working notes, not a file) before writing anything:
   - every `@parameter(...)`: code, `name=`, type, required, default, **every** `choices` value;
   - the pipeline body in order, with each `if` / `return` / `dataset_file_exists` guard (skips) —
     including the **report-only branch**: what runs, what is published, and what data the report
     reads when computation is skipped;
   - every input: `get_file_from_dataset` / `dataset_file_exists` / `load_configuration_snt` calls,
     and the `config_json$SNT_DATASET_IDENTIFIERS$…` ids the notebook resolves;
   - every file written — any write call (`write_parquet`, `to_parquet`, `write_csv`, `fwrite`,
     `write.csv`, `st_write`, `ggsave`, `export_data`, …);
   - **only** the paths passed to `add_files_to_dataset(...)` — the published set;
   - resolution from the `group_by(...)` / `by = .(…)` aggregation keys, not from prose.
3. **Write** to the standard's §1 structure and §2 section rules (template in §6). Keep the section
   order. Separate *workspace filesystem* from *published to dataset* in Outputs.
4. **Verify** every claim against the standard's **§3 checklist**, row by row. For the *Notebook
   parameters* row, run `python3 .claude/skills/handover/check_notebook_params.py <name>` (from the
   repo root). A claim that rests on `snt_lib` internals (`save_pipeline_parameters`,
   `check_outputs_generated`, …) is **not verifiable** unless you read the upstream repo — say so.
5. **Report discrepancies.** List every place where a comment, docstring, the old readme or another
   doc (notably [`docs/DATA_ARCHITECTURE.md` §3](../../../docs/DATA_ARCHITECTURE.md#3-data-lineage))
   disagrees with the code — the readme documents the code; the list goes in the PR description
   (standard §0). Don't fix code or other docs in the same pass.

## Audit mode

When asked only to *check* a readme: do steps 1, 2, 4 and 5 against the existing file, without
editing it, and report per §3 row — **matches / differs (what the code does, `file:line`) / not
verifiable**.

## Scope

A readme change belongs in the same PR as the `pipeline.py` change it describes (**R16**). Changing
only a readme does not deploy anything.

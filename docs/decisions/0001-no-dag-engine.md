# 0001. No DAG engine or scheduler: pipelines are launched by hand

Date: not recorded
Status: Accepted

## Context

The SNT system is about 20 OpenHEXA pipelines that feed each other through OpenHEXA datasets. How
they are used rules out a fixed dependency graph:

- Pipelines are **re-run independently with different parameters**.
- **Alternative methods deliberately overwrite each other** (five outlier-imputation pipelines, two
  reporting-rate pipelines), and the analyst picks one by running it last
  ([0002](0002-alternative-methods-last-run-wins.md)).
- An operator may **supply their own input** (population, care-seeking, facility locations) instead
  of running the upstream pipeline.

A DAG engine would fight all three.

## Decision

No scheduler, no DAG engine, no Airflow or dbt. Each pipeline is a standalone OpenHEXA pipeline,
launched by hand from the OpenHEXA UI, one at a time. Pipelines couple only through OpenHEXA
datasets, identified by logical keys in `SNT_config.json`. A downstream pipeline knows a dataset key
and a filename, never which pipeline produced them.

## Consequences

- Nothing enforces run order. Run order is a strong suggestion, not a dependency, and the
  operator-facing order is not written down yet ([BACKLOG DOC-1](../BACKLOG.md#doc-1)).
- A results table can mix data vintages, e.g. incidence from January's imputation run joined to
  reporting rates from March's routine data. Never assume an upstream ran today.
- Provenance rests on the `{CC}_parameters.json` published beside each output and on the OpenHEXA
  dataset version names.
- Some pipelines that look like independent roots are not: the external-source pipelines need
  `{CC}_shapes.geojson` from `snt_dhis2_formatting` first.

Sources: `CLAUDE.md` ("Why no DAG engine"); [`DATA_ARCHITECTURE.md` §1, §4.1](../DATA_ARCHITECTURE.md#4-orchestration-model).

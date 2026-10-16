# 0002. Alternative methods overwrite each other: last run wins

Date: not recorded
Status: Accepted

## Context

Several pipelines are interchangeable methods for the same step:

- five outlier-imputation pipelines (`iqr`, `mean`, `median`, `path`, `magic_glasses`), all writing
  `{CC}_routine_outliers_{detected,removed,imputed}.parquet` to `DHIS2_OUTLIERS_IMPUTATION`;
- two reporting-rate pipelines (`_dataelement`, `_dataset`), both writing to `DHIS2_REPORTING_RATE`.

The analyst runs several methods on the same data, compares the reports, settles on one, and moves
on. Downstream pipelines should not need to know which method was chosen.

## Decision

Alternative methods write **the same filenames to the same dataset**. Whatever ran last is what
downstream consumes. This is the override mechanism, not a collision. Do not rename outputs per
method.

## Consequences

- Downstream pipelines stay method-agnostic. Renaming outputs per method would break the override and
  force every consumer to choose a method.
- The file carries no method label. To know which method produced it, read the `{CC}_parameters.json`
  published beside it in the same dataset version, or the dataset version name. Never infer the
  method from the filename.
- The same runs also overwrite the workspace DB table `outliers_detected`, which has no parameters
  file beside it, so it loses provenance on every run. Open: [BACKLOG PIPE-4](../BACKLOG.md#pipe-4).

Sources: `CLAUDE.md` *Traps*; [`DATA_ARCHITECTURE.md` §3.2 Stage C](../DATA_ARCHITECTURE.md#32-stages).

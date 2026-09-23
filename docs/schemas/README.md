# `docs/schemas/` — schemas and conventions for the SNT configuration files

One subfolder per documented file. Each holds that file's JSON Schema (the source of truth for its
shape), a canonical valid example, any reference instances worth keeping, and a `README.md` carrying
the reasoning, the conventions and the open questions.

| Subfolder | Documents | Status |
|---|---|---|
| [`snt_metadata_json/`](snt_metadata_json/README.md) | `SNT_metadata.json` — the catalogue of data layers the SNT Explorer offers for one country | **work in progress**, 2026-09-08 shape, being validated with the IASO developers |

Planned, not written yet: `snt_config_json/` for `SNT_config.json`.

**These schemas describe target shapes, not what is deployed.** The files that run today live in
[`configuration/`](../../configuration/) and are, for `SNT_metadata.json`, still the older format.
Read the subfolder's `README.md` before assuming a field's meaning.

**A new subfolder needs nothing in `.gitignore`.** The repo ignores `*.json` wholesale; the negation
`!docs/schemas/**/*.json` already reaches into any subfolder here. Do not add a per-folder negation,
and do not weaken it to a single `*` — that would stop matching subfolders and silently untrack every
schema below this one.

# 🦟 SNT — Subnational Tailoring pipelines

Data pipelines that turn a country's routine health data into the evidence base for **subnational
tailoring (SNT) of malaria interventions** — deciding which interventions to deploy where, at a
sub-national level.

They take monthly malaria case data from a country's **DHIS2** health information system and
combine it with rainfall, population, malaria-prevalence and survey data from external sources.
Each pipeline produces one or more **data layers**: tables of indicators — incidence, reporting
completeness, seasonality, quality of care — for every administrative unit in the country, plus
the plots and HTML reports that go with them. Which administrative level that is varies by
country; throughout this repository it is called **`ADM2`**, and what it means on the ground
(district, health zone, commune…) depends on the country's own hierarchy.

The layers are the deliverable. They can be used as they are, or loaded into the
[SNT Explorer](https://www.snt-toolbox.org/snt-explorer/)
([docs](https://docs.snt-toolbox.org/en/a-propos-du-snt-explorer.html)), a separate tool where
national malaria programmes explore the results and build intervention scenarios.

Everything here runs on **[OpenHEXA](https://openhexa.org)**, Bluesquare's open-source data
platform. This repository is the **source** for ~20 pipelines; OpenHEXA publishes them as a
*template* that each country's workspace subscribes to.

---

## 🗺️ How it works, in one picture

```mermaid
flowchart TD
    DHIS2["DHIS2 API<br/>routine malaria data"] --> A
    A["A. snt_dhis2_extract<br/>download raw data"] --> B
    B["B. snt_dhis2_formatting<br/>raw into SNT schema<br/>the hinge of the system"]
    B --> C["C. clean and adjust<br/>outlier imputation x5<br/>population transformation"]
    B -->|ADM2 shapes| AP["A2. external sources<br/>ERA5 rainfall, WorldPop, MAP<br/>DHS surveys, healthcare access"]
    SRC["Copernicus, WorldPop<br/>Malaria Atlas Project, DHS"] --> AP
    B --> D
    C --> D["D. compute indicators<br/>incidence, reporting rate x2<br/>quality of care, seasonality x2"]
    AP --> D
    D --> OUT["data layers, one or more tables per pipeline<br/>SNT Explorer, exported tables, plots, reports"]
```

Each box is one or more standalone pipelines, **launched by hand** from the OpenHEXA UI. They pass
data to each other through OpenHEXA **datasets** — never by calling each other directly. Stages
`C` and `A2` publish usable layers of their own, not just intermediates, and the Explorer reads
those datasets directly. (A former stage `E`, `snt_assemble_results`, flattened everything into one
table per ADM2; it is retired and kept under [`deprecated/`](deprecated/).)

Some pipelines also create **blank CSV templates** in the workspace `uploads/` folder, for an
operator to fill in and upload to a later pipeline. `snt_dhis2_formatting`, for example, writes
population templates with one row per ADM2. These files are not published to any dataset. See
that pipeline's `readme.md`.

The full pipeline-by-pipeline inventory is in
[`docs/DATA_ARCHITECTURE.md` §1.1](docs/DATA_ARCHITECTURE.md#11-the-pipelines-at-a-glance), where
the stage letters above are used throughout (stage `A2` here is written `A′` there).

---

## 📂 What's in this repo

| Path | What it is |
|---|---|
| `<pipeline_name>/pipeline.py` | Orchestration, Python. **Deployed automatically** by CI on merge to `main`. |
| `<pipeline_name>/requirements.txt` | That pipeline's dependencies. Deployed with it. |
| `<pipeline_name>/readme.md` | The pipeline's user-facing contract — read this before running it. |
| [`pipelines/`](pipelines/) | **The analytics: R notebooks and helpers.** *Not* deployed by CI — see below. |
| [`code/`](code/) | Shared R library used across pipelines (`snt_utils`, `snt_report`, `snt_palettes`). |
| [`contracts/`](contracts/README.md) | Contracts between this repo and its consumers: the `SNT_config.json` skeleton, the `SNT_metadata.json` schema, … The live config lives in each workspace. |
| [`docs/`](docs/) | Architecture, glossary, standards, ownership, the backlog of open items, and decision records. |
| [`dev/`](dev/) | Optional local Python tooling. Never runs on OpenHEXA. |
| [`.claude/`](.claude/) | Guardrails for AI coding agents. Active on clone. |
| [`deprecated/`](deprecated/) | History. Never use as a template. |

---

## 🧭 Start here

Pick the row that matches what you came for.

| I want to… | Read |
|---|---|
| understand the system in 10 minutes | [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md) §1–2 |
| know what a term means — `N1`, `PfPR`, `CSB`, ADM2, "Magic Glasses" | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |
| run a pipeline, or know exactly what it produces | that pipeline's own `readme.md` |
| follow the data end to end | [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md) §3 |
| read the epidemiological method | the R notebooks in `pipelines/<name>/code/*.ipynb`, with [`docs/GLOSSARY.md`](docs/GLOSSARY.md) §4 open beside them |
| **change any code** | [`CLAUDE.md`](CLAUDE.md) — start at the [rule Register](CLAUDE.md#register) |
| write or update a pipeline `readme.md` | [`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md) |
| add or edit an SNT Explorer data layer in `SNT_metadata.json` | [`contracts/snt_metadata_json/README.md`](contracts/snt_metadata_json/README.md) |
| know who to ask about a pipeline | [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md) |
| set up locally | [`dev/README.md`](dev/README.md) |

---

## ⚠️ Non-obvious things

Facts that are not visible from the code and that newcomers reliably get wrong. Each is
deliberate; none is a bug to fix.

1. **These pipelines are not a DAG.** Nothing schedules them; a person presses Run, one at a time,
   so a results table can mix data vintages. → [§4](docs/DATA_ARCHITECTURE.md#4-orchestration-model)
2. **The Python and R halves ship by different routes.** Merging a notebook deploys **nothing** —
   it reaches a workspace only via a run with **`Pull scripts` = ON**.
   → [`CLAUDE.md` rule 2](CLAUDE.md#the-five-rules-that-matter-most)
3. **A country may not be running the notebook you are looking at.** A `<generic_name>_<CC>.ipynb`
   variant runs *instead*, in that one country's workspace, and no mechanism propagates it.
   → [`pipelines/CLAUDE.md`](pipelines/CLAUDE.md#country-specific-notebook-variants)
4. **OpenHEXA datasets are the contract, not the filesystem.** A file not registered with
   `add_files_to_dataset(...)` is invisible downstream. → [R4](docs/CONVENTIONS.md#r4)
5. **Some pipelines deliberately overwrite each other** — alternative methods, last run wins.
   → [Traps](CLAUDE.md#traps)
6. **There is no test suite and no CI lint.** `ruff` is the only automated check and nothing runs
   it for you. → [`CLAUDE.md`](CLAUDE.md#commands)

---

## 🛠️ Working on this repo

### 🔑 Before you start

- Access to the relevant **OpenHEXA workspace with the `Admin` role** — ask your team lead.
  Everything real runs there: the data, the datasets, the R kernel.
- A local clone of this repo. Nothing else is required to *read* it.
- Optional but recommended: the local Python tooling (`ruff`, `nbstripout`) — see
  [`dev/README.md`](dev/README.md).

### 🗓️ Your first day

1. Read [`docs/DATA_ARCHITECTURE.md`](docs/DATA_ARCHITECTURE.md) §1–2, then skim
   [`docs/GLOSSARY.md`](docs/GLOSSARY.md).
2. Read [`CLAUDE.md`](CLAUDE.md) — the [Register](CLAUDE.md#register) is the whole rulebook in one
   table. It applies to humans and AI agents alike.
3. Open the OpenHEXA workspace and run one pipeline end to end, reading its `readme.md` first.
4. Set up locally: `conda env create -f dev/environment.yml`.
5. Take a small ticket. Ask your pipeline's contact in [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md)
   before changing anything in `pipelines/*/code/`.

### 🤖 Working with an AI coding assistant

Assumed, not merely tolerated — most of the documentation in this repo is written to be read by a
person *or* an agent.

- **[`CLAUDE.md`](CLAUDE.md) is the rulebook, and it is not Claude-specific.** [Claude
  Code](https://claude.com/claude-code) loads it automatically; with any other assistant (Cursor,
  Copilot, Gemini, …) paste it in or point the tool at it. The
  [Register](CLAUDE.md#register) is all 22 rules in one table — the cheapest thing to give a model
  that is about to touch this repo.
- **[`.claude/`](.claude/) ships live guardrails.** They are active the moment you clone: a hook
  that blocks destructive and history-rewriting `git` commands, plus a matching deny list. If an
  agent gets blocked, that is the guardrail working — don't route around it.
  → [`.claude/README.md`](.claude/README.md)
- **[`.claude/skills/`](.claude/skills/) holds three task skills** — `handover` (runs the
  handover checklist, including a script that checks injected notebook parameters against their
  `exists()` fallbacks), `write-pipeline-readme` and `add-explorer-layer` (with the metadata
  validator). Claude Code offers them automatically; with another assistant, point it at the
  `SKILL.md`. The scripts are plain Python and work without any assistant.
- **[`pipelines/CLAUDE.md`](pipelines/CLAUDE.md) holds the notebook-only rules.** Claude Code loads
  it only when working under `pipelines/`; with another assistant, give it that file too when the
  task touches a notebook.
- **[`docs/PIPELINE_README_STANDARD.md`](docs/PIPELINE_README_STANDARD.md) is deliberately
  tool-agnostic** — it works as instructions for a human or as a prompt for any model.
- **An agent's output still needs the domain review below.** An assistant cannot tell you that an
  epidemiological number moved in the wrong direction, and neither can anything else in this repo.
  Nothing here changes who has to approve a change to the analytics.

### 🔀 How changes get in

**Every change to `main` goes through a pull request, and every PR should be reviewed by a team
member** — for analytics changes, by the person listed for that pipeline in
[`docs/OWNERSHIP.md`](docs/OWNERSHIP.md). Core data-processing logic (the R notebooks and `code/`)
is not something to change unreviewed, however small the diff looks: nothing downstream will tell
you that a number moved.

Four things that are never OK:

- **Committing country data.** No `.csv` / `.xlsx` / `.parquet` / `.rds` / `.geojson` with real
  values. This is health data for real districts. `.gitignore` blocks most of it — never
  `git add -f` past it.
- **Committing notebook outputs.** Strip them first (`nbstripout`).
- **Putting business logic in `pipeline.py`.** It orchestrates; the analytics live in R.
- **Publishing from a workspace other than `snt-development`.** It creates a duplicate template
  nobody is subscribed to. → [why](docs/CONVENTIONS.md#r5)

Before opening a PR, walk the
[handover checklist](CLAUDE.md#handover-checklist) — it is short and it maps to the rules.

---

## 📦 Release management

Each GitHub release of this repo gets a `release_manifest.json`, built by
[`.github/workflows/generate_manifest.yaml`](.github/workflows/generate_manifest.yaml): every file
the release ships, with its hash. The tools that deploy a release into a workspace and check it
against one live in their own repo,
[`BLSQ/snt_workspace_bootstrap`](https://github.com/BLSQ/snt_workspace_bootstrap).

The manifest's shape is a **contract** between the two repos, defined by
[`release_manifest.schema.json`](https://github.com/BLSQ/snt_workspace_bootstrap/blob/main/docs/contracts/release_manifest.schema.json)
there. Do not change the shape without updating that schema in the same change.
→ [R22](docs/CONVENTIONS.md#r22)

---

## 🚧 Status and known limitations

Stated plainly, so nobody spends a day rediscovering them:

| | |
|---|---|
| **No automated tests** | `ruff` is the only check, and no CI job runs it. |
| **Dependencies are unpinned** | Both entries in every `requirements.txt` install from a Git branch, so two deploys of identical code can produce different runtimes. → [detail](docs/BACKLOG.md#dep-1) |
| **No full offline development** | R notebooks are edited locally but execute on the workspace kernel. → [the loop](pipelines/CLAUDE.md#editing-r-notebooks-the-vs-code-remote-kernel-loop) |
| **Run order is not written down yet** | Tracked in [`docs/DATA_ARCHITECTURE.md` §4.2](docs/DATA_ARCHITECTURE.md). |

Everything still open — tasks, questions, pending decisions — is in
[`docs/BACKLOG.md`](docs/BACKLOG.md). Why things are the way they are: [`docs/decisions/`](docs/decisions/README.md).

---

## ⚖️ Licence

[MIT](LICENSE) — **provisional**, pending confirmation by the team
([BACKLOG DOC-5](docs/BACKLOG.md#doc-5)).

This repository contains **code and documentation only**. No country health data is stored here,
and none ever should be.

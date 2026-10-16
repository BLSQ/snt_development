# 0009. conda for desk tools, with an accepted PEP 621 duplicate

Date: not recorded
Status: Accepted

## Context

Contributors need a few local Python desk tools (`ruff`, `nbstripout`, `nbdime`, the `openhexa`
CLI). The team convention is conda. `ruff` also needs its rulebook in a file it finds by walking up
from the file it checks, which means the repo root. Some contributors use PEP 621 tools
(`pip install -e .`, `uv sync`). None of these tools run on OpenHEXA: a workspace installs only what
each `<pipeline>/requirements.txt` declares.

## Decision

- **`dev/environment.yml`** is the supported way to install the desk tools.
- **`pyproject.toml`** at the root holds `ruff`'s rulebook, plus a `[project]` block with PEP 621
  metadata, `requires-python = ">=3.11"` and a short dependency list, so a PEP 621 tool can also set
  the repo up.
- The two dependency lists **deliberately overlap**. Unifying them waits on a decision about the
  local-workflow strategy ([BACKLOG DEP-2](../BACKLOG.md#dep-2)).

## Consequences

- Keep the two lists roughly in step when adding a tool. If they disagree, `dev/environment.yml`
  describes what the team actually runs.
- Neither file affects a pipeline run, so drift between them is harmless to production.
- `requires-python` is where `ruff` gets its target version (driving the `UP` and `FA` rules). If the
  `[project]` block is ever removed, add `target-version = "py311"` under `[tool.ruff]`.
- The rulebook cannot move into `dev/`: a copy there would govern only `dev/` and leave every
  pipeline on ruff's defaults.

Source: `CLAUDE.md` *Getting set up locally*.

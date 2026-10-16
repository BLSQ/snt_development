"""Validate an SNT_metadata.json-shaped file (master list, example, or a workspace copy).

Runs the JSON Schema, plus the two checks JSON Schema cannot express on its own:

- ``SCALE`` is in ascending order;
- every ``SOURCE_DATA.DATASET.NAME`` is a key of ``SNT_DATASET_IDENTIFIERS`` in the config skeleton
  (``contracts/snt_config_json/SNT_config.skeleton.json`` — the trusted key set; the archived
  per-country copies were never verified).

What it still cannot catch is listed in contracts/snt_metadata_json/README.md §5.

Usage, from anywhere (needs ``jsonschema``, which is not in dev/environment.yml)::

    python3 .claude/skills/add-explorer-layer/validate_metadata.py            # the master list
    python3 .claude/skills/add-explorer-layer/validate_metadata.py <file.json>

Exit code 1 if anything failed, else 0.
"""

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:
    sys.exit("jsonschema is not installed: `pip install jsonschema` (or use a Python that has it)")

REPO = Path(__file__).resolve().parents[3]
SCHEMA = REPO / "contracts/snt_metadata_json/snt_metadata.schema.json"
CONFIG = REPO / "contracts/snt_config_json/SNT_config.skeleton.json"  # config to resolve NAME against
DEFAULT = REPO / "metadata/SNT_metadata_master.json"


def load(path: Path) -> dict:
    """Read a JSON file as UTF-8.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    dict
        The parsed content.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def validate(metadata: Path) -> list[str]:
    """Return every problem found in one metadata file.

    Parameters
    ----------
    metadata : Path
        The file being checked.

    Returns
    -------
    list[str]
        One line per problem; empty when the file is valid.
    """
    inst = load(metadata)
    bad = [
        ("/".join(map(str, e.path)) or "(root)") + ": " + e.message
        for e in sorted(Draft202012Validator(load(SCHEMA)).iter_errors(inst), key=lambda e: list(e.path))
    ]

    # SCALE must ascend - not expressible in JSON Schema
    for lid, layer in inst.items():
        s = layer.get("SCALE", [])
        if s != sorted(s):
            bad.append(lid + ": SCALE is not in ascending order")

    # every DATASET.NAME must be a key of SNT_DATASET_IDENTIFIERS in the skeleton
    keys = load(CONFIG).get("SNT_DATASET_IDENTIFIERS", {})
    for lid, layer in inst.items():
        n = layer.get("SOURCE_DATA", {}).get("DATASET", {}).get("NAME")
        if n not in keys:
            bad.append(
                f"{lid}: DATASET.NAME {n!r} is not a key of SNT_DATASET_IDENTIFIERS in "
                f"{CONFIG.relative_to(REPO)}"
            )
    return bad


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    problems = validate(target)
    print(f"{target}: {len(problems)} problem(s)")
    print(*problems, sep="\n")
    print("FAIL" if problems else "OK")
    sys.exit(1 if problems else 0)

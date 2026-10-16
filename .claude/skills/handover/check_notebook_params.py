"""Compare the parameters each pipeline.py injects into its notebooks against the notebooks' fallbacks.

Rules checked (see CLAUDE.md Register and pipelines/CLAUDE.md):

- R12: every key in the dict passed as ``run_notebook(parameters=...)`` has an
  ``if (!exists("KEY"))`` fallback in the notebook, spelled identically, case included.
  ERROR when the notebook reads the key (in any case) without such a fallback. WARNING when the
  default sits in a papermill ``parameters``-tagged cell instead (works, wrong form), or when the
  notebook never reads the key at all (dead, harmless).
- R11: every injected key is UPPERCASE (warning only; two pipelines are known exceptions).

R comments are stripped before matching, so a commented-out ``exists()`` does not count.

Anything the script cannot resolve statically (a notebook path or a parameters dict built at
runtime) is reported as UNVERIFIED, never as passing. Country variants (``<generic>_<CC>.ipynb``)
committed next to a generic notebook receive the same parameters, so they are checked too.

Usage, from the repo root (standard library only, any Python >= 3.10)::

    python3 .claude/skills/handover/check_notebook_params.py                      # every pipeline
    python3 .claude/skills/handover/check_notebook_params.py snt_dhis2_incidence  # one or more

Exit code 1 if any ERROR was found, else 0.
"""

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXISTS_RE = re.compile(r"""exists\(\s*["']([A-Za-z_.][A-Za-z0-9_.]*)["']""")
IDENT_RE = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*")
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_.][A-Za-z0-9_.]*)\s*(?:<-|=)", re.MULTILINE)
VARIANT_RE = re.compile(r"_[A-Z]{3}$")
# Pipelines that predate R11 and inject lowercase globals (BACKLOG PIPE-3).
KNOWN_LOWERCASE = {"snt_seasonality_cases", "snt_seasonality_rainfall"}


def call_name(node: ast.Call) -> str:
    """Return the bare name of a called function.

    Parameters
    ----------
    node : ast.Call
        The call node.

    Returns
    -------
    str
        ``f`` for ``f(...)`` and ``x.f(...)``, else an empty string.
    """
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def kwarg(node: ast.Call, name: str) -> ast.expr | None:
    """Return the value of keyword argument ``name`` in a call, if given.

    Parameters
    ----------
    node : ast.Call
        The call node.
    name : str
        Keyword argument name.

    Returns
    -------
    ast.expr | None
        The argument expression, or None when absent.
    """
    return next((k.value for k in node.keywords if k.arg == name), None)


def last_assignment(scope: ast.AST, name: str, before: int) -> ast.expr | None:
    """Return the value of the last plain assignment to ``name`` before line ``before``.

    Parameters
    ----------
    scope : ast.AST
        The enclosing function (or module).
    name : str
        Variable name.
    before : int
        Line number of the use.

    Returns
    -------
    ast.expr | None
        The assigned expression, or None when there is none.
    """
    found = None
    for node in ast.walk(scope):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.lineno < before:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
                if found is None or node.lineno > found.lineno:
                    found = node
    return found.value if found else None


def resolve_keys(expr: ast.expr, scope: ast.AST, line: int) -> tuple[list[str] | None, str]:
    """Statically resolve the keys of the dict passed as ``parameters=``.

    Parameters
    ----------
    expr : ast.expr
        The ``parameters=`` expression.
    scope : ast.AST
        The enclosing function, searched for the assignment of a variable.
    line : int
        Line of the ``run_notebook`` call.

    Returns
    -------
    tuple[list[str] | None, str]
        The keys (None when unresolvable) and a reason when unresolvable.
    """
    start = 0
    var = None
    if isinstance(expr, ast.Name):
        var = expr.id
        value = last_assignment(scope, var, line)
        if value is None:
            return None, f"`{var}` is not assigned in the enclosing function (passed in as an argument?)"
        start = value.lineno
        expr = value

    if isinstance(expr, ast.Dict):
        if any(not (isinstance(k, ast.Constant) and isinstance(k.value, str)) for k in expr.keys):
            return None, "dict has non-literal keys or ** unpacking"
        keys = [k.value for k in expr.keys]
    elif isinstance(expr, ast.Call) and call_name(expr) == "dict" and not expr.args:
        if any(k.arg is None for k in expr.keywords):
            return None, "dict(...) uses ** unpacking"
        keys = [k.arg for k in expr.keywords]
    else:
        return None, f"parameters built by `{ast.unparse(expr)[:60]}`"

    # Keys added between the assignment and the call: VAR["K"] = ... and VAR.update({...}).
    if var:
        for node in ast.walk(scope):
            if not (start < getattr(node, "lineno", 0) < line):
                continue
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == var
                        and isinstance(t.slice, ast.Constant)
                    ):
                        keys.append(t.slice.value)
            if (
                isinstance(node, ast.Call)
                and call_name(node) == "update"
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == var
            ):
                return None, f"`{var}.update(...)` — check by hand"
    return keys, ""


def resolve_notebook(expr: ast.expr | None, scope: ast.AST, line: int) -> str | None:
    """Return the ``.ipynb`` filename named in an ``nb_path=`` expression.

    Parameters
    ----------
    expr : ast.expr | None
        The ``nb_path=`` expression.
    scope : ast.AST
        The enclosing function, searched when the path is held in a variable.
    line : int
        Line of the ``run_notebook`` call.

    Returns
    -------
    str | None
        The filename, or None when it is not a literal anywhere in the expression.
    """
    if expr is None:
        return None
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".ipynb"):
            return Path(node.value).name
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            value = last_assignment(scope, node.id, line)
            if value is not None:
                found = resolve_notebook(value, scope, value.lineno)
                if found:
                    return found
    return None


def scan(nb_file: Path) -> tuple[set[str], set[str], set[str]]:
    """Collect what a notebook's code cells do with names, R comments stripped.

    Parameters
    ----------
    nb_file : Path
        Path to the notebook.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        Names tested with ``exists("...")``; names assigned in a papermill ``parameters``-tagged
        cell; every identifier in the code. All exactly as spelled.
    """
    nb = json.loads(nb_file.read_text(encoding="utf-8"))
    tested, tagged, idents = set(), set(), set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())  # naive, good enough
        tested |= set(EXISTS_RE.findall(code))
        idents |= set(IDENT_RE.findall(code))
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            tagged |= set(ASSIGN_RE.findall(code))
    return tested, tagged, idents


def classify(key: str, tested: set[str], tagged: set[str], idents: set[str]) -> str | None:
    """Grade one injected key against one notebook.

    Parameters
    ----------
    key : str
        The injected parameter name.
    tested, tagged, idents : set[str]
        As returned by :func:`scan`.

    Returns
    -------
    str | None
        A problem line starting ``ERROR`` or ``WARNING``, or None when the key is fine.
    """

    def other_case(names: set[str]) -> str | None:
        return next((n for n in sorted(names) if n != key and n.lower() == key.lower()), None)

    if key in tested:
        return None
    if n := other_case(tested):
        return f'ERROR   R12 case mismatch: injects "{key}", notebook tests exists("{n}")'
    if key in tagged:
        return f'WARNING R12 form: default for "{key}" is in the papermill `parameters` cell, not exists()'
    if n := other_case(tagged):
        return f'ERROR   R12 case mismatch: injects "{key}", `parameters` cell sets "{n}"'
    if n := other_case(idents):
        return f'ERROR   R12 case mismatch: injects "{key}", notebook reads "{n}" with no fallback'
    if key in idents:
        return f'ERROR   R12 no `if (!exists("{key}"))` fallback'
    return f'WARNING injected "{key}" is never read by the notebook (harmless, but dead)'


def enclosing_functions(tree: ast.Module) -> dict[int, ast.AST]:
    """Map the id of each call node to its innermost enclosing function (or the module).

    Parameters
    ----------
    tree : ast.Module
        Parsed ``pipeline.py``.

    Returns
    -------
    dict[int, ast.AST]
        ``id(call_node) -> scope``.
    """
    scopes = {}

    def visit(node: ast.AST, scope: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Call):
                scopes[id(child)] = scope
            visit(child, inner)

    visit(tree, tree)
    return scopes


def check_pipeline(pipeline_dir: Path) -> tuple[list[str], int]:
    """Check every ``run_notebook`` call in one pipeline.

    Parameters
    ----------
    pipeline_dir : Path
        The pipeline's Python directory (contains ``pipeline.py``).

    Returns
    -------
    tuple[list[str], int]
        Report lines and the number of ERRORs.
    """
    name = pipeline_dir.name
    tree = ast.parse((pipeline_dir / "pipeline.py").read_text(encoding="utf-8"))
    scopes = enclosing_functions(tree)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and call_name(n) == "run_notebook"]
    if not calls:
        return [f"{name}: no run_notebook() call — nothing to check"], 0

    out, errors = [], 0
    for call in sorted(calls, key=lambda c: c.lineno):
        where = f"{name}/pipeline.py:{call.lineno}"
        scope = scopes.get(id(call), tree)
        params = kwarg(call, "parameters")
        keys, why = ([], "") if params is None else resolve_keys(params, scope, call.lineno)
        if keys == []:
            out.append(f"  OK         {where}: injects nothing")
            continue

        nb_name = resolve_notebook(kwarg(call, "nb_path"), scope, call.lineno)
        if nb_name is None:
            out.append(f"  UNVERIFIED {where}: notebook path is not a literal — check by hand")
            continue
        if keys is None:
            out.append(f"  UNVERIFIED {where}: {nb_name} — {why}")
            continue
        # code/ and reporting/ only: country_specific/ folders hold inert archived copies.
        candidates = [
            p for sub in ("code", "reporting") for p in (REPO / "pipelines" / name / sub).glob(nb_name)
        ]
        if not candidates:
            out.append(f"  UNVERIFIED {where}: {nb_name} not found in pipelines/{name}/code or /reporting")
            continue
        nb_file = candidates[0]

        # The generic notebook plus any committed _<CC> variant beside it (same parameters).
        variants = sorted(
            p
            for p in nb_file.parent.glob(f"{nb_file.stem}_*.ipynb")
            if VARIANT_RE.search(p.stem[len(nb_file.stem) :])
        )
        for nb in [nb_file, *variants]:
            rel = nb.relative_to(REPO).as_posix()
            found = scan(nb)
            problems = [p for p in (classify(k, *found) for k in keys) if p]
            for k in keys:
                if k != k.upper():
                    known = " (known exception, BACKLOG PIPE-3)" if name in KNOWN_LOWERCASE else ""
                    problems.append(f'WARNING R11 lowercase global "{k}"{known}')
            n_err = sum(p.startswith("ERROR") for p in problems)
            errors += n_err
            status = "ERROR     " if n_err else "OK        "
            out.append(f"  {status} {where} → {rel}: {len(keys)} key(s)")
            out.extend(f"               {p}" for p in problems)
    return [f"{name}:", *out], errors


def main(argv: list[str]) -> int:
    """Run the check on the named pipelines, or on all of them.

    Parameters
    ----------
    argv : list[str]
        Pipeline directory names (or paths to them / their ``pipeline.py``).

    Returns
    -------
    int
        Process exit code.
    """
    if argv:
        dirs = [REPO / (Path(a).parent.name if Path(a).name == "pipeline.py" else Path(a).name) for a in argv]
    else:
        dirs = sorted(p.parent for p in REPO.glob("*/pipeline.py"))
    total = 0
    for d in dirs:
        if not (d / "pipeline.py").is_file():
            print(f"{d.name}: no pipeline.py — skipped")
            continue
        lines, errors = check_pipeline(d)
        total += errors
        print(*lines, sep="\n")
    print(f"\n{total} ERROR(s)." if total else "\nNo ERRORs (UNVERIFIED lines still need a human).")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

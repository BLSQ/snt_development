# `.claude/` — agent guardrails (R19)

The rule is in [`../CLAUDE.md`](../CLAUDE.md) → *Agent guardrails*: agents never run destructive or
history-rewriting `git` / `gh` commands, even when asked. This folder is how that rule is
enforced. It is committed, so it is active on every clone.

## How it is enforced

Two layers:

| File | What it does | Needs |
|---|---|---|
| [`hooks/block-destructive-git.py`](hooks/block-destructive-git.py) | A `PreToolUse(Bash)` hook: regex-matches the command about to run and returns a `deny` decision. The precise layer — it sees through `git -C <path> …`, catches chained commands (`… && git reset --hard`), and deliberately *permits* the recovery forms `git rebase --abort/--continue/--skip`, `git clean --dry-run` and `git restore --staged`. | `python3` on `PATH`, standard library only |
| [`settings.json`](settings.json) | Wires up that hook, **and** carries a coarser `permissions.deny` list that Claude Code enforces itself. | nothing |

The deny list is not redundant. A hook that cannot start (missing interpreter, syntax error) is a
*non-blocking* error — the tool call it was meant to stop then proceeds, silently. The deny list
needs no interpreter, so it still applies. The trade-off is that prefix rules cannot express the
hook's exceptions, so it denies `git rebase` outright, `--abort` included. Run recovery commands
like that by hand.

A block from either layer is **expected behaviour, not an error to work around.**

## Which Python the hook uses

The hook runs in a plain shell with no conda environment activated, so it needs a **system**
`python3` on `PATH` — not the one inside `snt_development`. macOS, Linux and WSL have one. On
native Windows, Claude Code runs hooks through Git Bash: check `python3` resolves there, because if
it does not the `permissions.deny` half still applies and the hook half quietly does not.

## Testing the hook

Feed it a payload directly — nothing destructive runs:

```bash
echo '{"tool_input":{"command":"git reset --hard"}}' | python3 .claude/hooks/block-destructive-git.py
# prints a "deny" decision on a blocked command; prints nothing on an allowed one
```

## Changing a rule

Adding or relaxing a pattern means editing **both** layers — `RULES` in the Python file and the
`deny` list in `settings.json` — or the two disagree. Neither layer is a security boundary: they
stop an agent behaving normally, not one determined to get around them, and they do not constrain a
human at a terminal. Do not weaken them to make a task easier; if a rule is genuinely wrong, change
it in a PR of its own.

`settings.json` reaches git only because `.gitignore` has a negation for it (`*.json` is ignored
repo-wide). That negation is load-bearing: without it this guardrail never reaches a colleague's
clone.

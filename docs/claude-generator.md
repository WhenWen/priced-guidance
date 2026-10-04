# Optional Claude Code backend

The maintained runtime also contains a Claude Code compatibility backend. The
paper's main Claude experiments use the Anthropic API with common memory; use
the README command for that configuration. This optional backend is not a
replacement for those reported conditions.

To explore the native backend, use `python tools/install_claude.py --help`,
`python tools/claude_account.py --help`, and `idea-arena run --help` for installation,
your own separate authentication directory, and `--generator-backend claude`.
Pinned installation metadata lives in `tools/claude/`. Account files and generated
installation directories are local artifacts and must not be committed.

Use `tools/probe_claude_model.py --help` or `tools/probe_claude_native.py --help`
for explicit live diagnostics. These can contact providers and are not part of
the offline release checks.

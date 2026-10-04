# Source-built Codex Generator

The compatibility backend keeps the reference Generator's Python strategy and
Arena contract, and runs its semantic calls through the open-source Codex
app-server. It uses a native conversation across calls, including Codex's
persisted encrypted reasoning. It is an opt-in development runner for macOS
and Linux. Supply your own account and build the pinned backend below.

## Build from the pinned source

```bash
python3 tools/build_codex.py
```

`tools/codex/cache-source.lock.json` pins `openai/codex` release `rust-v0.153.4`,
commit `3d2ee51ca2d5db578f328aa75e20aa22c0197c9a`, and Rust `1.95.0`.
One upstream patch aligns the release's workspace versions in Cargo.lock.
`lineage-cache.patch` adds a broker-owned cache routing key, stable generated
prefix IDs, omits tool definitions for tool-free Arena calls, adds opt-in request
diagnostics, and preserves complete Arena developer context instead of Codex's
default 1,000-token truncation. It changes no dependencies. Cargo uses `--locked`.
The script installs a private Rust toolchain without changing the shell profile.

The checkout, build cache, executable, and build receipt live under the ignored
`.idea-arena/codex-cache-v1/`. The versioned lock, patches, build script, and adapter are the
reproducible inputs. The receipt records source, patch, lockfile and executable
hashes; the runtime verifies it and never searches PATH for an installed Codex.
This is local provenance, not a signed third-party build attestation.

The original `tools/codex/source.lock.json` and `.idea-arena/codex/` binary/receipt
remain available for recorded v1/v2 runs. Build that version explicitly with
`--source-lock tools/codex/source.lock.json --build-dir ANOTHER_DIRECTORY`.
An existing checkout with different patches is refused rather than overwritten.

Release `0.147.0` authenticated successfully but the service rejected Astra as
requiring a newer client. Release `0.153.4` was built and successfully tested
with `gpt-6-astra`; no model or client-version substitution was used.

## Use an independent account

```bash
uv run python tools/codex_account.py \
  --auth-home ~/.local/share/idea-arena/codex-accounts/experiment login

# Optional headless/device-code login:
uv run python tools/codex_account.py \
  --auth-home ~/.local/share/idea-arena/codex-accounts/experiment login --device-auth

uv run python tools/codex_account.py \
  --auth-home ~/.local/share/idea-arena/codex-accounts/experiment status
```

Choose the intended account in the official browser flow. The source-built CLI
saves credentials only in the selected directory. The Generator requires an
existing login and never silently imports the desktop account. To explicitly
copy an existing file-based ChatGPT login into a new directory, use the
`import-existing` subcommand, optionally with `--source /path/to/auth.json`.
API-key credentials are rejected. Credentials are never stored in Git, run
manifests, service tapes, or native-history artifacts.

The default independent directory, when no `--auth-home` is supplied, is
`~/.local/share/idea-arena/codex-account`. Multiple directories support separate
accounts. Authentication and model entitlement remain subject to the chosen
account. The official documentation describes
[ChatGPT login and credential storage](https://developers.openai.com/codex/auth).

## Run the Generator

```bash
# Full offline contract smoke test, with both Generator processes sandboxed:
uv run idea-arena run submissions/reference_pair \
  --target-pack smoke --generator-backend codex \
  --generator-model gpt-6-astra \
  --generator-codex-auth-home ~/.local/share/idea-arena/codex-accounts/experiment

# Live generation, sampling choices without loading a private target:
uv run idea-arena sample-ideas submissions/reference_pair \
  --target-pack development40 --generator-backend codex \
  --generator-model gpt-6-astra \
  --generator-codex-auth-home ~/.local/share/idea-arena/codex-accounts/experiment

# Scored evaluation; Guide/Judge keep their existing configured backends:
uv run idea-arena run submissions/reference_pair \
  --target-pack development40 --target TARGET_ID --judge directional \
  --generator-backend codex --generator-model gpt-6-astra \
  --generator-codex-effort xhigh \
  --generator-codex-auth-home ~/.local/share/idea-arena/codex-accounts/experiment \
  --env-file /path/to/guide-and-judge.env
```

The smoke pair makes no semantic model calls; use the native-history probe below
for live account/model validation. `sample-ideas` uses no Guide/Judge API calls.
Scored runs still need the existing Guide/Judge credentials unless their
existing alternative backends are selected. `resume` automatically reconstructs
the recorded Codex backend and requires the same pinned build and account path.
`--generator-codex-effort xhigh` selects Extra High for every Generator call;
the profile override is recorded and also applied during offline replay.

Native Codex calls have a **1,500-second** wall timeout, configurable by the
host with `IDEA_ARENA_CODEX_TURN_TIMEOUT_S`. The effective timeout is recorded in
each call's metadata. It is separate from the logical API service timeout, so
changing it preserves existing replay request hashes and native prompt prefixes.
The outer Codex Generator step defaults to **6,000 seconds**, allowing several
sequential eager preview calls; `IDEA_ARENA_ACTOR_TIMEOUT_S` overrides it. Ordinary
API participants retain their existing 900-second outer default.

Read quota and model capabilities for the selected account with:

```bash
uv run python tools/codex_account.py \
  --auth-home ~/.local/share/idea-arena/codex-accounts/experiment \
  inspect --model gpt-6-astra --output account-before.json
```

Repeat after the run to compare included quota and purchased-credit balance.
The native event stream supplies tokens, including cached input and reasoning
output; `tools/audit_codex_eager_run.py RUN_DIR` estimates Standard credit
equivalents using the [published token rates](https://learn.chatgpt.com/docs/pricing#what-are-tokens-and-credits).
That estimate is distinct from actual purchased-credit deductions. Account
percentages are rounded and may include concurrent usage on the same account.

After execution, append `--thread-id NATIVE_THREAD_ID` to `inspect` to query the
backend's estimated credits and token breakdown for that native conversation.
Use `metadata.codex_context.thread_id` from its recorded service calls. A missing
`threadUsage` means this billing view is unavailable. Query after execution so
account refreshes cannot disrupt the active session's cache reuse.

## Thinking, checkpoints, and replay

Each successful model call stores the entire native legacy rollout in Arena's
private content-addressed artifact store. Its service record and actor checkpoint
contain the rollout reference, native thread/turn IDs, and cumulative usage.

Normal forward calls reuse the same app-server process and native thread, keeping
the connection and routing state warm. Every completed turn is archived as an
immutable snapshot before the next call. A checkout or branch switch invokes
`thread/fork` with that exact snapshot and final turn ID. After process restart,
the broker resumes only the recorded snapshot in a new private workspace.
The complete current developer prompt and payload schema arrive as a new native
developer message at the history tail. No handwritten summary replaces the
native history, and extending the live session cannot mutate archived snapshots.

On an Arena checkout, deterministic actor reconstruction restores the context
attached to that checkpoint's service prefix. The next live call forks that
context. A discarded branch cannot become the parent of the restored Generator.
Replay returns recorded outputs without starting Codex; interrupted-run recovery
retains the context and existing global usage accounting.

Exact eager dispatch previews remain in the Python scaffold: all displayed
routes are authored before routing, and the selected cached Question is reused
without another model call. The `eager-dispatch-cache-v3` Codex prompt explicitly
requires complete slates and distinguishes unselected previews in native
history from confirmed facts or paid negative evidence. Current public state
and reported choices determine the active ledger. The prompt is supplied on
both native thread creation and every fork. Its version and SHA-256 are recorded
in new run profiles and service metadata; older runs retain their original
prompt when resumed.

## Prompt cache reuse

The v3 transport stores a random cache lineage in every native checkpoint and
reuses it across continuations and alternative forks. Each fork still has its
own native thread and turn IDs. The key is an opaque routing hint with no target
information; it neither changes the ancestor nor exposes sibling history.

The app-server stays alive between semantic calls and closes when the Arena run
finishes or fails. A changed login cache causes a fresh process to load the exact
checkpoint with the current credentials. Base instructions and the wire output schema stay constant. The wire response
is `{"payload_json": "..."}`; its string decodes to the original Generator
object. The broker validates that object against the current original schema,
including slate sizes and required fields, before returning it to the scaffold.
Thus inner-schema enforcement is local rather than server-constrained, and an
invalid payload fails the call without an automatic paid repair loop.
The task developer message has a 32,000-byte bound and is rejected before
inference if too large, rather than silently truncating its instructions/schema.

Changing schemas at the top level can invalidate the rendered prompt prefix.
Appending them as task context enables reuse of the earlier native history.
Identical prefixes and a stable routing key improve cache reuse but do not
guarantee hits: service-side routing, eviction and compaction still matter.
See the official [prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

Use the bounded synthetic test before restarting an expensive paper run:

```bash
uv run python tools/probe_codex_cache.py \
  --auth-home ~/.local/share/idea-arena/codex-accounts/experiment
```

It performs four short Astra xhigh calls (start, continue, checkpoint fork,
continue) with changing payload schemas, checks
actual outgoing prefix settings and message prefixes, records cache reads and
writes, and checks immutable ancestry and encrypted reasoning inheritance.
Request diagnostics are opt-in and archived privately; normal runs omit them.
Reports distinguish token-rate credit equivalents from actual account charges.
Cache-write premiums, when reported, use the API guide's 1.25x rate as an
estimate; the published Codex credit table does not separately specify writes.

This retains completed-turn native reasoning under Codex's own context handling.
Normal Codex compaction may summarize older history as conversations grow; it
does not promise unlimited retention of every old reasoning item or resumption
of an interrupted in-flight inference.

## Isolation boundary

Two separate OS-enforced sandboxes wrap the entire Python Generator worker
and the entire source-built Codex app-server process. macOS uses Seatbelt;
Linux uses rootless bubblewrap namespaces and a restricted TLS relay:

* The Python worker receives a staged submission, declared dependencies, the
  message contract, and wire code. It has no network, API credentials, target
  pack, evaluator, run logs, or access to the repository. Symlinked source and
  dependency trees are refused. Public resources and legal choices arrive only
  through the existing broker protocol.
* Codex receives only a private random workspace for its live session, its selected native ancestor
  rollout, the current prompt, and a copy of the selected account credential.
  It cannot read the repository, personal Codex directory, other workspaces,
  or private run artifacts. It has the system runtime reads needed to execute
  and the network access needed by the Codex transport. Host loopback is
  inaccessible. On Linux, DNS/TCP egress happens only in a host-owned relay
  restricted to approved OpenAI TLS endpoints; sandbox loopback carries the
  private proxy, never a host service.
* Model-facing environments, plugins, apps, web search, shell and subagents are
  disabled. Unexpected tool/client capability requests fail the call. Only the
  expected executable can be launched inside each macOS sandbox; Linux mounts
  only the executable and required runtimes, with no host shell or executable
  search path.
* The trusted host broker hashes and archives outputs after every turn, then removes
  the temporary workspace when the live session closes. Only updated account credentials return to the selected account
  directory, with a lock protecting refresh-token rotation.

Unsupported platforms or missing OS sandbox support fail closed. These profiles supplement the local
development runner. It does not enable the separate attested `hardened` runner
or make a hidden-target security claim against a compromised host/kernel.
System runtime directories must not be used as evaluation data stores.

## Compatibility and limits

| Function | Behavior |
| --- | --- |
| Stage strategies, finite options, probabilities, drafts, corrections, eager previews | Existing submission code and schemas |
| Judge, paid choices, checkout, rejected-attempt recovery, stage promotion | Existing Arena machinery |
| Generator thinking across semantic calls | Live session for forward calls; immutable snapshots and native forks on checkout |
| Concurrent channel calls | Serialized to establish one unambiguous reasoning history |
| Replay and resume | Recorded responses plus checkpoint-bound native context |
| Token accounting | Delta of native cumulative totals; failed calls retain reported usage and flag uncertain accounting |
| Dollar accounting | `cost_usd=0` means no metered API dollars; it does not mean no subscription quota was consumed |
| Per-call output cap | Checked after Codex returns; app-server does not expose the API's equivalent server-side cap |
| Physical HTTP attempt audit | Existing API-role journal stays separate; Codex records its native event stream |
| Tools and arbitrary API conversation-state inputs | Disabled in this compatibility profile |

Original Generator control logic remains in Python. Codex executes semantic
turns and owns their reasoning history. This deliberately changes the context
seen by later calls, so this backend is a distinct experiment configuration;
no recovery-performance equivalence is claimed from infrastructure tests.

## Verification

```bash
uv run pytest -q tests/unit/test_codex_generator_context.py \
  tests/unit/test_codex_account.py tests/integration/test_codex_sandbox.py \
  tests/integration/test_codex_recording.py

# Four small live Codex account calls; synthetic nonces only:
uv run python tools/probe_codex_generator.py \
  --auth-home ~/.local/share/idea-arena/codex-accounts/experiment
```

The live probe verifies a reasoning-producing first turn, a discarded branch,
an alternate native fork, and a continuation of that fork. It checks original
memory, absence of discarded context, immutable parent artifacts, zero model
calls during replay, and matching encrypted-reasoning fingerprints. Reports
include counts and booleans; no credentials or reasoning contents are printed.
The private rollout artifacts remain in the usual `.artifacts/sha256` store.

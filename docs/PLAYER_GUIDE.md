# Player guide

Create and validate a pair:

```bash
idea-arena new my-pair
idea-arena validate my-pair
idea-arena run my-pair --target-pack smoke --seed 1 --progress
```

## Generator

```python
from tech_tree_arena import (
    Choice,
    Idea,
    Option,
    Question,
    Submission,
    SubmitOption,
)

class Generator:
    def __init__(self, services):
        self.services = services
        self.answer = None

    def step(self, choice: Choice | None):
        if choice is not None:
            payload = choice.public_payload
            if payload["action"] == "submit":
                return Submission((
                    Idea("best", {"answer": self.answer}, "1"),
                ))
            self.answer = payload["answer"]

        return Question("Which?", (
            Option("a", {"action": "answer", "answer": "A"}, "0.63"),
            Option("b", {"action": "answer", "answer": "B"}, "0.27"),
            SubmitOption("submit", {"action": "submit"}, "0.10"),
        ))
```

Every question and submission contains one finite, strictly positive probability distribution. A question may mix ordinary `Option` values and explicit `SubmitOption` values in that same distribution. These are the final probabilities used for `-log2(p)` pricing; there is no separate smoothing parameter in the protocol.

The Arena starts with `step(None)`, which must return a `Question`. After each question it returns the selected option ID and the Generator-authored `public_payload` as a `Choice`. After an ordinary option the Generator must return another `Question`. Only a selected `SubmitOption` authorizes one `Submission`, and the Generator must return that submission immediately; returning a submission at any other time, or returning a question after a submit choice, is a protocol error.

Keep all beliefs, transcripts, and agent state on `self`. The Arena does not impose a strategy deadline such as 12 or 14 questions and never forces a submission. Default question, Oracle-decision, and submission-attempt limits are fail-closed resource safeguards (256 each), not instructions for when to submit.

## Oracle

```python
from tech_tree_arena import Checkout, Choice, SubmissionFeedback

class Oracle:
    def __init__(self, target, services):
        self.target = target
        self.services = services
        self.seen = []

    def step(self, message):
        if isinstance(message, SubmissionFeedback):
            # A rejected attempt requires one of these exact recovery handles.
            return Checkout(message.source.question_id)

        self.seen.append(message.question_id)
        return Choice("a" if len(self.seen) == 1 else "submit")
```

For a `PresentedQuestion`, the Oracle returns an offered option ID. In a time-travel run it may instead return `Checkout(old_question_id)` for an eligible earlier question. The Arena restores the Generator to the checkpoint immediately after it authored that question; the same Oracle instance stays alive and can remember the abandoned branch.

If a submission attempt is rejected, the Judge's verdicts are delivered only to the Oracle in `SubmissionFeedback`. The Oracle must then return `Checkout(...)`, and the handle must be in `valid_checkout_question_ids`. This recovery list includes the source question itself, so even a rejected root attempt can retry. When ordinary time travel is disabled, recovery to that source question is still allowed.

Checkout is a zero-bit control action. It restores the destination question's `path_K`, unwinding every abandoned choice, including a rejected `SubmitOption`. The next `Choice` is priced normally and also pays the non-rewindable continuation/option repetition surcharge. A rejected submission has no accepted-mass pointer cost; only a passing attempt retains its submit choice, adds the accepted-mass pointer cost, and terminates the match.

Neither role may attach free-form text to its decision. To communicate a value to the Generator, the Generator must first include it in a finite option payload before the Oracle chooses. Judge verdicts and failure reasons never reach the Generator directly or through an Oracle-authored payload.

## Services

Use only injected services for replayable nondeterminism:

```python
services.random()
services.randint(1, 10)
services.choice(["a", "b"])
services.structured_model(
    developer="...",
    user="...",
    schema={...},
    schema_name="answer",
    max_output_tokens=2000,
    reasoning_effort="medium",
)
```

The evaluation profile, not participant code or `submission.toml`, selects the concrete model, provider, endpoint, credentials, and limits. Calls, responses, randomness, and output hashes are recorded for deterministic reconstruction.

## Manifest

```toml
schema_version = 1
name = "my-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
```

Submit a directory or zip. If `pyproject.toml` declares third-party dependencies, include the exact `uv.lock`. Local dependency builds are unverified; hidden evaluation requires an external reviewed builder and runner.

## Judge promotion (staged submissions)

A submission may declare `[modules]` groups (`shared` plus one group per Judge
stage) in `submission.toml`. In such runs the Arena delivers
`StageTransition(from_stage, to_stage)` messages to **both** participants — at
each `--promote-judge` fork, and to the Generator again when a checkout
reconstructs it behind the active stage. Each participant must answer with
`StageReady(stage, handoff)` for the destination stage **without calling any
injected service**; the engine fails the run otherwise. `handoff` is a private
JSON-serializable state summary. Transitions add zero bits and never change K.
Submissions without `[modules]` never receive these messages.

Treat `shared` as a permanently frozen actor shell, not as the place for all
three stages' implementation. A promotion-compatible design keeps the ledger,
transition validation, and dynamic loader in shared, then delegates ordinary
messages to functions owned by the active stage:

```python
# shared entrypoint/shell (abridged)
from importlib import import_module

def stage_policy(stage):
    return import_module(f"participant.stages.{stage}")

class Generator:
    def step(self, message):
        if isinstance(message, StageTransition):
            # validate, flip stage, call destination generator_on_enter
            ...
        return stage_policy(self.state["stage"]).generator_step(self, message)
```

Each stage should own both actor entry functions (`generator_step`,
`oracle_step`), service-free `*_on_enter` hooks, its Judge criterion and route
menu, plus any finer route functions it expects to revise. Stage files may
statically import shared helpers or files in their own hash group. Shared may
not statically import a stage, and one stage may not import another; the
manifest validator rejects those dependencies. Use the frozen dynamic loader
only to select the active stage.

Actor reconstruction replays the completed prefix against the replacement
tree. Consequently, a Directional call must read only shared + Directional
policy—not Essence/Strict criteria or menus. Historical Questions can still be
selected after promotion, so the newly active stage must preserve the earlier
public payload shapes (or dispatch them through a frozen compatibility
adapter). Initialize genuinely new later-stage state in `*_on_enter` with
`setdefault` rather than changing the shared constructor.

## Results and replay

`run` prints a JSON result on stdout; its `run_dir` field names the run
directory (other fields include `status`, `score`, `K`, and `run_id`). Verify
the run with:

```bash
idea-arena replay runs/<run-id>
idea-arena replay runs/<run-id> --actor
```

The first verifies the event hash chains and score. The second starts fresh role processes, replays method inputs and recorded service responses, and verifies every participant output hash.

Inspect or continue a long run with:

```bash
idea-arena status runs/<run-id>
idea-arena resume runs/<run-id> --progress

# Retry only a terminal, uncaught provider/service failure:
idea-arena resume runs/<run-id> --retry-interrupted-call --progress
```

Run-level recovery starts a new run and leaves the interrupted or failed source run untouched. It resumes from the last committed Generator/Oracle/Judge boundary, including an in-flight submission attempt or post-rejection recovery. It preserves the active `path_K`, submission-attempt counters, private feedback, checkout capabilities, branch counters, and Judge/service tapes. Before continuing, the Arena verifies that the role/branch tape after the checkpoint is exactly the ordered write-ahead service-journal suffix, including failures and resource/RNG metadata. The default mode replays that complete sequence in its original order, including exceptions that participant code may have caught.

Use `--retry-interrupted-call` only when the final uncommitted call died from an uncaught provider/service exception and you want a fresh attempt. The Arena verifies that exact condition, records the old suffix as abandoned, carries forward its cost, usage, service counters, and RNG advancement, and makes the pending call live. It will not use this policy to bypass protocol errors, resource limits, or an unrelated exception raised after a successful service response. This is separate from a rejected submission attempt, which normally recovers inside the same run. Original submission and target paths may be moved or deleted: each run references integrity-checked, content-addressed snapshots.

For provider-backed runs, load credentials explicitly and set a spend ceiling:

```bash
idea-arena doctor --env-file /path/to/.env --live
idea-arena run my-pair --target-pack development40 \
  --env-file /path/to/.env --progress --max-cost-usd-per-role 25
```

The live doctor makes a small paid request to every distinct configured model. Without `--live`, doctor remains offline. Never publish a run directory wholesale: files named `private`, role logs, checkpoints, and the shared `.artifacts` store can contain prompts, responses, participant state, and private target material.

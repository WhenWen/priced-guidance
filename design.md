# Standalone Generator–Oracle Arena

This repository implements a local arena for submitted `(generator, oracle)` pairs. It is standalone: after installation, it does not import from or connect to the original `deepq-private` repository.

The first protocol, `idea-recovery-v1`, defines a small, general participant API. The repository's reference pair is one ordinary submission; its source layout and internal implementation are not part of the protocol.

## 1. The game

The Generator authors finite-choice questions. The Oracle knows the private target and chooses among their options. An option is either an ordinary `Option`, which asks the Generator to continue questioning, or a `SubmitOption`, which explicitly authorizes exactly one submission attempt. Both participants are ordinary stateful classes; the Arena does not prescribe their internal reasoning or a turn at which they must submit.

### Normative loop

The following pseudocode omits validation and resource-limit checks only where their placement is unambiguous:

```text
function run(generator_class, oracle_class, target):
    generator = runtime.start(generator_class, generator_services)
    oracle = runtime.start(oracle_class, target.private_view, oracle_services)
    K = 0
    output = generator.step(None)
    require output is Question

    while true:
        current = store_question(
            validate(output),
            generator_checkpoint = runtime.checkpoint(generator),
            path_K = K,
        )

        while true:
            decision = validate(oracle.step(present(current)))

            if decision is Checkout(target_question_id):
                # Ordinary time travel: only an eligible earlier question.
                destination = validate_checkout(current, target_question_id)
                generator = runtime.fork(destination.generator_checkpoint)
                K = destination.path_K
                current = destination
                continue

            choice = resolve_choice(decision, current.question)
            choice_bits = -log2(choice.probability)
            branch_bits = choice_branch_cost(current, choice.option_id)
            selected_bits = choice_bits + branch_bits
            K += selected_bits
            output = generator.step(choice.for_generator)

            if choice.option is ordinary Option:
                require output is Question
                break

            require choice.option is SubmitOption
            require output is Submission
            submission = validate(output)
            verdicts = judge.evaluate(target, submission)
            passing_mass = sum(probability[i] for each passing verdict i)

            if passing_mass > 0:
                K += -log2(passing_mass)
                return pass(score = 2^(-K), K, verdicts)

            # A rejection has no accepted-mass pointer cost. The selected
            # submit branch is provisional and will be unwound by recovery.
            feedback = SubmissionFeedback(
                source = present(current),
                submission = submission,
                verdicts = verdicts,
                valid_checkout_question_ids = eligible_recovery_questions(
                    current,
                    time_travel_enabled,
                ),
            )
            recovery = oracle.step(feedback)
            require recovery is Checkout to a listed question
            destination = validate_recovery_checkout(current, recovery)
            generator = runtime.fork(destination.generator_checkpoint)
            K = destination.path_K
            current = destination
            continue
```

The initial Generator output and every output after an ordinary option must be a `Question`. A `Submission` is legal only immediately after the Oracle selects a `SubmitOption`; at that point it is mandatory. An unauthorized early submission and a question returned after submit authorization both fail closed as protocol errors.

Every attempt is judged. A passing attempt retains its priced `SubmitOption`, adds the accepted-mass pointer cost, and terminates. A rejected attempt does not terminate: private feedback goes only to the Oracle, and the Oracle must choose a legal recovery checkout. That checkout unwinds the rejected submit choice together with the rest of the abandoned branch.

Time travel restores only the Generator. The Oracle stays alive and may remember every explored branch and every rejection. `Checkout` is a zero-bit Oracle control action outside the Generator-authored probability distribution. When ordinary time travel is disabled, a rejection may still recover to its own source question, ensuring that a root-level attempt has a retry point.

## 2. Participant API

```python
class Generator:
    def __init__(self, services): ...

    def step(
        self,
        choice: Choice | StageTransition | None,
    ) -> Question | Submission | StageReady: ...


class Oracle:
    def __init__(self, target, services): ...

    def step(
        self,
        message: PresentedQuestion | SubmissionFeedback | StageTransition,
    ) -> Choice | Checkout | StageReady: ...
```

All participant state is private instance state. Participants do not receive Arena-owned contexts, transcripts, memory dictionaries, checkpoints, branch APIs, or costs. The Generator never receives Judge output. The Oracle receives `SubmissionFeedback` only after a rejected attempt and must answer that message with a `Checkout`.

Runs that use staged Judge promotion (a submission with `[modules]` groups) additionally deliver Arena-authored `StageTransition(from_stage, to_stage)` messages to both participants: once at each promotion, and to the Generator again whenever a checkout reconstructs it behind the active stage. The participant must answer each transition with `StageReady(stage, handoff)` for the destination stage without calling any injected service (model, agent, or random) — the engine fails the run otherwise. The `handoff` value is a private, JSON-serializable state summary; stage switching is control flow, adds zero bits, and never changes K.

The arena injects deterministic random and model services. Official replayable runs prohibit other nondeterminism and external access.

## 3. Messages and scoring

```text
Question {
  question: string
  options: [Option | SubmitOption, ...]
}

Option            { option_id, public_payload, probability }
SubmitOption      { option_id, public_payload, probability, kind = "submit" }
PresentedQuestion { question_id, question }
Choice            { option_id }
Checkout          { question_id }
StageTransition   { from_stage, to_stage }   # Arena-authored; promotion runs only
StageReady        { stage, handoff }         # participant acknowledgement

Submission {
  ideas: [{ idea_id, content, probability }, ...]
}

IdeaVerdict       { idea_id, passed, private_reason }
SubmissionFeedback {
  source: PresentedQuestion
  submission: Submission
  verdicts: [IdeaVerdict, ...]
  valid_checkout_question_ids: [question_id, ...]
}
```

Question and submission probabilities are final, strictly positive distributions. The generator may derive them however it wants; the arena only validates and prices them.

For a chosen option with probability `p`, let `j` count uses of that option at the exact stored checkpoint. The fixed occurrence prior has `epsilon = 0.05`.

```text
choice_bits = -log2(p)
pi(1) = 0.95
pi(j) = 0.05 / (j * (j - 1))  for j >= 2
branch_bits = -log2(pi(j))
selected_bits = choice_bits + branch_bits
```

The global continuation and option counters never rewind. The total continuation index is audit data; only the same-option index selects the occurrence surcharge. This pricing applies equally to ordinary and submit options, including their first use. Checkout pair/target counts are retained for audit, and checkout itself adds zero bits. `contract/recovery.py` defines the shared `occurrence-prior-v2` pricing used by execution, budget enforcement, replay, and record repricing.

The evaluation profile fixes a positive repeat count `k`. The Judge independently evaluates the complete submission `k` times. A repetition passes when at least one submitted idea passes in that repetition. Let `l` be the number of passing repetitions and let `M` be the nonempty union of idea IDs accepted in at least one repetition:

```text
passing_mass = sum(probability[i] for i in M)
submission_cost = -log2(passing_mass)
repeat_cost = -log2(l / k)

K = active_path_bits + submission_cost + repeat_cost
score = 2^(-K)
```

If `l = 0`, there is no submission pointer cost, no repeat cost, and no final score yet. The selected submit option remains part of the provisional branch until the mandatory Oracle-selected recovery checkout restores a question node and its `path_K`. The Arena aggregates all repeated reasons into the complete verdict set placed in private `SubmissionFeedback`. Only `l >= 1` is terminal. Essence profiles default to `k = 3`; Directional and Strict default to `k = 1`, and a recorded evaluation profile may explicitly choose another `k`. Exhausting a safety limit fails closed; it never forces a submission or converts the last rejection into a terminal score.

## 4. Time-travel semantics

Each accepted question creates an immutable node:

```text
QuestionNode {
  question_id
  parent_question_id
  question
  generator_checkpoint
  path_K
  created_index
  integrity_hash
}
```

An ordinary checkout returns to an eligible earlier question. Rejected-submission recovery may additionally return to its source question, which is included in `SubmissionFeedback.valid_checkout_question_ids`. The runtime restores the Generator to the state immediately after it produced that exact question, then presents the stored question to the still-live Oracle.

The portable checkpoint implementation uses deterministic reconstruction:

1. Create a new generator.
2. Replay its previous `step` inputs.
3. Replay the exact recorded model and random responses.
4. Verify every output hash.
5. Use a fresh branch-specific service stream after the fork.

The Arena keeps continuation, option, checkout-pair, and resource counters globally; they never rewind. The `j`-th use of an option at a checkpoint costs `-log2(pi(j))` in addition to its bid. The first occurrence costs approximately 0.074 bits and the second costs approximately 5.322 bits. Other option choices do not change this option's occurrence index. Checkout-pair counts are audit data only and add no cost.

After any checkout to node `d`, the authoritative cost is restored as:

```text
K = d.path_K
```

This rewinds every abandoned path choice, including a rejected `SubmitOption`. The next choice from `d` then pays its ordinary information cost and the globally indexed branch/repetition surcharge. Submission-attempt, continuation, option-use, service, and resource counters remain global and do not rewind.

The default fail-closed safeguards are 256 questions, 256 Oracle decisions, 64 checkouts, 128 eligible checkout targets, 256 questions of rewind, depth 256, 256 submission attempts, and 1024 total bits. They bound resource use and do not define a strategy horizon; in particular there is no 12- or 14-question forced-submit rule.

## 5. `idea-recovery-v1`

The private target is a structured summary of a research paper. The oracle and judge receive it. The generator does not receive the paper identifier, title, summary, target filename, or other target-derived metadata.

The protocol supplies the frozen taxonomy and rankings used by the source project. An evaluation profile may also enable a hash-pinned, cutoff-safe paper-search corpus.

Multiple-choice prompts, keyword hints, corrections, mode selection, and taxonomy navigation are generator strategies expressed with ordinary `Question` messages. They are not special arena transitions.

The reference submission owns both participant roles and their strategy code. Its internal probability transformation emits only final probabilities. The Arena-owned semantic judge evaluates every idea in its own context; declared probabilities are visible only to contract scoring.

## 6. Standalone repository

```text
idea_generation/
  design.md
  README.md
  pyproject.toml
  src/tech_tree_arena/
    contract/                    # public messages, validation, pricing, scoring
    runtime/                     # actors, services, branching, match engine
    evaluation/                  # trusted per-attempt semantic judges
    submission_io/               # manifests, archives, dependency environments
    replay/                      # authoritative recording and replay
    targets/                     # target-pack loading and integrity
  submissions/
    examples/minimal_pair/
    reference_pair/              # one canonical reference source
  tools/                         # non-installed authoring utilities
  tests/
```

Arena conformance is behavioral: validation, pricing, branching, attempt judging and recovery, scoring, isolation, and replay must satisfy this document regardless of implementation history.

Resources resolve from installed package data or `IDEA_ARENA_HOME`, never from the caller's working directory or the original repository.

## 7. Submission format

```text
my-pair/
  submission.toml
  participant/
    generator.py
    oracle.py
  pyproject.toml       # optional
  uv.lock              # required with third-party dependencies
```

```toml
schema_version = 1
name = "my-pair"
version = "0.1.0"
protocol = "idea-recovery-v1"
generator = "participant.generator:Generator"
oracle = "participant.oracle:Oracle"
```

The archive validator rejects traversal, links, special files, case-fold collisions, oversized content, unknown manifest fields, and unlocked dependencies.

## 8. Trust and isolation

Participant code, dependencies, messages, logs, archives, and submitted idea text are untrusted. Authoritative costs, branch state, judge results, scores, target data, credentials, and event records remain in trusted arena code.

The following invariants are mandatory:

1. Target data reaches the generator only through validated oracle decisions.
2. Judge output never reaches the Generator. After a rejection, only the Oracle receives the validated `SubmissionFeedback` needed to choose recovery; a passing verdict terminates without another participant turn.
3. Participant code cannot choose the judge, scorer, credentials, endpoint, replay tape, or recorded costs.
4. Messages use versioned schemas, canonical encoding, bounded sizes, unique IDs, and bounded decimal probabilities; ambiguous or invalid inputs fail closed.
5. Checkout handles are run-bound, integrity-protected capabilities.
6. Resource and branch counters are trusted, atomic, and non-rewindable.
7. The participants — generator and oracle — run in separate subprocesses with separate service namespaces, credentials, logs, and caches. The judge is trusted Arena-owned code and runs inside the Arena process, with its own service namespace, credentials, and journal; it is isolated from participants by ownership, not by a process boundary.
8. Public exports contain no target data, oracle logs, judge prompts, private service tapes, or per-target hidden-evaluation diagnostics. Once the first submission attempt is recorded, the append-only public timeline seals intermediate attempt and recovery events until a sanitized terminal result; that terminal result omits attempt count, so recovery control flow cannot disclose a rejected-verdict bit.

The local runner is convenient but untrusted and is limited to public targets. Hidden targets require separate hardened guests with no shared filesystem, IPC, network, credentials, or writable cache; read-only pinned images; external CPU, memory, disk, process, output, model, and time limits; and access only to schema-limited brokers.

No design is literally unhackable. Hidden evaluation requires an independent security review of the implemented runner and threat model before use.

## 9. Persistence and replay

Every run stores a hash-chained, append-only record containing:

- hashes of the submission, protocol, profile, target pack, target, judge, models, runner, seed, and code;
- validated participant messages, private submission feedback, and authoritative costs;
- branch nodes plus non-rewindable continuation, option, checkout-audit, submission-attempt, service, and resource counters;
- private actor-call and deterministic service tapes;
- every attempt's source question, submit option, selected-choice bits, submission, Judge verdicts, and recovery decision;
- usage, final verdicts and score when passing, and any run-level failure category.

Protocol replay reads recorded actions without running participants and recomputes attempt authorization, active-path checkout restoration, branch pricing, and the final passing score. Actor replay reconstructs participants from their call inputs and service tapes—including Oracle feedback calls—and also binds each branch-aware actor input/output to the corresponding protocol question, submission, or Oracle decision; two internally consistent but mutually different tapes are invalid. A failed run's exact pending-call marker may explain one call that was still in flight or returned before its result became a protocol event, never two. Its post-checkpoint actor/Judge tape must equal the checkpoint prefix plus the ordered write-ahead service-journal suffix, including failures, meters, usage, branch IDs, and RNG state. Durable checkpoints retain all phase-local attempt and recovery state plus the Judge tape, so resume cannot skip or duplicate a judgment, reset service budgets, or alter participant cursors. Strict resume replays that suffix. The explicit interrupted-call retry policy instead classifies a proven terminal service exception as abandoned execution history: it is excluded from participant control-flow replay but retained in the hash chain and in meter, usage, cost, and RNG accounting before a fresh call is allowed.

## 10. Interfaces

```bash
idea-arena doctor
idea-arena new my-pair
idea-arena validate ./my-pair
idea-arena run ./my-pair --target-pack smoke --seed 1
idea-arena replay runs/<run-id>
idea-arena tournament ./my-pair --target-pack development40
idea-arena leaderboard
idea-arena serve
```

Offline installation, validation, smoke runs, deterministic tests, and replay require no network or API key. Live reference runs use arena-configured model credentials or a local compatible endpoint.

The optional local web interface shows submissions, run status, permitted event timelines, branch graphs, usage, scores, and profile-specific leaderboards. It binds to localhost by default and never expands hidden-profile disclosure.

## 11. Completion criteria

The implementation is complete when:

- a clean install works without the original repository;
- the executable `idea-recovery-v1` contract and bundled resource integrity are verifiable;
- the reference pair, judges, taxonomy, 71 development targets, prompts, providers, and relevant regression tests are preserved;
- ordinary play, normal time travel, rejected-attempt recovery, and repeated submissions obey the loop and accounting above;
- every question checkpoint reconstructs deterministically;
- `new`, `validate`, `run`, `replay`, `tournament`, `leaderboard`, and `doctor` work offline where applicable;
- malformed distributions, messages, submissions, archives, checkouts, and replay tapes fail closed;
- differential tests show that changing a hidden target cannot change generator-visible behavior when oracle decisions are held fixed;
- public run exports contain no private target, oracle, judge, credential, or service-tape data;
- the full architecture, unit, integration, replay, and security test suite passes.

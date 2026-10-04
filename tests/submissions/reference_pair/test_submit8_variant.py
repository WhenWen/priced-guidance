"""Synthetic-only checks of the eight-candidate experimental participant."""
import copy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from tech_tree_arena import Choice, IdeaVerdict, Option, Question, StageTransition
from tech_tree_arena.contract.scoring import IDEA_RECOVERY_V1
from tech_tree_arena.contract.validation import validate_question
from tech_tree_arena.submission_io.manifest import load_manifest, load_participant_classes

ROOT = Path(__file__).resolve().parents[3]
VARIANT = ROOT / 'submissions/reference_pair_submit8'
Generator, Oracle = load_participant_classes(load_manifest(VARIANT))
G = Generator.__init__.__globals__


class Services:
    public_resources = {'active_stage': 'directional'}
    supports_concurrent_calls = False

    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows if rows is not None else [
            {'draft': f'A synthetic mechanism {i} maps signals into a learned representation.', 'prob': 1}
            for i in range(8)]

    def structured_model(self, **request):
        self.calls.append(request)
        return {'ideas': copy.deepcopy(self.rows)}


def prepared():
    services = Services()
    generator = Generator(services)
    generator.state['current_draft'] = 'A public synthetic starting hypothesis.'
    # The route internals have their own tests. Supply deterministic complete
    # Questions here to test the real dispatch, bundle, and submit machinery.
    def previews(modes):
        result = {}
        for mode in modes:
            q = Question(f'Synthetic {mode}', (Option(mode + '-retry', {'kind': 'retry'}, '1'),))
            result[mode] = {'question': q, 'question_hash': G['_stable_hash'](G['_serialized_question'](q))}
        return result
    generator._build_directional_previews = previews
    return generator, services


def test_eight_candidates_are_bound_before_dispatch_and_submitted_without_calls():
    generator, services = prepared()
    before = copy.deepcopy(generator.state['facts'])
    question = generator._directional_dispatch_question()
    validate_question(question)
    submit = next(o for o in question.options if o.option_id == 'mode-submit')
    expected = submit.public_payload['preview']['submission']
    assert len(expected['ideas']) == 8
    assert all(o.public_payload['generator_idea_snapshot'] == expected for o in question.options)
    assert len(services.calls) == 1
    assert services.calls[0]['schema_name'] == 'directional_submission_eight'
    assert sum(Decimal(r['probability']) for r in expected['ideas']) == 1
    # An actor checkpoint restoration retains the exact cached slate.
    restored = Generator(services)
    restored.state = copy.deepcopy(generator.state)
    restored._validate_directional_submit(submit.public_payload)
    result = restored._shared_submission()
    assert G['_submission_snapshot'](result) == expected
    assert len(services.calls) == 1
    assert generator.state['facts'] == before
    # One of eight passes => 3 bits; two pass => 2 bits. Existing scoring only.
    for count, cost in [(1, 3), (2, 2), (8, 0)]:
        verdicts = [IdeaVerdict(x.idea_id, i < count) for i, x in enumerate(result.ideas)]
        outcome = IDEA_RECOVERY_V1.score_submission(result, verdicts, path_k=10)
        assert outcome.k == 10 + cost


def test_stale_and_tampered_slates_cannot_be_submitted():
    generator, _ = prepared()
    question = generator._directional_dispatch_question()
    payload = next(o.public_payload for o in question.options if o.option_id == 'mode-submit')
    changed = copy.deepcopy(payload)
    changed['preview']['submission']['ideas'][0]['probability'] = '0.5'
    with pytest.raises(ValueError):
        generator._validate_directional_submit(changed)
    generator.state['current_draft'] += ' Changed after the preview.'
    with pytest.raises(ValueError):
        generator._validate_directional_submit(payload)
    with pytest.raises(ValueError):
        generator._shared_submission()


@pytest.mark.parametrize("inherited", [False, True])
def test_essence_submit_preserves_the_published_slate_after_checkout(inherited):
    generator, services = prepared()
    if not inherited:
        generator.step(StageTransition("directional", "essence"))
    question = generator._directional_dispatch_question()
    option = next(o for o in question.options if o.option_id == "mode-submit")
    expected = copy.deepcopy(option.public_payload["preview"]["submission"])
    calls_before = len(services.calls)
    # Reconstruct the checkpoint, then apply the same free transition as the
    # runtime does on checkout into an earlier-stage question.
    restored = Generator(services)
    restored.state = copy.deepcopy(generator.state)
    if inherited:
        restored.step(StageTransition("directional", "essence"))
    result = restored.step(Choice(option.option_id, public_payload=option.public_payload))
    actual = G["_submission_snapshot"](result)
    assert actual == expected
    assert len(result.ideas) == (8 if inherited else 1)
    assert len(services.calls) == calls_before
    # A failed submission must have the same key as the next presented slate,
    # otherwise the Oracle's unchanged-failed-submission guard cannot block it.
    assert G["_draft_content_key"]("essence", actual) == G["_draft_content_key"]("essence", expected)
    result.ideas[0].content["setting_and_object"] = "mutated returned copy"
    assert restored.state["pending_dispatch_previews"]["submission_preview"] == expected


@pytest.mark.parametrize("change", ["probability", "draft", "facts"])
def test_essence_inherited_submit_rejects_corrupted_checkpoint(change):
    generator, _ = prepared()
    generator._directional_dispatch_question()
    generator.step(StageTransition("directional", "essence"))
    if change == "probability":
        generator.state["pending_dispatch_previews"]["submission_preview"]["ideas"][0]["probability"] = "0.99"
    elif change == "draft":
        generator.state["current_draft"] += " changed"
    else:
        generator.state["pending_dispatch_previews"]["fact_ledger_hash"] = "changed"
    with pytest.raises(ValueError, match="exact inherited cached slate"):
        generator._submission()


@pytest.mark.parametrize('rows', [
    [{'draft': 'duplicate', 'prob': 1}] * 8,
    [{'draft': f'candidate {i}', 'prob': 0} for i in range(8)],
    [{'draft': f'candidate {i}', 'prob': 1} for i in range(7)],
])
def test_malformed_expansion_fails_closed_without_one_candidate_fallback(rows):
    services = Services(rows)
    generator = Generator(services)
    with pytest.raises(ValueError, match='eight distinct'):
        generator._build_submission_preview('Synthetic draft')
    assert len(services.calls) == 3


def test_oracle_code_and_route_policies_are_unchanged():
    base = ROOT / 'submissions/reference_pair'
    marker = '# The Oracle below'
    assert (base/'participant/pair.py').read_text().split(marker, 1)[1] == (VARIANT/'participant/pair.py').read_text().split(marker, 1)[1]
    for path in (base/'participant/stages').glob('*.py'):
        if path.name == 'essence.py':
            # Only the inherited-slate activation adapter differs; prompts and
            # every other Essence policy entrypoint remain frozen.
            original = path.read_text()
            variant = (VARIANT/'participant/stages'/path.name).read_text()
            assert original.split('def submission(generator):')[0] == variant.split('def submission(generator):')[0]
            assert original.split('def oracle_on_enter')[1] == variant.split('def oracle_on_enter')[1]
            continue
        assert path.read_bytes() == (VARIANT/'participant/stages'/path.name).read_bytes()


def test_complete_preview_above_old_soft_cap_fits_protocol_without_truncation():
    from tech_tree_arena.contract.validation import canonical_json, MAX_PAYLOAD_BYTES
    generator, services = prepared()
    original = generator._build_directional_previews
    padding = ['x' * 11500 for _ in range(20)]
    def previews(modes):
        entries = original(modes)
        mode = next(iter(entries))
        q = Question('Synthetic large preview', (Option('full', {'content': padding}, '1'),))
        entries[mode] = {'question': q, 'question_hash': G['_stable_hash'](G['_serialized_question'](q))}
        return entries
    generator._build_directional_previews = previews
    question = generator._directional_dispatch_question()
    validate_question(question)
    assert 220000 < len(canonical_json(question)) <= MAX_PAYLOAD_BYTES
    assert any(o.public_payload.get('preview', {}).get('options', [{}])[0].get('public_payload', {}).get('content') == padding
               for o in question.options)


def test_oversized_dispatch_deduplicates_only_whiteboard_with_identical_oracle_view():
    from tech_tree_arena import PresentedQuestion
    from tech_tree_arena.contract.validation import canonical_json, MAX_PAYLOAD_BYTES
    generator, _ = prepared()
    generator.state['whiteboard'] = 'w' * 12000
    original = generator._build_directional_previews
    padding = ['x' * 11500 for _ in range(20)]
    def previews(modes):
        entries = original(modes)
        mode = next(iter(entries))
        q = Question('Synthetic large preview', (Option('full', {'content': padding}, '1'),))
        entries[mode] = {'question': q, 'question_hash': G['_stable_hash'](G['_serialized_question'](q))}
        return entries
    generator._build_directional_previews = previews
    compact = generator._directional_dispatch_question()
    validate_question(compact)
    expanded = Question(compact.question, tuple(type(o)(o.option_id,
        {**copy.deepcopy(o.public_payload), 'generator_whiteboard': generator.state['whiteboard']},
        o.probability) for o in compact.options))
    assert len(canonical_json(expanded, max_bytes=1000000)) > MAX_PAYLOAD_BYTES
    assert len(canonical_json(compact)) <= MAX_PAYLOAD_BYTES
    assert sum('generator_whiteboard' in o.public_payload for o in compact.options) == 1
    for left, right in zip(compact.options, expanded.options):
        assert left.public_payload['preview'] == right.public_payload['preview']
        assert left.public_payload['generator_idea_snapshot'] == right.public_payload['generator_idea_snapshot']
        assert left.probability == right.probability
    def event(question):
        return {'event_type': 'presented_question', 'question_id': 'synthetic',
                **G['_serialized_question'](question)}
    assert G['_render_oracle_event'](event(compact)) == G['_render_oracle_event'](event(expanded))
    assert G['_idea_snapshot'](PresentedQuestion('synthetic', compact)) == G['_idea_snapshot'](PresentedQuestion('synthetic', expanded))

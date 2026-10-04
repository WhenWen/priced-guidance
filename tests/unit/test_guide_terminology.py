"""Guide naming preserves legacy entrypoints and the original model requests."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from tech_tree_arena import GuideInput, OracleInput, Option, PresentedQuestion, Question
from tech_tree_arena.cli import build_parser
from tech_tree_arena.errors import ValidationError
from tech_tree_arena.runtime.engine import RunLimits
from tech_tree_arena.submission_io.manifest import (
    _submission_import_path, load_manifest, load_participant_classes,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name('guide_prompt_hashes.json')


def guide_requests(root, stage):
    class Services:
        public_resources = {'active_stage': stage, 'time_travel_enabled': True}

        def __init__(self):
            self.requests = []

        def structured_model(self, **request):
            self.requests.append(request)
            return {'reasoning': 'Synthetic private rationale.',
                    'state_summary': 'Synthetic durable state.', 'action': 'choose',
                    'option_id': 'option-a', 'question_id': None}

    services = Services()
    prior = {k: v for k, v in sys.modules.items()
             if k == 'participant' or k.startswith('participant.')}
    try:
        for key in prior:
            sys.modules.pop(key)
        _, guide_class = load_participant_classes(load_manifest(root))
        with _submission_import_path(root):
            guide = guide_class({'summary': {'synthetic': 'PRIVATE_GOLD_MARKER'}}, services)
            for index in range(2):
                guide.step(PresentedQuestion(f'q-{index:04d}', Question(
                    'Synthetic finite question.', (
                        Option('option-a', {'kind': 'dispatch', 'mode': 'mc'}, '0.5'),
                        Option('option-b', {'kind': 'dispatch', 'mode': 'audit'}, '0.5'),
                    ),
                )))
    finally:
        for key in list(sys.modules):
            if key == 'participant' or key.startswith('participant.'):
                sys.modules.pop(key)
        sys.modules.update(prior)
    return hashlib.sha256(json.dumps(services.requests, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize('pair', ['reference_pair', 'reference_pair_submit8', 'reference_pair_fable51'])
@pytest.mark.parametrize('stage', ['directional', 'essence', 'strict'])
def test_guide_requests_match_before_rename(pair, stage):
    baseline = json.loads(BASELINE.read_text())
    assert guide_requests(ROOT / 'submissions' / pair, stage) == baseline['requests'][f'{pair}/{stage}']


def test_old_and_new_cli_flags_are_equivalent():
    parser = build_parser()
    flags = {'agent': 'human', 'agent-model': 'test', 'agent-executable': 'test',
             'agent-reasoning-effort': 'high', 'agent-timeout-seconds': '5',
             'agent-max-budget-usd-per-turn': '2'}
    def parse(role):
        arguments = ['run', 'submissions/examples/minimal_pair']
        for suffix, value in flags.items():
            arguments.extend([f'--{role}-{suffix}', value])
        return vars(parser.parse_args(arguments))
    assert parse('guide') == parse('oracle')
    assert GuideInput is OracleInput
    assert RunLimits(max_oracle_decisions=7) == RunLimits(max_guide_decisions=7)


def test_legacy_manifest_and_entrypoint_still_work(tmp_path):
    pair = tmp_path / 'pair'
    shutil.copytree(ROOT / 'submissions/examples/minimal_pair', pair)
    path = pair / 'submission.toml'
    current = path.read_text()
    canonical = load_participant_classes(load_manifest(pair))[1]
    path.write_text(current.replace('guide = "participant.guide:Guide"',
                                    'oracle = "participant.oracle:Oracle"'))
    legacy_manifest = load_manifest(pair)
    legacy = load_participant_classes(legacy_manifest)[1]
    question = PresentedQuestion('q', Question('Pick.', (Option('a', {}, '1'),)))
    assert canonical({'answer': 'a'}, None).step(question) == legacy({'answer': 'a'}, None).step(question)
    assert legacy_manifest.oracle == legacy_manifest.guide
    path.write_text(current + '\noracle = "participant.oracle:Different"\n')
    with pytest.raises(ValidationError, match='conflicting'):
        load_manifest(pair)

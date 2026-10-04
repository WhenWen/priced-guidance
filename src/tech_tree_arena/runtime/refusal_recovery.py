"""Provider refusals are terminal for recovery, including already-created children."""
import json
from pathlib import Path

from ..errors import ArenaError
from ..replay.recorder import verify_hash_chain


class ProviderRefusalBlocked(ArenaError):
    code = 'provider_refused'
    retryable = False


def provider_refusal(root):
    root = Path(root).resolve()
    runs = root.parent
    seen = set()
    while root not in seen:
        seen.add(root)
        events = root / 'events.private.jsonl'
        if events.exists():
            for event in reversed(verify_hash_chain(events, tolerate_truncated_tail=True)):
                message = event.get('error_message') or ''
                if (event.get('error_code') == 'provider_refused' or
                    event.get('error_type') == 'NativeCallError' and message.startswith('Provider refused request')):
                    return {'run_dir': str(root), 'reason': message or 'Provider explicitly refused request'}
        manifest = root / 'manifest.json'
        if not manifest.exists():
            return None
        parent = json.loads(manifest.read_text()).get('resumed_from')
        if not parent:
            return None
        candidate = (runs / parent).resolve()
        if candidate.parent != runs:
            raise ArenaError('Invalid recovery ancestor path')
        root = candidate
    raise ArenaError('Recovery ancestry contains a cycle')


def require_no_provider_refusal(root):
    refusal = provider_refusal(root)
    if refusal:
        raise ProviderRefusalBlocked(refusal['reason'] + '; further recovery is disabled')

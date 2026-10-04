"""Explicit continuation-only changes to the common summary character cap.

The original backend and its source receipt stay intact for historical replay.
Only listed ancestor policies can enter a continuation; native history is not
rewritten or regenerated when moving to the new cap.
"""
import hashlib
from pathlib import Path

from ..errors import ReplayDivergence
from .common_memory import CommonMemoryBackend, digest


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_ancestors(config, ancestors):
    for old in ancestors:
        if (not isinstance(old.get("memory_chars"), int)
                or old["memory_chars"] > config["memory_chars"]
                or {**old, "memory_chars": config["memory_chars"]} != config):
            raise ReplayDivergence("Summary cap transition changed another policy field")


class ContinuedMemoryBackend(CommonMemoryBackend):
    def __init__(self, config, artifacts, *, ancestors, transport=None):
        validate_ancestors(config, ancestors)
        super().__init__(config, artifacts, transport=transport)
        self.accepted_policies = {digest(old) for old in ancestors} | {digest(config)}

    def structured_in_context(self, context, **request):
        with self._lock:
            current_hash = digest(self.config)
            parent_hash = (context or {}).get("policy_sha256", current_hash)
            if parent_hash not in self.accepted_policies:
                raise ReplayDivergence("Unrecorded summary policy ancestor")
            # Keep the original parent object (including its policy receipt) in
            # the service journal and summary artifact. Only new output adopts
            # the continuation policy. The lock also serializes sibling calls.
            self.policy_hash = parent_hash
            try:
                return self._structured(context, request)
            finally:
                self.policy_hash = current_hash
                metadata = getattr(self._local, "metadata", None)
                if metadata is not None:
                    metadata["policy_sha256"] = current_hash
                    if metadata.get("native_context"):
                        metadata["native_context"]["policy_sha256"] = current_hash

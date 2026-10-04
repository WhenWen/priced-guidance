"""Separate the requested summary length from its hard validation ceiling."""
import copy
import hashlib
from pathlib import Path

from ..errors import ReplayDivergence
from .common_memory import CommonMemoryBackend, SUMMARY_PROMPT, digest
from .memory_policy_transition import validate_ancestors


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def policy_hash(config, target):
    return digest({"configuration": config, "summary_target_chars": target,
                   "target_source_sha256": source_hash()})


class TargetedMemoryBackend(CommonMemoryBackend):
    def __init__(self, config, artifacts, *, ancestors, target, target_ancestors=(), transport=None):
        validate_ancestors(config, ancestors)
        if not isinstance(target, int) or not 256 <= target <= config["memory_chars"]:
            raise ReplayDivergence("Invalid summary length target")
        super().__init__(config, artifacts, transport=transport)
        self.target = target
        self.current_policy = policy_hash(config, target)
        self.accepted_policies = {digest(old) for old in ancestors} | {digest(config), self.current_policy}
        for old in target_ancestors:
            validate_ancestors(config, [old["configuration"]])
            if not 256 <= old["target"] <= old["configuration"]["memory_chars"]:
                raise ReplayDivergence("Invalid ancestor summary target")
            self.accepted_policies.add(policy_hash(old["configuration"], old["target"]))

    def structured_in_context(self, context, **request):
        with self._lock:
            parent_hash = (context or {}).get("policy_sha256", self.current_policy)
            if parent_hash not in self.accepted_policies:
                raise ReplayDivergence("Unrecorded summary policy ancestor")
            self.policy_hash = parent_hash
            try:
                return self._structured(context, request)
            finally:
                self.policy_hash = self.current_policy
                metadata = getattr(self._local, "metadata", None)
                if metadata is not None:
                    metadata["policy_sha256"] = self.current_policy
                    if metadata.get("native_context"):
                        metadata["native_context"]["policy_sha256"] = self.current_policy

    def _invoke(self, native, request, role, metadata):
        if role == "summary":
            request = copy.deepcopy(request)
            request["developer"] = SUMMARY_PROMPT + f"\nAim for a memory summary of at most {self.target} characters."
        return super()._invoke(native, request, role, metadata)

"""Recorded live output allowance, outside frozen participants/native history.

The original service request remains replayable. The effective request and all
physical operations carry the new allowance. Astra is deliberately excluded.
"""
import copy
import hashlib
from pathlib import Path
import threading

from ..errors import ReplayDivergence
from .common_memory import validate_metadata as validate_common_metadata
from .services import _request_hash

DEFAULT_OUTPUT_TOKENS = 128_000
MODELS = {"anthropic/claude-fable-5-1", "anthropic/claude-fable-5", "anthropic/claude-opus-5",
          "together/zai-org/GLM-5.3"}


def configuration(model, tokens=DEFAULT_OUTPUT_TOKENS):
    if model not in MODELS or type(tokens) is not int or not 50_000 <= tokens <= 128_000:
        raise ValueError("Common API output allowance requires a supported native API model and 50000..128000 tokens")
    return {"kind": "common-api-output-v1", "model": model, "max_output_tokens": tokens,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def validate_policy(policy):
    try:
        expected = configuration(policy["model"], policy["max_output_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayDivergence("Invalid common API output policy") from exc
    legacy = {**expected, "source_sha256": "717f14bf37bcca9946c1bca04ea74b4fecf865abbf753906db0e500def031305"}
    legacy_valid = policy["model"] in {"anthropic/claude-fable-5-1", "together/zai-org/GLM-5.3"} and policy == legacy
    if policy != expected and not legacy_valid:
        raise ReplayDivergence("Common API output policy implementation changed")


class OutputBudgetBackend:
    def __init__(self, backend, policy):
        validate_policy(policy)
        if backend.config["model"] != policy["model"]:
            raise ReplayDivergence("Output policy model differs from Generator")
        self.backend, self.policy = backend, copy.deepcopy(policy)
        self._local = threading.local()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def structured(self, **request):
        return self.structured_in_context(None, **request)

    def structured_in_context(self, context, **request):
        # Retain exactly the participant request in its service tape. Only the
        # Arena-owned live call (including summary/GLM formatter) gets more room.
        effective = {**request, "max_output_tokens": self.policy["max_output_tokens"]}
        self._local.metadata = {}
        try:
            return self.backend.structured_in_context(context, **effective)
        finally:
            metadata = self.backend.last_call_metadata()
            metadata["output_budget_policy"] = copy.deepcopy(self.policy)
            metadata["effective_request_hash"] = _request_hash("model.structured", effective)
            metadata["request_hash"] = _request_hash("model.structured", request)
            self._local.metadata = metadata

    def last_call_metadata(self):
        return copy.deepcopy(getattr(self._local, "metadata", {}))


def validate_metadata(service, artifacts=None):
    metadata = service.get("metadata") or {}
    policy = metadata.get("output_budget_policy")
    if policy is None:
        return validate_common_metadata(service, artifacts)
    validate_policy(policy)
    request = service.get("request") or {}
    if (request.get("model") != policy["model"]
            or metadata.get("request_hash") != _request_hash("model.structured", request)):
        raise ReplayDivergence("Output allowance service binding diverges")
    effective = {**request, "max_output_tokens": policy["max_output_tokens"]}
    effective_hash = _request_hash("model.structured", effective)
    if metadata.get("effective_request_hash") != effective_hash:
        raise ReplayDivergence("Effective output allowance request diverges")
    for operation in metadata.get("operations", []):
        if operation["request"].get("max_output_tokens") != policy["max_output_tokens"]:
            raise ReplayDivergence("Native operation output allowance diverges")
    # The existing validator verifies the full effective request, exact native
    # history, physical usage, output and summary. No payload fields are relaxed.
    effective_service = copy.deepcopy(service)
    effective_service["request"] = effective
    effective_service["metadata"]["request_hash"] = effective_hash
    validate_common_metadata(effective_service, artifacts)

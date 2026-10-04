"""Immutable question graph and non-rewindable branch counters."""

from __future__ import annotations

import copy
import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass, replace
from typing import Any, Callable

from .actor import ActorCheckpoint
from ..contract.messages import Question
from ..contract.recovery import (
    CURRENT_ACCOUNTING, LEGACY_ACCOUNTING, choice_surcharge, validate_accounting_version,
)
from ..contract.validation import message_hash, validate_question
from .wire import decode_message, encode_message
from ..errors import InvalidCheckout, ReplayDivergence, ValidationError


@dataclass(frozen=True, slots=True)
class QuestionNode:
    question_id: str
    parent_question_id: str | None
    question: Question
    generator_checkpoint: ActorCheckpoint
    path_k: float
    created_index: int
    integrity_hash: str


@dataclass(frozen=True, slots=True)
class CheckoutAudit:
    source_question_id: str
    target_question_id: str
    target_count: int
    pair_index: int
    branch_bits: float


@dataclass(frozen=True, slots=True)
class CheckoutPreview:
    source: QuestionNode
    target: QuestionNode
    target_count: int
    pair_index: int
    branch_bits: float
    branch_id: str


@dataclass(frozen=True, slots=True)
class ChoicePreview:
    source_question_id: str
    option_id: str
    continuation_index: int
    option_index: int
    branch_bits: float


class BranchStore:
    def __init__(self, *, run_id: str, seed: int, capability_secret: bytes | None = None,
                 accounting_version: str = CURRENT_ACCOUNTING) -> None:
        self.run_id = run_id
        self.accounting_version = validate_accounting_version(accounting_version)
        # Capabilities must not be forgeable from public run IDs or declared
        # reproducibility seeds.
        self._secret = capability_secret or secrets.token_bytes(32)
        self.nodes: dict[str, QuestionNode] = {}
        self.order: list[str] = []
        self.head: str | None = None
        self.continuation_counts: dict[str, int] = {}
        self.option_counts: dict[tuple[str, str], int] = {}
        self.checkout_pair_counts: dict[tuple[str, str], int] = {}
        self.checkout_audit: list[CheckoutAudit] = []
        self._branch_counter = 0

    def _capability(self, index: int) -> str:
        label = f"q{index:08d}"
        signature = hmac.new(self._secret, label.encode(), hashlib.sha256).hexdigest()[:24]
        return f"{label}.{signature}"

    def add_question(
        self,
        question: Question,
        *,
        checkpoint: ActorCheckpoint,
        path_k: float,
        parent_question_id: str | None,
    ) -> QuestionNode:
        question = copy.deepcopy(question)
        index = len(self.order)
        question_id = self._capability(index)
        digest = message_hash({
            "question_id": question_id,
            "parent_question_id": parent_question_id,
            "question": question,
            "path_k": path_k,
            "created_index": index,
        })
        node = QuestionNode(
            question_id,
            parent_question_id,
            question,
            checkpoint,
            path_k,
            index,
            digest,
        )
        self.nodes[question_id] = node
        self.order.append(question_id)
        self.head = question_id
        return node

    def get(self, question_id: str) -> QuestionNode:
        if not isinstance(question_id, str):
            raise InvalidCheckout("checkout handle must be a string")
        node = self.nodes.get(question_id)
        if node is None:
            raise InvalidCheckout("checkout handle is invalid for this run")
        label, separator, signature = question_id.partition(".")
        expected = hmac.new(self._secret, label.encode(), hashlib.sha256).hexdigest()[:24]
        if not separator or not hmac.compare_digest(signature, expected):
            raise InvalidCheckout("checkout capability failed authentication")
        return node

    def valid_checkout_targets(
        self,
        source: QuestionNode,
        *,
        include_source: bool = False,
        allow_earlier: bool = True,
    ) -> tuple[QuestionNode, ...]:
        return tuple(
            self.nodes[question_id]
            for question_id in self.order
            if (
                allow_earlier
                and self.nodes[question_id].created_index < source.created_index
            )
            or (include_source and question_id == source.question_id)
        )

    def preview_checkout(
        self,
        source: QuestionNode,
        target_id: str,
        *,
        include_source: bool = False,
        allow_earlier: bool = True,
    ) -> CheckoutPreview:
        """Validate a checkout and compute its audit record without mutation."""
        target = self.get(target_id)
        valid = self.valid_checkout_targets(
            source,
            include_source=include_source,
            allow_earlier=allow_earlier,
        )
        if target not in valid:
            raise InvalidCheckout("checkout target is not a resumable question")
        pair = (source.question_id, target.question_id)
        pair_index = self.checkout_pair_counts.get(pair, 0) + 1
        return CheckoutPreview(
            source,
            target,
            len(valid),
            pair_index,
            0.0,
            f"branch-{self._branch_counter + 1:08d}",
        )

    def commit_checkout(
        self, preview: CheckoutPreview, *, commit_branch: bool = False
    ) -> None:
        """Commit a previously validated checkout after reconstruction succeeds."""
        pair = (preview.source.question_id, preview.target.question_id)
        if self.checkout_pair_counts.get(pair, 0) + 1 != preview.pair_index:
            raise RuntimeError("checkout preview is stale")
        if commit_branch and preview.branch_id != f"branch-{self._branch_counter + 1:08d}":
            raise RuntimeError("checkout branch preview is stale")
        self.checkout_pair_counts[pair] = preview.pair_index
        self.checkout_audit.append(
            CheckoutAudit(
                preview.source.question_id,
                preview.target.question_id,
                preview.target_count,
                preview.pair_index,
                preview.branch_bits,
            )
        )
        self.head = preview.target.question_id
        if commit_branch:
            self._branch_counter += 1

    def preview_choice(self, source: QuestionNode, option_id: str) -> ChoicePreview:
        """Compute the next non-rewindable choice surcharge without mutation."""
        continuation_index = self.continuation_counts.get(source.question_id, 0) + 1
        option_key = (source.question_id, option_id)
        option_index = self.option_counts.get(option_key, 0) + 1
        return ChoicePreview(
            source.question_id,
            option_id,
            continuation_index,
            option_index,
            choice_surcharge(continuation_index, option_index, version=self.accounting_version),
        )

    def commit_choice(self, preview: ChoicePreview) -> None:
        """Commit a choice preview after its edge passes all resource checks."""
        option_key = (preview.source_question_id, preview.option_id)
        if (
            self.continuation_counts.get(preview.source_question_id, 0) + 1
            != preview.continuation_index
            or self.option_counts.get(option_key, 0) + 1 != preview.option_index
        ):
            raise RuntimeError("choice preview is stale")
        self.continuation_counts[preview.source_question_id] = (
            preview.continuation_index
        )
        self.option_counts[option_key] = preview.option_index

    def choice_branch_cost(self, source: QuestionNode, option_id: str) -> float:
        """Compatibility helper that previews and immediately commits a choice."""
        preview = self.preview_choice(source, option_id)
        self.commit_choice(preview)
        return preview.branch_bits

    def export_state(
        self,
        encode_checkpoint: Callable[[ActorCheckpoint], dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "accounting_version": self.accounting_version,
            "capability_secret": self._secret.hex(),
            "head": self.head,
            "order": list(self.order),
            "continuation_counts": dict(self.continuation_counts),
            "option_counts": [
                [question_id, option_id, count]
                for (question_id, option_id), count in self.option_counts.items()
            ],
            "checkout_pair_counts": [
                [source_id, target_id, count]
                for (source_id, target_id), count in self.checkout_pair_counts.items()
            ],
            "checkout_audit": [
                {
                    "source_question_id": item.source_question_id,
                    "target_question_id": item.target_question_id,
                    "target_count": item.target_count,
                    "pair_index": item.pair_index,
                    "branch_bits": item.branch_bits,
                }
                for item in self.checkout_audit
            ],
            "branch_counter": self._branch_counter,
            "nodes": {
                question_id: {
                    "question_id": node.question_id,
                    "parent_question_id": node.parent_question_id,
                    "question": encode_message(node.question),
                    "generator_checkpoint": encode_checkpoint(node.generator_checkpoint),
                    "path_k": node.path_k,
                    "created_index": node.created_index,
                    "integrity_hash": node.integrity_hash,
                }
                for question_id, node in self.nodes.items()
            },
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        seed: int,
        decode_checkpoint: Callable[[dict[str, Any]], ActorCheckpoint],
    ) -> "BranchStore":
        try:
            secret = bytes.fromhex(str(state["capability_secret"]))
            if len(secret) != 32:
                raise ValueError("checkpoint capability secret has an invalid length")
            run_id = state["run_id"]
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("checkpoint run ID is invalid")
            store = cls(
                run_id=run_id,
                seed=seed,
                capability_secret=secret,
                accounting_version=state.get("accounting_version", LEGACY_ACCOUNTING),
            )
            store.head = state.get("head")
            raw_order = state.get("order", [])
            if (
                not isinstance(raw_order, list)
                or not all(isinstance(value, str) and value for value in raw_order)
                or len(raw_order) != len(set(raw_order))
            ):
                raise ValueError("checkpoint branch order is invalid")
            store.order = list(raw_order)
            raw_nodes = state.get("nodes") or {}
            if not isinstance(raw_nodes, dict) or set(store.order) != set(raw_nodes):
                raise ValueError("checkpoint branch order does not match nodes")
            store.nodes = {}
            for index, question_id in enumerate(store.order):
                item = raw_nodes[question_id]
                if not isinstance(item, dict) or item.get("question_id") != question_id:
                    raise ValueError("checkpoint node ID is inconsistent")
                if question_id != store._capability(index):
                    raise ValueError("checkpoint node capability is invalid")
                parent = item.get("parent_question_id")
                if parent is not None and parent not in store.nodes:
                    raise ValueError("checkpoint node parent is not an earlier node")
                question = decode_message(item["question"])
                if not isinstance(question, Question):
                    raise TypeError("checkpoint node is not a question")
                validate_question(question)
                path_k = float(item["path_k"])
                created_index = int(item["created_index"])
                if (
                    not math.isfinite(path_k)
                    or path_k < 0.0
                    or created_index != index
                ):
                    raise ValueError("checkpoint node accounting is invalid")
                integrity = message_hash({
                    "question_id": question_id,
                    "parent_question_id": parent,
                    "question": question,
                    "path_k": item["path_k"],
                    "created_index": created_index,
                })
                if item.get("integrity_hash") != integrity:
                    raise ValueError("checkpoint node integrity hash is invalid")
                generator_checkpoint = decode_checkpoint(item["generator_checkpoint"])
                if (
                    not generator_checkpoint.calls
                    or generator_checkpoint.calls[-1].output_hash != message_hash(question)
                ):
                    raise ValueError(
                        "checkpoint question is not bound to its generator output"
                    )
                store.nodes[question_id] = QuestionNode(
                    question_id,
                    parent,
                    question,
                    generator_checkpoint,
                    path_k,
                    created_index,
                    integrity,
                )
            if store.head is not None and store.head not in store.nodes:
                raise ValueError("checkpoint branch head is invalid")

            raw_continuations = state.get("continuation_counts") or {}
            if not isinstance(raw_continuations, dict):
                raise TypeError("checkpoint continuation counters are invalid")
            store.continuation_counts = {}
            for key, value in raw_continuations.items():
                count = int(value)
                if key not in store.nodes or count < 1:
                    raise ValueError("checkpoint continuation counter is invalid")
                store.continuation_counts[key] = count

            store.option_counts = {}
            for value in state.get("option_counts", []):
                if not isinstance(value, list) or len(value) != 3:
                    raise TypeError("checkpoint option counter is invalid")
                question_id, option_id, raw_count = value
                key = (question_id, option_id)
                count = int(raw_count)
                node = store.nodes.get(question_id)
                if (
                    node is None
                    or not isinstance(option_id, str)
                    or option_id not in {option.option_id for option in node.question.options}
                    or count < 1
                    or key in store.option_counts
                    or count > store.continuation_counts.get(question_id, 0)
                ):
                    raise ValueError("checkpoint option counter is invalid")
                store.option_counts[key] = count

            store.checkout_pair_counts = {}
            for value in state.get("checkout_pair_counts", []):
                if not isinstance(value, list) or len(value) != 3:
                    raise TypeError("checkpoint checkout counter is invalid")
                source_id, target_id, raw_count = value
                key = (source_id, target_id)
                count = int(raw_count)
                if (
                    source_id not in store.nodes
                    or target_id not in store.nodes
                    or store.nodes[target_id].created_index
                    > store.nodes[source_id].created_index
                    or count < 1
                    or key in store.checkout_pair_counts
                ):
                    raise ValueError("checkpoint checkout counter is invalid")
                store.checkout_pair_counts[key] = count

            store.checkout_audit = []
            audit_indices: dict[tuple[str, str], int] = {}
            for item in state.get("checkout_audit", []):
                if not isinstance(item, dict):
                    raise TypeError("checkpoint checkout audit is invalid")
                audit = CheckoutAudit(
                    str(item["source_question_id"]),
                    str(item["target_question_id"]),
                    int(item["target_count"]),
                    int(item["pair_index"]),
                    float(item["branch_bits"]),
                )
                pair = (audit.source_question_id, audit.target_question_id)
                expected_pair_index = audit_indices.get(pair, 0) + 1
                source = store.nodes.get(audit.source_question_id)
                target = store.nodes.get(audit.target_question_id)
                if (
                    source is None
                    or target is None
                    or target.created_index > source.created_index
                    or audit.target_count < 1
                    or audit.target_count > source.created_index + 1
                    or audit.pair_index != expected_pair_index
                    or audit.branch_bits != 0.0
                ):
                    raise ValueError("checkpoint checkout audit is invalid")
                audit_indices[pair] = audit.pair_index
                store.checkout_audit.append(audit)
            if audit_indices != store.checkout_pair_counts:
                raise ValueError("checkpoint checkout audit disagrees with counters")
            store._branch_counter = int(state.get("branch_counter", 0))
            if store._branch_counter != len(store.checkout_audit):
                raise ValueError("checkpoint branch counter is invalid")
            return store
        except ReplayDivergence:
            raise
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ReplayDivergence("invalid durable branch checkpoint") from exc

    def upgrade_accounting(self, node_costs: dict[str, float]) -> None:
        """Install a verified prefix's current prices at an explicit boundary."""
        if self.accounting_version != LEGACY_ACCOUNTING or set(node_costs) != set(self.nodes):
            raise ReplayDivergence("invalid accounting upgrade")
        replacement = {}
        for key, node in self.nodes.items():
            cost = node_costs[key]
            if not math.isfinite(cost) or cost < 0:
                raise ReplayDivergence("invalid repriced checkpoint cost")
            digest = message_hash({
                "question_id": key, "parent_question_id": node.parent_question_id,
                "question": node.question, "path_k": cost,
                "created_index": node.created_index,
            })
            replacement[key] = replace(node, path_k=cost, integrity_hash=digest)
        self.nodes = replacement
        self.accounting_version = CURRENT_ACCOUNTING

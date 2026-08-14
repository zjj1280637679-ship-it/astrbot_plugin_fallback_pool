from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .domain import (
    CandidateIdentity,
    ContextBucket,
    DisabledCandidate,
    RankedCandidate,
)
from .ledger import TrustLedger


@dataclass(slots=True)
class RankingResult:
    active: list[RankedCandidate]
    disabled: list[DisabledCandidate]

    @property
    def ordered_items(self) -> list[Any]:
        return [candidate.item for candidate in self.active]


class AdaptiveScheduler:
    """Convert decaying failure evidence into a stable, non-linear pool order."""

    def __init__(
        self,
        ledger: TrustLedger,
        *,
        burst_to_bottom_count: int = 5,
    ) -> None:
        self.ledger = ledger
        self.burst_to_bottom_count = max(2, int(burst_to_bottom_count))

    def rank(
        self,
        candidates: Iterable[Any],
        identity_for: Callable[[Any], CandidateIdentity],
        bucket: ContextBucket,
        *,
        now: float | None = None,
    ) -> RankingResult:
        now = time.time() if now is None else float(now)
        unique: list[tuple[Any, CandidateIdentity]] = []
        seen_keys: set[str] = set()
        for item in candidates:
            identity = identity_for(item)
            if identity.key in seen_keys:
                continue
            seen_keys.add(identity.key)
            self.ledger.ensure_record(identity)
            unique.append((item, identity))

        pool_size = len(unique)
        active: list[RankedCandidate] = []
        disabled: list[DisabledCandidate] = []
        for base_index, (item, identity) in enumerate(unique):
            is_disabled, reason, disabled_until, manual = self.ledger.disabled_state(
                identity.key,
                now=now,
            )
            if is_disabled:
                disabled.append(
                    DisabledCandidate(
                        item=item,
                        identity=identity,
                        reason=reason,
                        disabled_until=disabled_until,
                        manual=manual,
                    )
                )
                continue

            evidence = self.ledger.evidence(identity.key, bucket, now=now)
            recent_failures = self.ledger.recent_failure_count(
                identity.key,
                bucket,
                now=now,
            )
            shift = self._shift_for(
                evidence=evidence,
                recent_failures=recent_failures,
                pool_size=pool_size,
            )
            effective_rank = min(max(0, pool_size - 1), base_index + shift)
            active.append(
                RankedCandidate(
                    item=item,
                    identity=identity,
                    base_index=base_index,
                    evidence=evidence,
                    recent_failures=recent_failures,
                    shift=shift,
                    effective_rank=effective_rank,
                )
            )

        # A healthy candidate wins ties against a penalized candidate that was
        # shifted onto the same effective rank. Therefore one failure on rank 0
        # really produces [rank 1, rank 0, ...], rather than a no-op stable tie.
        active.sort(
            key=lambda candidate: (
                candidate.effective_rank,
                0 if candidate.shift == 0 else 1,
                candidate.evidence,
                candidate.base_index,
            )
        )
        return RankingResult(active=active, disabled=disabled)

    def _shift_for(
        self,
        *,
        evidence: float,
        recent_failures: int,
        pool_size: int,
    ) -> int:
        if pool_size <= 1:
            return 0
        if recent_failures >= self.burst_to_bottom_count:
            return pool_size - 1

        # Deliberately non-linear. Isolated uncertainty is treated gently;
        # clustered evidence quickly pushes the candidate toward the tail.
        if evidence < 0.55:
            return 0
        if evidence < 1.45:
            return 1
        if evidence < 2.45:
            return min(2, pool_size - 1)
        if evidence < 3.45:
            return min(3, pool_size - 1)
        if evidence < 4.25:
            return max(1, pool_size - 2)
        return pool_size - 1

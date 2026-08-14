from pathlib import Path

from fallback_pool.domain import (
    CandidateIdentity,
    ContextBucket,
    ErrorCategory,
    ErrorSignal,
)
from fallback_pool.ledger import TrustLedger
from fallback_pool.scheduler import AdaptiveScheduler


class Candidate:
    def __init__(self, name: str) -> None:
        self.name = name


def identity(candidate: Candidate) -> CandidateIdentity:
    return CandidateIdentity(
        key=f"{candidate.name}::model",
        provider_id=candidate.name,
        model="model",
        label=candidate.name,
    )


def make_ledger(path: Path, *, half_life: float = 1800.0) -> TrustLedger:
    return TrustLedger(
        path,
        half_life_seconds=half_life,
        burst_window_seconds=600,
        success_retention_ratio=0.65,
    )


def transient_signal(global_share: float = 1.0) -> ErrorSignal:
    return ErrorSignal(
        category=ErrorCategory.TRANSIENT,
        severity=1.0,
        global_share=global_share,
        summary="timeout",
    )


def test_one_failure_moves_first_candidate_back_one_place(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path / "ledger.json")
    scheduler = AdaptiveScheduler(ledger)
    candidates = [Candidate("A"), Candidate("B"), Candidate("C")]
    ledger.record_failure(
        identity(candidates[0]),
        transient_signal(),
        ContextBucket.SMALL,
        now=1000,
    )

    ranking = scheduler.rank(
        candidates,
        identity,
        ContextBucket.SMALL,
        now=1000,
    )

    assert [item.item.name for item in ranking.active] == ["B", "A", "C"]
    assert ranking.active[1].shift == 1


def test_five_failures_in_ten_minutes_move_candidate_to_tail(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path / "ledger.json")
    scheduler = AdaptiveScheduler(ledger, burst_to_bottom_count=5)
    candidates = [Candidate(name) for name in "ABCDE"]
    for timestamp in (1000, 1080, 1160, 1240, 1320):
        ledger.record_failure(
            identity(candidates[0]),
            transient_signal(),
            ContextBucket.SMALL,
            now=timestamp,
        )

    ranking = scheduler.rank(
        candidates,
        identity,
        ContextBucket.SMALL,
        now=1320,
    )

    assert ranking.active[-1].item.name == "A"
    assert ranking.active[-1].recent_failures == 5


def test_time_decay_eventually_restores_base_order(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path / "ledger.json", half_life=600)
    scheduler = AdaptiveScheduler(ledger)
    candidates = [Candidate("A"), Candidate("B"), Candidate("C")]
    ledger.record_failure(
        identity(candidates[0]),
        transient_signal(),
        ContextBucket.SMALL,
        now=0,
    )

    ranking = scheduler.rank(
        candidates,
        identity,
        ContextBucket.SMALL,
        now=2400,
    )

    assert [item.item.name for item in ranking.active] == ["A", "B", "C"]


def test_success_reduces_existing_failure_evidence(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path / "ledger.json")
    candidate = Candidate("A")
    candidate_identity = identity(candidate)
    for _ in range(3):
        ledger.record_failure(
            candidate_identity,
            transient_signal(),
            ContextBucket.SMALL,
            now=1000,
        )
    before = ledger.evidence(candidate_identity.key, ContextBucket.SMALL, now=1000)
    ledger.record_success(
        candidate_identity,
        ContextBucket.SMALL,
        now=1000,
    )
    after = ledger.evidence(candidate_identity.key, ContextBucket.SMALL, now=1000)

    assert after < before
    assert round(after / before, 2) == 0.65


def test_context_error_only_penalizes_matching_load_bucket(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path / "ledger.json")
    candidate = Candidate("A")
    signal = ErrorSignal(
        category=ErrorCategory.CONTEXT,
        severity=1.0,
        global_share=0.0,
        summary="context too long",
    )
    ledger.record_failure(
        identity(candidate),
        signal,
        ContextBucket.HUGE,
        now=1000,
    )

    huge = ledger.evidence(identity(candidate).key, ContextBucket.HUGE, now=1000)
    small = ledger.evidence(identity(candidate).key, ContextBucket.SMALL, now=1000)

    assert huge == 1.0
    assert small == 0.0


def test_hard_disabled_candidate_is_removed_until_expiry(tmp_path: Path) -> None:
    ledger = make_ledger(tmp_path / "ledger.json")
    scheduler = AdaptiveScheduler(ledger)
    candidates = [Candidate("A"), Candidate("B")]
    quota = ErrorSignal(
        category=ErrorCategory.QUOTA,
        severity=3.0,
        global_share=1.0,
        summary="credit exhausted",
        hard_disable=True,
    )
    ledger.record_failure(
        identity(candidates[0]),
        quota,
        ContextBucket.SMALL,
        now=1000,
        hard_disable_seconds=3600,
    )

    ranking = scheduler.rank(
        candidates,
        identity,
        ContextBucket.SMALL,
        now=1001,
    )
    assert [item.item.name for item in ranking.active] == ["B"]
    assert [item.item.name for item in ranking.disabled] == ["A"]

    after_expiry = scheduler.rank(
        candidates,
        identity,
        ContextBucket.SMALL,
        now=5000,
    )
    assert {item.item.name for item in after_expiry.active} == {"A", "B"}

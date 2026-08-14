from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCategory(StrEnum):
    """Normalized error families used by the adaptive scheduler."""

    QUOTA = "quota"
    OVERLOAD = "overload"
    TRANSIENT = "transient"
    CONTEXT = "context"
    AUTH = "auth"
    REQUEST = "request"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class ContextBucket(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"


@dataclass(slots=True)
class ErrorSignal:
    """A provider failure after best-effort normalization.

    ``global_share`` is the probability-like share of evidence assigned to the
    provider/model itself. The remainder is assigned to the current request-load
    bucket. This lets an ambiguous timeout penalize both hypotheses without
    pretending that the root cause is known.
    """

    category: ErrorCategory
    severity: float
    global_share: float
    summary: str
    status_code: int | None = None
    retry_after_seconds: float | None = None
    hard_disable: bool = False
    daily_quota: bool = False


@dataclass(slots=True)
class EvidenceEvent:
    timestamp: float
    severity: float
    category: str
    bucket: str
    global_share: float
    summary: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvidenceEvent:
        return cls(
            timestamp=float(raw.get("timestamp", 0.0)),
            severity=max(0.0, float(raw.get("severity", 0.0))),
            category=str(raw.get("category", ErrorCategory.UNKNOWN.value)),
            bucket=str(raw.get("bucket", ContextBucket.SMALL.value)),
            global_share=min(1.0, max(0.0, float(raw.get("global_share", 1.0)))),
            summary=str(raw.get("summary", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelRecord:
    provider_id: str
    model: str
    label: str
    events: list[EvidenceEvent] = field(default_factory=list)
    disabled_until: float | None = None
    disable_reason: str = ""
    manual_disabled: bool = False
    last_success_at: float | None = None
    last_failure_at: float | None = None
    success_count: int = 0
    failure_count: int = 0
    last_category: str = ""
    last_summary: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelRecord:
        events_raw = raw.get("events", [])
        events = [
            EvidenceEvent.from_dict(item)
            for item in events_raw
            if isinstance(item, dict)
        ]
        disabled_until_raw = raw.get("disabled_until")
        return cls(
            provider_id=str(raw.get("provider_id", "<unknown>")),
            model=str(raw.get("model", "")),
            label=str(raw.get("label", raw.get("provider_id", "<unknown>"))),
            events=events,
            disabled_until=(
                float(disabled_until_raw)
                if isinstance(disabled_until_raw, int | float)
                else None
            ),
            disable_reason=str(raw.get("disable_reason", "")),
            manual_disabled=bool(raw.get("manual_disabled", False)),
            last_success_at=_optional_float(raw.get("last_success_at")),
            last_failure_at=_optional_float(raw.get("last_failure_at")),
            success_count=max(0, int(raw.get("success_count", 0))),
            failure_count=max(0, int(raw.get("failure_count", 0))),
            last_category=str(raw.get("last_category", "")),
            last_summary=str(raw.get("last_summary", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model": self.model,
            "label": self.label,
            "events": [event.to_dict() for event in self.events],
            "disabled_until": self.disabled_until,
            "disable_reason": self.disable_reason,
            "manual_disabled": self.manual_disabled,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_category": self.last_category,
            "last_summary": self.last_summary,
        }


@dataclass(slots=True)
class CandidateIdentity:
    key: str
    provider_id: str
    model: str
    label: str


@dataclass(slots=True)
class RankedCandidate:
    item: Any
    identity: CandidateIdentity
    base_index: int
    evidence: float
    recent_failures: int
    shift: int
    effective_rank: int


@dataclass(slots=True)
class DisabledCandidate:
    item: Any
    identity: CandidateIdentity
    reason: str
    disabled_until: float | None
    manual: bool


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None

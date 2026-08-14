from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from .domain import CandidateIdentity, ContextBucket, ErrorSignal, EvidenceEvent, ModelRecord

_SCHEMA_VERSION = 1


class TrustLedger:
    """Persistent, decaying evidence ledger keyed by provider instance and model."""

    def __init__(
        self,
        path: Path,
        *,
        half_life_seconds: float,
        burst_window_seconds: float,
        success_retention_ratio: float,
        retention_seconds: float = 7 * 24 * 3600,
        max_events_per_model: int = 200,
    ) -> None:
        self.path = path
        self.half_life_seconds = max(1.0, float(half_life_seconds))
        self.burst_window_seconds = max(1.0, float(burst_window_seconds))
        self.success_retention_ratio = min(
            1.0, max(0.0, float(success_retention_ratio))
        )
        self.retention_seconds = max(self.half_life_seconds, float(retention_seconds))
        self.max_events_per_model = max(10, int(max_events_per_model))
        self._records: dict[str, ModelRecord] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._load()

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def ensure_record(self, identity: CandidateIdentity) -> ModelRecord:
        with self._lock:
            record = self._records.get(identity.key)
            if record is None:
                record = ModelRecord(
                    provider_id=identity.provider_id,
                    model=identity.model,
                    label=identity.label,
                )
                self._records[identity.key] = record
                self._dirty = True
            else:
                record.provider_id = identity.provider_id
                record.model = identity.model
                record.label = identity.label
            return record

    def record_failure(
        self,
        identity: CandidateIdentity,
        signal: ErrorSignal,
        bucket: ContextBucket,
        *,
        now: float | None = None,
        hard_disable_seconds: float | None = None,
    ) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self.ensure_record(identity)
            record.failure_count += 1
            record.last_failure_at = now
            record.last_category = signal.category.value
            record.last_summary = signal.summary

            if signal.severity > 0:
                record.events.append(
                    EvidenceEvent(
                        timestamp=now,
                        severity=signal.severity,
                        category=signal.category.value,
                        bucket=bucket.value,
                        global_share=min(1.0, max(0.0, signal.global_share)),
                        summary=signal.summary,
                    )
                )

            if signal.hard_disable and hard_disable_seconds and hard_disable_seconds > 0:
                requested_until = now + hard_disable_seconds
                if signal.retry_after_seconds:
                    requested_until = max(
                        requested_until,
                        now + max(0.0, signal.retry_after_seconds),
                    )
                record.disabled_until = max(
                    record.disabled_until or 0.0,
                    requested_until,
                )
                record.disable_reason = signal.summary or signal.category.value

            self._prune_record(record, now)
            self._dirty = True

    def record_success(
        self,
        identity: CandidateIdentity,
        bucket: ContextBucket,
        *,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self.ensure_record(identity)
            record.success_count += 1
            record.last_success_at = now

            # A success is direct counter-evidence. It cannot create a positive
            # "credit balance" that masks future failures; it only shrinks
            # evidence that already exists and contributes to this request.
            for event in record.events:
                contributes_globally = event.global_share > 0
                contributes_locally = event.bucket == bucket.value
                if contributes_globally or contributes_locally:
                    event.severity *= self.success_retention_ratio

            self._prune_record(record, now)
            self._dirty = True

    def evidence(
        self,
        key: str,
        bucket: ContextBucket,
        *,
        now: float | None = None,
    ) -> float:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return 0.0
            return sum(
                self._event_contribution(event, bucket, now)
                for event in record.events
            )

    def global_evidence(self, key: str, *, now: float | None = None) -> float:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return 0.0
            total = 0.0
            for event in record.events:
                total += self._decayed_severity(event, now) * event.global_share
            return total

    def bucket_evidence(
        self,
        key: str,
        bucket: ContextBucket,
        *,
        now: float | None = None,
    ) -> float:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return 0.0
            total = 0.0
            for event in record.events:
                if event.bucket == bucket.value:
                    total += self._decayed_severity(event, now) * (
                        1.0 - event.global_share
                    )
            return total

    def recent_failure_count(
        self,
        key: str,
        bucket: ContextBucket,
        *,
        now: float | None = None,
    ) -> int:
        """Count still-credible failures in the configured burst window."""

        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return 0
            count = 0
            for event in record.events:
                if now - event.timestamp > self.burst_window_seconds:
                    continue
                contributes = event.global_share >= 0.95 or event.bucket == bucket.value
                if contributes and event.severity >= 0.8:
                    count += 1
            return count

    def disabled_state(
        self,
        key: str,
        *,
        now: float | None = None,
    ) -> tuple[bool, str, float | None, bool]:
        now = time.time() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return False, "", None, False
            if record.manual_disabled:
                return True, record.disable_reason or "手动禁用", None, True
            if record.disabled_until and record.disabled_until > now:
                return (
                    True,
                    record.disable_reason or "额度类硬暂停",
                    record.disabled_until,
                    False,
                )
            if record.disabled_until is not None and record.disabled_until <= now:
                record.disabled_until = None
                if not record.manual_disabled:
                    record.disable_reason = ""
                self._dirty = True
            return False, "", None, False

    def manual_disable(
        self,
        keys: list[str],
        *,
        minutes: int | None,
        reason: str = "手动禁用",
        now: float | None = None,
    ) -> int:
        now = time.time() if now is None else float(now)
        changed = 0
        with self._lock:
            for key in keys:
                record = self._records.get(key)
                if record is None:
                    continue
                if minutes is None:
                    record.manual_disabled = True
                    record.disabled_until = None
                else:
                    record.manual_disabled = False
                    record.disabled_until = now + max(1, minutes) * 60
                record.disable_reason = reason
                changed += 1
            if changed:
                self._dirty = True
        return changed

    def enable(self, keys: list[str]) -> int:
        changed = 0
        with self._lock:
            for key in keys:
                record = self._records.get(key)
                if record is None:
                    continue
                if record.manual_disabled or record.disabled_until is not None:
                    record.manual_disabled = False
                    record.disabled_until = None
                    record.disable_reason = ""
                    changed += 1
            if changed:
                self._dirty = True
        return changed

    def reset(self, keys: list[str] | None = None) -> int:
        with self._lock:
            if keys is None:
                count = len(self._records)
                self._records.clear()
                self._dirty = True
                return count
            count = 0
            for key in keys:
                if self._records.pop(key, None) is not None:
                    count += 1
            if count:
                self._dirty = True
            return count

    def match_keys(self, query: str) -> list[str]:
        normalized = query.strip().lower()
        with self._lock:
            if not normalized or normalized in {"all", "*", "全部"}:
                return sorted(self._records)
            return sorted(
                key
                for key, record in self._records.items()
                if normalized in key.lower()
                or normalized in record.label.lower()
                or normalized in record.provider_id.lower()
                or normalized in record.model.lower()
            )

    def records_snapshot(self) -> dict[str, ModelRecord]:
        with self._lock:
            return {
                key: ModelRecord.from_dict(record.to_dict())
                for key, record in self._records.items()
            }

    def save(self) -> bool:
        with self._lock:
            if not self._dirty:
                return False
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "models": {
                    key: record.to_dict() for key, record in self._records.items()
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)
            self._dirty = False
            return True

    def _event_contribution(
        self,
        event: EvidenceEvent,
        bucket: ContextBucket,
        now: float,
    ) -> float:
        local_share = 1.0 - event.global_share if event.bucket == bucket.value else 0.0
        return self._decayed_severity(event, now) * (
            event.global_share + local_share
        )

    def _decayed_severity(self, event: EvidenceEvent, now: float) -> float:
        age = max(0.0, now - event.timestamp)
        return event.severity * math.pow(0.5, age / self.half_life_seconds)

    def _prune_record(self, record: ModelRecord, now: float) -> None:
        cutoff = now - self.retention_seconds
        record.events = [
            event
            for event in record.events
            if event.timestamp >= cutoff and event.severity >= 0.02
        ]
        if len(record.events) > self.max_events_per_model:
            record.events = record.events[-self.max_events_per_model :]

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            models = payload.get("models", {}) if isinstance(payload, dict) else {}
            if not isinstance(models, dict):
                return
            loaded: dict[str, ModelRecord] = {}
            now = time.time()
            for key, raw in models.items():
                if not isinstance(key, str) or not isinstance(raw, dict):
                    continue
                record = ModelRecord.from_dict(raw)
                self._prune_record(record, now)
                loaded[key] = record
            self._records = loaded
            self._dirty = False
        except (OSError, ValueError, TypeError):
            # A corrupt ledger must never prevent AstrBot from starting. Preserve
            # the file for manual inspection and start with an empty in-memory state.
            try:
                backup = self.path.with_suffix(
                    f"{self.path.suffix}.corrupt-{int(time.time())}"
                )
                os.replace(self.path, backup)
            except OSError:
                pass
            self._records = {}
            self._dirty = False

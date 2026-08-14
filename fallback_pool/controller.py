from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .classifier import ErrorClassifier
from .domain import CandidateIdentity, ContextBucket, ErrorCategory, ErrorSignal
from .ledger import TrustLedger
from .scheduler import AdaptiveScheduler, RankingResult


@dataclass(slots=True)
class FallbackPoolSettings:
    enabled: bool = True
    half_life_minutes: float = 30.0
    burst_window_minutes: float = 10.0
    burst_to_bottom_count: int = 5
    success_retention_ratio: float = 0.65
    ambiguous_global_share: float = 0.65
    hard_quota_enabled: bool = True
    quota_disable_hours: float = 12.0
    daily_quota_disable_hours: float = 24.0
    log_reordering: bool = True
    store_error_summary: bool = True
    save_debounce_seconds: float = 0.5
    max_events_per_model: int = 200

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> FallbackPoolSettings:
        raw = raw or {}
        recovery = _mapping(raw.get("recovery"))
        penalty = _mapping(raw.get("penalty"))
        hard_disable = _mapping(raw.get("hard_disable"))
        diagnostics = _mapping(raw.get("diagnostics"))
        return cls(
            enabled=_as_bool(raw.get("enabled"), True),
            half_life_minutes=_bounded_float(
                recovery.get("half_life_minutes"), 30.0, 1.0, 24 * 60.0
            ),
            burst_window_minutes=_bounded_float(
                penalty.get("burst_window_minutes"), 10.0, 1.0, 24 * 60.0
            ),
            burst_to_bottom_count=_bounded_int(
                penalty.get("burst_to_bottom_count"), 5, 2, 100
            ),
            success_retention_ratio=_bounded_float(
                recovery.get("success_retention_ratio"), 0.65, 0.0, 1.0
            ),
            ambiguous_global_share=_bounded_float(
                penalty.get("ambiguous_global_share"), 0.65, 0.0, 1.0
            ),
            hard_quota_enabled=_as_bool(hard_disable.get("enabled"), True),
            quota_disable_hours=_bounded_float(
                hard_disable.get("quota_disable_hours"), 12.0, 0.1, 24 * 30.0
            ),
            daily_quota_disable_hours=_bounded_float(
                hard_disable.get("daily_quota_disable_hours"),
                24.0,
                0.1,
                24 * 30.0,
            ),
            log_reordering=_as_bool(diagnostics.get("log_reordering"), True),
            store_error_summary=_as_bool(
                diagnostics.get("store_error_summary"), True
            ),
            save_debounce_seconds=_bounded_float(
                diagnostics.get("save_debounce_seconds"), 0.5, 0.0, 30.0
            ),
            max_events_per_model=_bounded_int(
                diagnostics.get("max_events_per_model"), 200, 20, 2000
            ),
        )


class FallbackPoolController:
    """Coordinates classification, evidence updates, ranking and persistence."""

    def __init__(
        self,
        data_dir: Path,
        settings: FallbackPoolSettings,
        *,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self._clock = clock or time.time
        self.classifier = ErrorClassifier(settings.ambiguous_global_share)
        self.ledger = TrustLedger(
            data_dir / "trust_ledger.json",
            half_life_seconds=settings.half_life_minutes * 60,
            burst_window_seconds=settings.burst_window_minutes * 60,
            success_retention_ratio=settings.success_retention_ratio,
            max_events_per_model=settings.max_events_per_model,
        )
        self.scheduler = AdaptiveScheduler(
            self.ledger,
            burst_to_bottom_count=settings.burst_to_bottom_count,
        )
        self._save_task: asyncio.Task[None] | None = None
        self._recent_failure_tokens: dict[str, float] = {}
        self._patch_error = ""

    @property
    def patch_error(self) -> str:
        return self._patch_error

    def set_patch_error(self, message: str) -> None:
        self._patch_error = message

    def identity_for(
        self,
        provider: Any,
        *,
        override_model: str | None = None,
    ) -> CandidateIdentity:
        config = getattr(provider, "provider_config", None)
        if not isinstance(config, dict):
            config = {}
        provider_id = str(
            config.get("id")
            or config.get("name")
            or getattr(provider, "id", "")
            or type(provider).__name__
        )
        model = str(
            override_model
            or self._safe_get_model(provider)
            or config.get("model")
            or config.get("model_name")
            or "<default>"
        )
        key = f"{provider_id}::{model}"
        label = provider_id if model in {"", "<default>"} else f"{provider_id} / {model}"
        return CandidateIdentity(
            key=key,
            provider_id=provider_id,
            model=model,
            label=label,
        )

    def rank_candidates(
        self,
        runner: Any,
        candidates: list[Any],
        original_primary: Any,
    ) -> RankingResult:
        bucket = self.context_bucket(runner)
        requested_model = getattr(getattr(runner, "req", None), "model", None)

        def identity_for(candidate: Any) -> CandidateIdentity:
            override = requested_model if candidate is original_primary else None
            return self.identity_for(candidate, override_model=override)

        result = self.scheduler.rank(
            candidates,
            identity_for,
            bucket,
            now=self._clock(),
        )
        if self.settings.log_reordering:
            self._log_ranking(result)
        return result

    async def record_exception(
        self,
        runner: Any,
        provider: Any,
        exc: Exception,
        *,
        original_primary: Any,
    ) -> None:
        signal = self.classifier.from_exception(exc)
        await self._record_failure(
            runner,
            provider,
            signal,
            original_primary=original_primary,
        )

    async def record_error_response(
        self,
        runner: Any,
        provider: Any,
        response: Any,
        *,
        original_primary: Any,
    ) -> None:
        signal = self.classifier.from_response(response)
        await self._record_failure(
            runner,
            provider,
            signal,
            original_primary=original_primary,
        )

    async def record_success(
        self,
        runner: Any,
        provider: Any,
        *,
        original_primary: Any,
        elapsed_seconds: float | None = None,
    ) -> None:
        identity = self._identity_for_attempt(runner, provider, original_primary)
        bucket = self.context_bucket(runner)
        self.ledger.record_success(
            identity,
            bucket,
            now=self._clock(),
        )
        self._clear_failure_dedupe(runner, identity.key)
        self._schedule_save()
        if elapsed_seconds is not None:
            self.logger.debug(
                "Fallback pool success: %s (%.2fs)",
                identity.label,
                elapsed_seconds,
            )

    async def _record_failure(
        self,
        runner: Any,
        provider: Any,
        signal: ErrorSignal,
        *,
        original_primary: Any,
    ) -> None:
        if signal.category == ErrorCategory.CANCELLED:
            return
        identity = self._identity_for_attempt(runner, provider, original_primary)
        token = self._failure_token(runner, identity.key, signal.category.value)
        now = self._clock()
        if self._is_duplicate_failure(token, now):
            return

        if not self.settings.store_error_summary:
            signal = ErrorSignal(
                category=signal.category,
                severity=signal.severity,
                global_share=signal.global_share,
                summary=signal.category.value,
                status_code=signal.status_code,
                retry_after_seconds=signal.retry_after_seconds,
                hard_disable=signal.hard_disable,
                daily_quota=signal.daily_quota,
            )

        hard_disable_seconds: float | None = None
        if signal.hard_disable and self.settings.hard_quota_enabled:
            hours = (
                self.settings.daily_quota_disable_hours
                if signal.daily_quota
                else self.settings.quota_disable_hours
            )
            hard_disable_seconds = hours * 3600

        self.ledger.record_failure(
            identity,
            signal,
            self.context_bucket(runner),
            now=now,
            hard_disable_seconds=hard_disable_seconds,
        )
        self._recent_failure_tokens[token] = now
        self._prune_failure_dedupe(now)
        self._schedule_save()
        self.logger.warning(
            "Fallback pool failure: %s category=%s status=%s hard_disable=%s",
            identity.label,
            signal.category.value,
            signal.status_code,
            bool(hard_disable_seconds),
        )

    def context_bucket(self, runner: Any) -> ContextBucket:
        weighted_size = self._estimate_request_load(runner)
        if weighted_size < 32_000:
            return ContextBucket.SMALL
        if weighted_size < 128_000:
            return ContextBucket.MEDIUM
        if weighted_size < 320_000:
            return ContextBucket.LARGE
        return ContextBucket.HUGE

    def status_text(self, *, limit: int = 20) -> str:
        now = self._clock()
        records = self.ledger.records_snapshot()
        if not records:
            base = "智能回退模型池：尚无调用记录。"
            if self._patch_error:
                return f"{base}\n补丁状态：未生效：{self._patch_error}"
            return f"{base}\n补丁状态：已加载。"

        rows: list[tuple[float, str]] = []
        for key, record in records.items():
            disabled, reason, disabled_until, manual = self.ledger.disabled_state(
                key,
                now=now,
            )
            global_score = self.ledger.global_evidence(key, now=now)
            bucket_scores = {
                bucket.value: global_score
                + self.ledger.bucket_evidence(key, bucket, now=now)
                for bucket in ContextBucket
            }
            max_bucket, max_score = max(
                bucket_scores.items(),
                key=lambda item: item[1],
            )
            if disabled:
                if manual:
                    state = f"手动禁用（{reason}）"
                else:
                    state = f"暂停至 {_format_timestamp(disabled_until)}（{reason}）"
                sort_score = 10_000 + max_score
            elif max_score >= 4.25:
                state = "高降权"
                sort_score = max_score
            elif max_score >= 1.45:
                state = "中降权"
                sort_score = max_score
            elif max_score >= 0.55:
                state = "轻降权"
                sort_score = max_score
            else:
                state = "健康"
                sort_score = max_score

            detail = (
                f"- {record.label}\n"
                f"  状态：{state}；全局证据 {global_score:.2f}；"
                f"最高负载桶 {max_bucket}={max_score:.2f}\n"
                f"  成功/失败：{record.success_count}/{record.failure_count}；"
                f"最近失败：{record.last_category or '无'}"
            )
            rows.append((sort_score, detail))

        rows.sort(key=lambda item: item[0], reverse=True)
        header = "智能回退模型池状态"
        if self._patch_error:
            header += f"\n补丁状态：未生效：{self._patch_error}"
        else:
            header += "\n补丁状态：已加载"
        return "\n".join([header, *(row for _, row in rows[: max(1, limit)])])

    def reset(self, target: str = "all") -> tuple[int, list[str]]:
        if target.strip().lower() in {"all", "*", "全部", ""}:
            keys = self.ledger.match_keys("all")
            count = self.ledger.reset(None)
        else:
            keys = self.ledger.match_keys(target)
            count = self.ledger.reset(keys)
        self._schedule_save(immediate=True)
        return count, keys

    def disable(
        self,
        target: str,
        *,
        minutes: int | None,
    ) -> tuple[int, list[str]]:
        keys = self.ledger.match_keys(target)
        count = self.ledger.manual_disable(
            keys,
            minutes=minutes,
            reason=(
                "手动无限期禁用"
                if minutes is None
                else f"手动禁用 {max(1, minutes)} 分钟"
            ),
            now=self._clock(),
        )
        self._schedule_save(immediate=True)
        return count, keys

    def enable(self, target: str) -> tuple[int, list[str]]:
        keys = self.ledger.match_keys(target)
        count = self.ledger.enable(keys)
        self._schedule_save(immediate=True)
        return count, keys

    async def close(self) -> None:
        task = self._save_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self.ledger.save)

    def _identity_for_attempt(
        self,
        runner: Any,
        provider: Any,
        original_primary: Any,
    ) -> CandidateIdentity:
        requested_model = getattr(getattr(runner, "req", None), "model", None)
        override = requested_model if provider is original_primary else None
        return self.identity_for(provider, override_model=override)

    def _failure_token(self, runner: Any, model_key: str, category: str) -> str:
        signature = self._request_signature(runner)
        return f"{id(runner)}::{model_key}::{category}::{signature}"

    def _request_signature(self, runner: Any) -> str:
        messages = getattr(getattr(runner, "run_context", None), "messages", None)
        if messages is None:
            messages = getattr(getattr(runner, "req", None), "contexts", None)
        material = [str(len(messages or []))]
        if messages:
            for message in list(messages)[-2:]:
                material.append(self._compact_value(message, limit=1500))
        digest = hashlib.sha1(
            "\n".join(material).encode("utf-8", errors="ignore")
        ).hexdigest()
        return digest[:16]

    def _is_duplicate_failure(self, token: str, now: float) -> bool:
        previous = self._recent_failure_tokens.get(token)
        return previous is not None and now - previous < 8.0

    def _clear_failure_dedupe(self, runner: Any, model_key: str) -> None:
        prefix = f"{id(runner)}::{model_key}::"
        stale = [key for key in self._recent_failure_tokens if key.startswith(prefix)]
        for key in stale:
            self._recent_failure_tokens.pop(key, None)

    def _prune_failure_dedupe(self, now: float) -> None:
        cutoff = now - 120.0
        stale = [
            key
            for key, timestamp in self._recent_failure_tokens.items()
            if timestamp < cutoff
        ]
        for key in stale:
            self._recent_failure_tokens.pop(key, None)

    def _schedule_save(self, *, immediate: bool = False) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.ledger.save()
            return

        if immediate or self.settings.save_debounce_seconds <= 0:
            if self._save_task and not self._save_task.done():
                self._save_task.cancel()
            self._save_task = loop.create_task(self._save_after(0.0))
            return

        if self._save_task and not self._save_task.done():
            return
        self._save_task = loop.create_task(
            self._save_after(self.settings.save_debounce_seconds)
        )

    async def _save_after(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await asyncio.to_thread(self.ledger.save)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error("Failed to save fallback-pool ledger: %s", exc)

    def _log_ranking(self, result: RankingResult) -> None:
        if not result.active and not result.disabled:
            return
        changed = any(
            candidate.base_index != index
            for index, candidate in enumerate(result.active)
        ) or bool(result.disabled)
        if not changed:
            return
        active = " -> ".join(
            f"{item.identity.label}[证据={item.evidence:.2f},后移={item.shift}]"
            for item in result.active
        )
        disabled = ", ".join(
            f"{item.identity.label}({item.reason})" for item in result.disabled
        )
        message = f"Fallback pool reordered: {active or '<none>'}"
        if disabled:
            message += f"; skipped: {disabled}"
        self.logger.info(message)

    def _estimate_request_load(self, runner: Any) -> int:
        total = 0
        run_context = getattr(runner, "run_context", None)
        messages = getattr(run_context, "messages", None)
        if messages:
            for message in messages:
                total += self._measure_value(message, budget=500_000)

        req = getattr(runner, "req", None)
        if req is not None:
            total += len(str(getattr(req, "prompt", "") or ""))
            total += len(str(getattr(req, "system_prompt", "") or ""))
            total += 16_000 * len(getattr(req, "image_urls", None) or [])
            total += 24_000 * len(getattr(req, "audio_urls", None) or [])
            func_tool = getattr(req, "func_tool", None)
            if func_tool is not None:
                total += min(120_000, len(self._compact_value(func_tool, limit=120_000)))
        return total

    def _measure_value(self, value: Any, *, budget: int) -> int:
        if budget <= 0 or value is None:
            return 0
        if isinstance(value, str):
            lowered = value[:64].lower()
            if lowered.startswith("data:image/") or lowered.startswith("data:audio/"):
                return min(budget, 16_000)
            return min(budget, len(value))
        if isinstance(value, bytes):
            return min(budget, 16_000)
        if isinstance(value, dict):
            total = 0
            for key, item in value.items():
                total += min(128, len(str(key)))
                total += self._measure_value(item, budget=budget - total)
                if total >= budget:
                    return budget
            return total
        if isinstance(value, list | tuple | set):
            total = 0
            for item in value:
                total += self._measure_value(item, budget=budget - total)
                if total >= budget:
                    return budget
            return total
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return self._measure_value(model_dump(), budget=budget)
            except Exception:
                pass
        raw_dict = getattr(value, "__dict__", None)
        if isinstance(raw_dict, dict):
            return self._measure_value(raw_dict, budget=budget)
        return min(budget, len(str(value)))

    def _compact_value(self, value: Any, *, limit: int) -> str:
        try:
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                value = model_dump()
            text = str(value)
        except Exception:
            text = f"<{type(value).__name__}>"
        return text[:limit]

    @staticmethod
    def _safe_get_model(provider: Any) -> str:
        getter = getattr(provider, "get_model", None)
        if callable(getter):
            try:
                value = getter()
                if value:
                    return str(value)
            except Exception:
                pass
        value = getattr(provider, "model_name", None)
        return str(value) if value else ""


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return default


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))


def _format_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "未知"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

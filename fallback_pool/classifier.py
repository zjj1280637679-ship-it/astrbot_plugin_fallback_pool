from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .domain import ErrorCategory, ErrorSignal

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(api[_ -]?key\s*[:=]\s*)[A-Za-z0-9._~+/=-]{8,}"),
)

_HARD_QUOTA_PATTERNS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "billing_hard_limit_reached",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "project_usage_limit_exceeded",
    "account balance is insufficient",
    "insufficient balance",
    "余额不足",
    "账户余额不足",
    "额度耗尽",
    "额度已用完",
    "配额已用完",
    "欠费",
    "充值后重试",
)

_DAILY_QUOTA_PATTERNS = (
    "daily quota",
    "daily limit",
    "requests per day",
    "request per day",
    "rpd limit",
    "per-day quota",
    "per day quota",
    "每日额度",
    "日额度",
    "每日配额",
    "每天请求",
)

_OVERLOAD_PATTERNS = (
    "overloaded",
    "overload",
    "server is busy",
    "server busy",
    "temporarily unavailable",
    "service unavailable",
    "too many requests",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "请求过于频繁",
    "请求频繁",
    "服务器繁忙",
    "服务繁忙",
    "系统繁忙",
    "并发限制",
)

_TRANSIENT_PATTERNS = (
    "timeout",
    "timed out",
    "readtimeout",
    "connecttimeout",
    "connection reset",
    "connection aborted",
    "connection closed",
    "connection error",
    "remote protocol error",
    "eof",
    "empty model output",
    "emptymodeloutput",
    "gateway timeout",
    "bad gateway",
    "网络超时",
    "连接重置",
    "连接中断",
    "连接失败",
    "空输出",
)

_CONTEXT_PATTERNS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "prompt is too long",
    "input is too long",
    "token limit exceeded",
    "上下文长度",
    "上下文过长",
    "超出上下文",
    "输入过长",
    "token 数量超限",
)

_AUTH_PATTERNS = (
    "invalid_api_key",
    "invalid api key",
    "authentication_error",
    "authentication failed",
    "unauthorized",
    "permission denied",
    "forbidden",
    "密钥无效",
    "认证失败",
    "无权限",
)

_REQUEST_PATTERNS = (
    "invalid_request_error",
    "invalid request",
    "model not found",
    "does not exist",
    "unsupported parameter",
    "unsupported modality",
    "tool use is not supported",
    "function calling is not supported",
    "参数错误",
    "模型不存在",
    "不支持工具",
    "不支持该模态",
)

_STATUS_RE = re.compile(r"(?<!\d)(4\d\d|5\d\d)(?!\d)")
_RETRY_AFTER_RE = re.compile(
    r"(?i)(?:retry[-_ ]?after|retry after)\s*[:=]?\s*(\d+(?:\.\d+)?)"
)


class ErrorClassifier:
    """Normalize provider-specific failures into a small policy vocabulary."""

    def __init__(self, ambiguous_global_share: float = 0.65) -> None:
        self.ambiguous_global_share = min(1.0, max(0.0, ambiguous_global_share))

    def from_exception(self, exc: Exception) -> ErrorSignal:
        type_name = type(exc).__name__
        raw = self._collect_exception_text(exc)
        status_code = self._extract_status_code(exc, raw)
        retry_after = self._extract_retry_after(exc, raw)
        return self.from_text(
            f"{type_name}: {raw}",
            status_code=status_code,
            retry_after_seconds=retry_after,
        )

    def from_response(self, response: Any) -> ErrorSignal:
        text = str(getattr(response, "completion_text", "") or "")
        raw_completion = getattr(response, "raw_completion", None)
        if raw_completion is not None:
            text = f"{text} {self._safe_repr(raw_completion)}"
        status_code = self._extract_status_code(response, text)
        retry_after = self._extract_retry_after(response, text)
        return self.from_text(
            text or "LLM returned an error response",
            status_code=status_code,
            retry_after_seconds=retry_after,
        )

    def from_text(
        self,
        text: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> ErrorSignal:
        normalized = " ".join(text.lower().split())
        summary = sanitize_error_summary(text)
        status_code = status_code or self._status_from_text(normalized)

        if self._contains_any(normalized, _HARD_QUOTA_PATTERNS):
            daily = self._contains_any(normalized, _DAILY_QUOTA_PATTERNS)
            return ErrorSignal(
                category=ErrorCategory.QUOTA,
                severity=3.0,
                global_share=1.0,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
                hard_disable=True,
                daily_quota=daily,
            )

        if self._contains_any(normalized, _DAILY_QUOTA_PATTERNS):
            return ErrorSignal(
                category=ErrorCategory.QUOTA,
                severity=3.0,
                global_share=1.0,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
                hard_disable=True,
                daily_quota=True,
            )

        if self._contains_any(normalized, _CONTEXT_PATTERNS):
            return ErrorSignal(
                category=ErrorCategory.CONTEXT,
                severity=1.0,
                global_share=0.0,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )

        if self._contains_any(normalized, _AUTH_PATTERNS) or status_code in {401, 403}:
            return ErrorSignal(
                category=ErrorCategory.AUTH,
                severity=0.0,
                global_share=0.0,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )

        if self._contains_any(normalized, _REQUEST_PATTERNS) or status_code in {
            400,
            404,
            405,
            422,
        }:
            return ErrorSignal(
                category=ErrorCategory.REQUEST,
                severity=0.0,
                global_share=0.0,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )

        if (
            self._contains_any(normalized, _OVERLOAD_PATTERNS)
            or status_code in {429, 503, 529}
        ):
            return ErrorSignal(
                category=ErrorCategory.OVERLOAD,
                severity=1.25,
                global_share=1.0,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )

        if (
            self._contains_any(normalized, _TRANSIENT_PATTERNS)
            or status_code in {408, 500, 502, 504}
        ):
            return ErrorSignal(
                category=ErrorCategory.TRANSIENT,
                severity=1.0,
                global_share=self.ambiguous_global_share,
                summary=summary,
                status_code=status_code,
                retry_after_seconds=retry_after_seconds,
            )

        return ErrorSignal(
            category=ErrorCategory.UNKNOWN,
            severity=1.0,
            global_share=self.ambiguous_global_share,
            summary=summary,
            status_code=status_code,
            retry_after_seconds=retry_after_seconds,
        )

    @staticmethod
    def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in text for pattern in patterns)

    def _collect_exception_text(self, exc: Exception) -> str:
        parts = [str(exc)]
        for attr in ("message", "body", "code", "type", "error"):
            value = getattr(exc, attr, None)
            if value is not None:
                parts.append(self._safe_repr(value))
        response = getattr(exc, "response", None)
        if response is not None:
            for attr in ("text", "reason_phrase", "content"):
                value = getattr(response, attr, None)
                if value:
                    parts.append(self._safe_repr(value))
        return " ".join(part for part in parts if part)

    def _extract_status_code(self, obj: Any, text: str) -> int | None:
        for candidate in (
            getattr(obj, "status_code", None),
            getattr(getattr(obj, "response", None), "status_code", None),
            getattr(getattr(obj, "response", None), "status", None),
        ):
            if isinstance(candidate, int) and 100 <= candidate <= 599:
                return candidate
        return self._status_from_text(text)

    def _extract_retry_after(self, obj: Any, text: str) -> float | None:
        response = getattr(obj, "response", None)
        headers = getattr(response, "headers", None) or getattr(obj, "headers", None)
        if isinstance(headers, Mapping):
            for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
                parsed = _to_non_negative_float(headers.get(key))
                if parsed is not None:
                    return parsed
        match = _RETRY_AFTER_RE.search(text)
        if match:
            return _to_non_negative_float(match.group(1))
        return None

    @staticmethod
    def _status_from_text(text: str) -> int | None:
        match = _STATUS_RE.search(text)
        if not match:
            return None
        value = int(match.group(1))
        return value if 100 <= value <= 599 else None

    @staticmethod
    def _safe_repr(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping | list | tuple):
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                pass
        try:
            return str(value)
        except Exception:
            return f"<{type(value).__name__}>"


def sanitize_error_summary(text: str, max_length: int = 180) -> str:
    """Remove likely credentials and keep a compact, single-line diagnostic."""

    cleaned = " ".join(str(text).split())
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(lambda match: f"{match.group(1) if match.lastindex else ''}[REDACTED]", cleaned)
    if len(cleaned) > max_length:
        return f"{cleaned[: max_length - 1]}…"
    return cleaned


def _to_non_negative_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None

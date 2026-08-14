from fallback_pool.classifier import ErrorClassifier, sanitize_error_summary
from fallback_pool.domain import ErrorCategory


def test_plain_429_is_soft_overload() -> None:
    signal = ErrorClassifier().from_text("HTTP 429 Too Many Requests")
    assert signal.category == ErrorCategory.OVERLOAD
    assert signal.hard_disable is False
    assert signal.global_share == 1.0


def test_explicit_quota_is_hard_disable() -> None:
    signal = ErrorClassifier().from_text(
        "429 insufficient_quota: credit balance exhausted"
    )
    assert signal.category == ErrorCategory.QUOTA
    assert signal.hard_disable is True


def test_daily_quota_is_recognized() -> None:
    signal = ErrorClassifier().from_text("Requests per day daily quota exceeded")
    assert signal.category == ErrorCategory.QUOTA
    assert signal.daily_quota is True
    assert signal.hard_disable is True


def test_context_error_is_local_evidence() -> None:
    signal = ErrorClassifier().from_text(
        "context_length_exceeded: maximum context length reached"
    )
    assert signal.category == ErrorCategory.CONTEXT
    assert signal.global_share == 0.0
    assert signal.hard_disable is False


def test_error_summary_redacts_credentials() -> None:
    summary = sanitize_error_summary(
        "Authorization: Bearer abcdefghijklmnop and sk-supersecretvalue"
    )
    assert "abcdefghijklmnop" not in summary
    assert "sk-supersecretvalue" not in summary
    assert "[REDACTED]" in summary

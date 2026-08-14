from __future__ import annotations

import inspect
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from .controller import FallbackPoolController

_PATCH_TOKEN_ATTR = "__fallback_pool_patch_token__"
_ORIGINAL_ATTR = "__fallback_pool_original__"


class RunnerPatch:
    """Minimal wrappers around AstrBot's existing fallback implementation.

    The plugin deliberately does not copy or replace AstrBot's fallback loop.
    It only changes the candidate order before the original loop runs, fixes the
    original-primary model override after reordering, and observes per-candidate
    results through the original single-provider iterator.
    """

    def __init__(self) -> None:
        self.token = uuid.uuid4().hex
        self.runner_cls: type[Any] | None = None
        self.llm_response_cls: type[Any] | None = None
        self._previous_reset: Any = None
        self._previous_fallback: Any = None
        self._previous_single: Any = None
        self._installed_reset: Any = None
        self._installed_fallback: Any = None
        self._installed_single: Any = None
        self.installed = False

    def install(
        self,
        controller: FallbackPoolController,
        *,
        runner_cls: type[Any] | None = None,
        llm_response_cls: type[Any] | None = None,
    ) -> None:
        if self.installed:
            return
        if runner_cls is None or llm_response_cls is None:
            from astrbot.core.agent.runners.tool_loop_agent_runner import (
                ToolLoopAgentRunner,
            )
            from astrbot.core.provider.entities import LLMResponse

            runner_cls = runner_cls or ToolLoopAgentRunner
            llm_response_cls = llm_response_cls or LLMResponse

        self._validate_runner_shape(runner_cls)
        self.runner_cls = runner_cls
        self.llm_response_cls = llm_response_cls

        previous_reset = _unwrap_our_patch(getattr(runner_cls, "reset"))
        previous_fallback = _unwrap_our_patch(
            getattr(runner_cls, "_iter_llm_responses_with_fallback")
        )
        previous_single = _unwrap_our_patch(
            getattr(runner_cls, "_iter_llm_responses")
        )
        self._previous_reset = previous_reset
        self._previous_fallback = previous_fallback
        self._previous_single = previous_single
        token = self.token

        async def patched_reset(runner: Any, *args: Any, **kwargs: Any) -> None:
            await previous_reset(runner, *args, **kwargs)
            base_candidates = _dedupe_by_identity(
                [
                    getattr(runner, "provider", None),
                    *(getattr(runner, "fallback_providers", None) or []),
                ]
            )
            if not base_candidates:
                return
            runner.__fallback_pool_base_candidates__ = base_candidates
            runner.__fallback_pool_original_primary__ = base_candidates[0]
            runner.__fallback_pool_controller__ = controller

        async def patched_fallback(
            runner: Any,
        ) -> AsyncGenerator[Any, None]:
            bound_controller = getattr(
                runner,
                "__fallback_pool_controller__",
                controller,
            )
            if not bound_controller.settings.enabled:
                async for response in previous_fallback(runner):
                    yield response
                return

            base_candidates = list(
                getattr(runner, "__fallback_pool_base_candidates__", [])
            )
            if not base_candidates:
                base_candidates = _dedupe_by_identity(
                    [
                        getattr(runner, "provider", None),
                        *(getattr(runner, "fallback_providers", None) or []),
                    ]
                )
            if not base_candidates:
                yield llm_response_cls(
                    role="err",
                    completion_text="No chat provider is available.",
                )
                return

            original_primary = getattr(
                runner,
                "__fallback_pool_original_primary__",
                base_candidates[0],
            )
            ranking = bound_controller.rank_candidates(
                runner,
                base_candidates,
                original_primary,
            )
            ordered = ranking.ordered_items
            if not ordered:
                disabled_labels = ", ".join(
                    item.identity.label for item in ranking.disabled
                )
                yield llm_response_cls(
                    role="err",
                    completion_text=(
                        "All chat models are temporarily disabled by the "
                        "adaptive fallback pool."
                        + (f" Disabled: {disabled_labels}" if disabled_labels else "")
                    ),
                )
                return

            runner.provider = ordered[0]
            runner.fallback_providers = ordered[1:]
            runner.__fallback_pool_original_primary__ = original_primary
            async for response in previous_fallback(runner):
                yield response

        async def patched_single(
            runner: Any,
            *,
            include_model: bool = True,
        ) -> AsyncGenerator[Any, None]:
            bound_controller = getattr(
                runner,
                "__fallback_pool_controller__",
                controller,
            )
            if not bound_controller.settings.enabled:
                async for response in previous_single(
                    runner,
                    include_model=include_model,
                ):
                    yield response
                return

            provider = getattr(runner, "provider", None)
            original_primary = getattr(
                runner,
                "__fallback_pool_original_primary__",
                provider,
            )
            # Once the pool is reordered, idx == 0 no longer means "the provider
            # whose explicit request.model belongs here". Preserve that semantic
            # by keying it to provider identity instead.
            effective_include_model = provider is original_primary
            started = time.monotonic()
            saw_chunk = False
            saw_terminal = False
            try:
                async for response in previous_single(
                    runner,
                    include_model=effective_include_model,
                ):
                    if bool(getattr(response, "is_chunk", False)):
                        saw_chunk = True
                    else:
                        saw_terminal = True
                        if getattr(response, "role", None) == "err":
                            await bound_controller.record_error_response(
                                runner,
                                provider,
                                response,
                                original_primary=original_primary,
                            )
                        else:
                            await bound_controller.record_success(
                                runner,
                                provider,
                                original_primary=original_primary,
                                elapsed_seconds=time.monotonic() - started,
                            )
                    yield response
                if saw_chunk and not saw_terminal:
                    await bound_controller.record_success(
                        runner,
                        provider,
                        original_primary=original_primary,
                        elapsed_seconds=time.monotonic() - started,
                    )
            except Exception as exc:
                await bound_controller.record_exception(
                    runner,
                    provider,
                    exc,
                    original_primary=original_primary,
                )
                raise

        _mark_wrapper(patched_reset, previous_reset)
        _mark_wrapper(patched_fallback, previous_fallback)
        _mark_wrapper(patched_single, previous_single)
        setattr(runner_cls, "reset", patched_reset)
        setattr(runner_cls, "_iter_llm_responses_with_fallback", patched_fallback)
        setattr(runner_cls, "_iter_llm_responses", patched_single)
        self._installed_reset = patched_reset
        self._installed_fallback = patched_fallback
        self._installed_single = patched_single
        setattr(runner_cls, _PATCH_TOKEN_ATTR, token)
        self.installed = True

    def uninstall(self) -> None:
        runner_cls = self.runner_cls
        if not self.installed or runner_cls is None:
            return
        # A newer hot-reloaded instance may already own the class. Never let an
        # older instance's terminate() remove the newer patch.
        owns_patch = (
            getattr(runner_cls, _PATCH_TOKEN_ATTR, None) == self.token
            and getattr(runner_cls, "reset", None) is self._installed_reset
            and getattr(runner_cls, "_iter_llm_responses_with_fallback", None)
            is self._installed_fallback
            and getattr(runner_cls, "_iter_llm_responses", None)
            is self._installed_single
        )
        if not owns_patch:
            self.installed = False
            return
        setattr(runner_cls, "reset", self._previous_reset)
        setattr(
            runner_cls,
            "_iter_llm_responses_with_fallback",
            self._previous_fallback,
        )
        setattr(runner_cls, "_iter_llm_responses", self._previous_single)
        try:
            delattr(runner_cls, _PATCH_TOKEN_ATTR)
        except AttributeError:
            pass
        self.installed = False

    @staticmethod
    def _validate_runner_shape(runner_cls: type[Any]) -> None:
        required = (
            "reset",
            "_iter_llm_responses_with_fallback",
            "_iter_llm_responses",
        )
        missing = [name for name in required if not hasattr(runner_cls, name)]
        if missing:
            raise RuntimeError(
                "AstrBot ToolLoopAgentRunner is incompatible; missing: "
                + ", ".join(missing)
            )
        signature = inspect.signature(getattr(runner_cls, "_iter_llm_responses"))
        if "include_model" not in signature.parameters:
            raise RuntimeError(
                "AstrBot ToolLoopAgentRunner is incompatible; "
                "_iter_llm_responses has no include_model parameter."
            )


def _dedupe_by_identity(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[int] = set()
    for item in items:
        if item is None:
            continue
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def _mark_wrapper(wrapper: Any, original: Any) -> None:
    setattr(wrapper, _ORIGINAL_ATTR, original)


def _unwrap_our_patch(function: Any) -> Any:
    return getattr(function, _ORIGINAL_ATTR, function)

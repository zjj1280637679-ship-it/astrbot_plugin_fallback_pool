import asyncio
from pathlib import Path
from types import SimpleNamespace

from fallback_pool.controller import FallbackPoolController, FallbackPoolSettings
from fallback_pool.domain import ContextBucket, ErrorCategory, ErrorSignal
from fallback_pool.patcher import RunnerPatch


class FakeResponse:
    def __init__(self, role: str = "assistant", text: str = "ok", is_chunk: bool = False):
        self.role = role
        self.completion_text = text
        self.is_chunk = is_chunk
        self.raw_completion = None


class FakeProvider:
    def __init__(self, provider_id: str, model: str) -> None:
        self.provider_config = {"id": provider_id, "model": model}
        self.model_name = model
        self.outcome: FakeResponse | Exception = FakeResponse()
        self.include_model_calls: list[bool] = []

    def get_model(self) -> str:
        return self.model_name


class FakeRunner:
    async def reset(
        self,
        provider: FakeProvider,
        request: SimpleNamespace,
        *,
        fallback_providers: list[FakeProvider] | None = None,
        **_: object,
    ) -> None:
        self.provider = provider
        self.req = request
        self.fallback_providers = list(fallback_providers or [])
        self.run_context = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])

    async def _iter_llm_responses(self, *, include_model: bool = True):
        provider = self.provider
        provider.include_model_calls.append(include_model)
        if isinstance(provider.outcome, Exception):
            raise provider.outcome
        yield provider.outcome

    async def _iter_llm_responses_with_fallback(self):
        candidates = [self.provider, *self.fallback_providers]
        for index, candidate in enumerate(candidates):
            self.provider = candidate
            try:
                async for response in self._iter_llm_responses(
                    include_model=index == 0
                ):
                    if response.role == "err" and index < len(candidates) - 1:
                        break
                    yield response
                    return
            except Exception:
                continue
        yield FakeResponse(role="err", text="all failed")


def make_controller(path: Path) -> FallbackPoolController:
    return FallbackPoolController(
        path,
        FallbackPoolSettings(save_debounce_seconds=0),
    )


def test_reordered_fallback_does_not_receive_primary_model_override(tmp_path: Path) -> None:
    async def scenario() -> None:
        controller = make_controller(tmp_path)
        patch = RunnerPatch()
        patch.install(controller, runner_cls=FakeRunner, llm_response_cls=FakeResponse)
        try:
            a = FakeProvider("A", "model-a")
            b = FakeProvider("B", "model-b")
            c = FakeProvider("C", "model-c")
            controller.ledger.record_failure(
                controller.identity_for(a, override_model="requested-a"),
                ErrorSignal(
                    category=ErrorCategory.TRANSIENT,
                    severity=1.0,
                    global_share=1.0,
                    summary="timeout",
                ),
                ContextBucket.SMALL,
                now=controller._clock(),
            )
            runner = FakeRunner()
            await runner.reset(
                a,
                SimpleNamespace(model="requested-a", contexts=[]),
                fallback_providers=[b, c],
            )
            responses = [item async for item in runner._iter_llm_responses_with_fallback()]

            assert responses[-1].role == "assistant"
            assert b.include_model_calls == [False]
            assert a.include_model_calls == []
        finally:
            patch.uninstall()
            await controller.close()

    asyncio.run(scenario())


def test_original_primary_keeps_model_override_when_tried_later(tmp_path: Path) -> None:
    async def scenario() -> None:
        controller = make_controller(tmp_path)
        patch = RunnerPatch()
        patch.install(controller, runner_cls=FakeRunner, llm_response_cls=FakeResponse)
        try:
            a = FakeProvider("A", "model-a")
            b = FakeProvider("B", "model-b")
            c = FakeProvider("C", "model-c")
            b.outcome = RuntimeError("temporary timeout")
            controller.ledger.record_failure(
                controller.identity_for(a, override_model="requested-a"),
                ErrorSignal(
                    category=ErrorCategory.TRANSIENT,
                    severity=1.0,
                    global_share=1.0,
                    summary="timeout",
                ),
                ContextBucket.SMALL,
                now=controller._clock(),
            )
            runner = FakeRunner()
            await runner.reset(
                a,
                SimpleNamespace(model="requested-a", contexts=[]),
                fallback_providers=[b, c],
            )
            responses = [item async for item in runner._iter_llm_responses_with_fallback()]

            assert responses[-1].role == "assistant"
            assert b.include_model_calls == [False]
            assert a.include_model_calls == [True]
        finally:
            patch.uninstall()
            await controller.close()

    asyncio.run(scenario())

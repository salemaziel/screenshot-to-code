"""RED test: local-runner registration + CodeGenerationMiddleware dispatch (v0.1).

This is the third vertical slice: wiring a configured local runner (Codex) into
the existing WebSocket code-generation pipeline. The approved adapter
(``routes.local_runner_adapter``) is already covered by
``test_local_runner_websocket.py`` and is NOT under test here; these tests pin
the registration/selection rules, the single-variant announcement, the
once-per-runner probe, the fail-closed rules for unsupported/unavailable local
requests, and the CompleteEvent text_files -> setCode ordering requirement.

The production symbols under test (``_select_local_runner``,
``_local_dispatch_decision``, ``_expand_complete_text_files`` and the local
branch of ``CodeGenerationMiddleware`` / ``StatusBroadcastMiddleware``) do not
exist yet, so this file currently fails at import (strict TDD RED: the failure
is the *absence of the dispatch logic*, not a malformed test setup).
"""

from types import SimpleNamespace
import asyncio
from typing import Any, AsyncIterator, Iterator, List, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import runners.registry as registry
import routes.generate_code as gen_code
from routes.generate_code import (
    CodeGenerationMiddleware,
    ExtractedParams,
    PipelineContext,
    PromptCreationMiddleware,
    StatusBroadcastMiddleware,
    _expand_complete_text_files,
    _local_dispatch_decision,
    _probe_runner_once,
    _select_local_runner,
)
from prompts.prompt_types import UserTurnInput
from runners.models import (
    CompleteEvent,
    ErrorEvent,
    GenerationEvent,
    GenerationInput,
    GenerationResult,
    ProbeResult,
    RunnerCapabilities,
    SetCodeEvent,
)

UNSUPPORTED_MESSAGE = "This request is not supported by the configured local runner."


def _extracted(
    stack: str = "html_tailwind",
    input_mode: str = "text",
    generation_type: str = "create",
    prompt: UserTurnInput | None = None,
    file_state: dict[str, str] | None = None,
    design_system: str | None = None,
) -> ExtractedParams:
    """Build an ExtractedParams for dispatch tests (validated inputs only)."""
    return ExtractedParams(
        stack=cast(Any, stack),
        input_mode=cast(Any, input_mode),
        should_generate_images=False,
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        replicate_api_key=None,
        openai_base_url=None,
        generation_type=cast(Any, generation_type),
        prompt=prompt
        or UserTurnInput(text="Build a hello world page", images=[], videos=[]),
        history=[],
        file_state=file_state,
        option_codes=[],
        design_system=design_system,
    )


class FakeRunner:
    """Structural Runner for dispatch tests; records probe/generate calls."""

    def __init__(
        self,
        *,
        available: bool = True,
        unavailable_reason: str | None = None,
        events: List[GenerationEvent] | None = None,
        supported_stacks: List[str] | None = None,
    ) -> None:
        self._available = available
        self._unavailable_reason = unavailable_reason
        self._events = list(events or [])
        self._supported_stacks = supported_stacks or [
            "html_tailwind",
            "html_css",
            "react_tailwind",
            "vue_tailwind",
        ]
        self.probe_calls = 0
        self.generate_calls = 0
        self.generation_inputs: List[GenerationInput] = []

    async def probe(self) -> ProbeResult:
        self.probe_calls += 1
        return ProbeResult(
            available=self._available, unavailable_reason=self._unavailable_reason
        )

    async def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        self.generate_calls += 1
        self.generation_inputs.append(generation_input)
        for event in self._events:
            yield event

    async def cancel(self) -> None:
        return None

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            runner_name="codex",
            available=self._available,
            supported_stacks=cast(Any, list(self._supported_stacks)),
            supported_input_modes=["image", "text"],
        )


class BombRunner:
    """A runner whose probe/generate must never run for a given request."""

    def __init__(self) -> None:
        self.probe_calls = 0
        self.generate_calls = 0

    async def probe(self) -> ProbeResult:
        self.probe_calls += 1
        raise AssertionError("probe() must not run for this request")

    def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        self.generate_calls += 1
        raise AssertionError("generate() must not run for this request")

    async def cancel(self) -> None:
        return None

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            runner_name="codex",
            available=True,
            supported_stacks=[
                "html_tailwind",
                "html_css",
                "react_tailwind",
                "vue_tailwind",
            ],
            supported_input_modes=["image", "text"],
        )


class YieldingProbeRunner(FakeRunner):
    """A runner whose probe yields to the event loop, exposing cache races.

    Without a per-runner lock, two concurrent ``_probe_runner_once`` calls both
    observe a cache miss before either caches, so the probe runs twice. The
    ``await asyncio.sleep(0)`` makes that interleaving deterministic.
    """

    async def probe(self) -> ProbeResult:
        self.probe_calls += 1
        await asyncio.sleep(0)
        return ProbeResult(
            available=self._available, unavailable_reason=self._unavailable_reason
        )


@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every test from the process-global registry state."""
    monkeypatch.setattr(registry, "_registry", {})
    yield


@pytest.fixture
def codex_configured(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Simulate GENERATION_RUNNER=codex for the routes.generate_code module."""
    monkeypatch.setattr(gen_code, "GENERATION_RUNNER", "codex")
    yield


def _make_context(
    extracted: ExtractedParams, sent_messages: List[tuple[str, str | None, int]]
) -> PipelineContext:
    """Build a PipelineContext whose send_message records (type, value, index)."""
    context = PipelineContext(websocket=MagicMock())

    async def send_message(
        msg_type: str,
        value: str | None,
        variant_index: int,
        data: Any = None,
        eventId: str | None = None,
    ) -> None:
        sent_messages.append((msg_type, value, variant_index))

    context.ws_comm = cast(
        Any,
        SimpleNamespace(send_message=send_message, throw_error=AsyncMock()),
    )
    context.extracted_params = extracted
    return context


# --- Registration / selection -------------------------------------------------


def test_selection_declines_without_probe_when_disabled_or_unknown(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    """Unset/blank/unknown GENERATION_RUNNER selects nothing and probes nothing."""
    bomb = BombRunner()
    registry.register("codex", bomb)

    for value in (None, "", "   ", "codex_cli", "does-not-exist"):
        monkeypatch.setattr(gen_code, "GENERATION_RUNNER", value)
        assert _select_local_runner() is None
        assert _local_dispatch_decision(_extracted()).runner is None

    # Nothing probed or generated, and the registry was never mutated.
    assert bomb.probe_calls == 0
    assert bomb.generate_calls == 0
    assert registry.is_registered("codex") is True
    assert registry.names() == ["codex"]


def test_registers_codex_runner_only_when_configured_exactly(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    """Only GENERATION_RUNNER == 'codex' registers the real Codex runner."""
    assert registry.is_registered("codex") is False

    monkeypatch.setattr(gen_code, "GENERATION_RUNNER", "codex")
    first = _select_local_runner()
    assert first is not None
    assert registry.is_registered("codex") is True
    assert registry.names() == ["codex"]

    # Idempotent: repeated selection must not re-register / duplicate.
    second = _select_local_runner()
    assert second is first
    assert registry.names() == ["codex"]

    # A different configured name never registers the real runner.
    monkeypatch.setattr(gen_code, "GENERATION_RUNNER", "codex_cli")
    assert _select_local_runner() is None
    assert registry.names() == ["codex"]


def test_eligibility_requires_text_input_and_runner_stack(
    empty_registry, codex_configured
) -> None:
    """v0.1 eligibility: text input only, stack declared by the runner."""
    runner = FakeRunner()
    registry.register("codex", runner)

    assert (
        _local_dispatch_decision(
            _extracted(stack="html_tailwind", input_mode="text")
        ).eligible
        is True
    )
    assert (
        _local_dispatch_decision(
            _extracted(stack="react_tailwind", input_mode="text")
        ).eligible
        is True
    )
    # Image/video input is out of scope for v0.1.
    assert (
        _local_dispatch_decision(
            _extracted(stack="html_tailwind", input_mode="image")
        ).eligible
        is False
    )
    assert (
        _local_dispatch_decision(
            _extracted(stack="html_tailwind", input_mode="video")
        ).eligible
        is False
    )
    # Stacks outside RunnerStack (bootstrap / ionic_tailwind) are out of scope.
    assert (
        _local_dispatch_decision(
            _extracted(stack="bootstrap", input_mode="text")
        ).eligible
        is False
    )
    assert (
        _local_dispatch_decision(
            _extracted(stack="ionic_tailwind", input_mode="text")
        ).eligible
        is False
    )
    # A runner that does not declare the stack is ineligible too.
    limited = FakeRunner(supported_stacks=["html_tailwind"])
    registry.register("limited", limited)
    assert (
        _local_dispatch_decision(
            _extracted(stack="react_tailwind", input_mode="text")
        ).runner
        is runner
    )

    assert runner.probe_calls == 0
    assert runner.generate_calls == 0


# --- StatusBroadcastMiddleware: single-variant announcement -------------------


@pytest.mark.asyncio
async def test_status_broadcast_announces_single_variant_for_local_runner(
    empty_registry, codex_configured
) -> None:
    """A selected local runner announces exactly one variant and one status."""
    sent_messages: List[tuple[str, str | None, int]] = []
    runner = FakeRunner(events=[])
    registry.register("codex", runner)
    context = _make_context(_extracted(), sent_messages)

    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await StatusBroadcastMiddleware().process(context, next_func)

    assert sent_messages == [
        ("variantCount", "1", 0),
        ("status", "Generating code...", 0),
    ]
    assert next_called is True
    # Capability reads must not probe the runner.
    assert runner.probe_calls == 0


# --- CodeGenerationMiddleware: local dispatch ---------------------------------


@pytest.mark.asyncio
async def test_eligible_local_dispatch_probes_once_and_bypasses_legacy(
    empty_registry, codex_configured, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eligible local request: probe once, stream to variant 0, no legacy path."""
    code = "<html><body>Hello local</body></html>"
    runner = FakeRunner(
        events=[
            CompleteEvent(
                result=GenerationResult(
                    provider="codex",
                    status="success",
                    primary_path="index.html",
                    text_files={"index.html": code},
                )
            )
        ]
    )
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    # The legacy path must never be reached: any use of model selection or the
    # AgenticGenerationStage is a test failure.
    class BombModelSelector:
        def __init__(self, throw_error: Any) -> None:
            raise AssertionError("legacy model selection must be bypassed")

    class BombAgenticStage:
        def __init__(self, **kwargs: Any) -> None:
            raise AssertionError("AgenticGenerationStage must be bypassed")

    monkeypatch.setattr(gen_code, "ModelSelectionStage", BombModelSelector)
    monkeypatch.setattr(gen_code, "AgenticGenerationStage", BombAgenticStage)

    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await CodeGenerationMiddleware().process(context, next_func)

    assert runner.probe_calls == 1
    assert runner.generate_calls == 1
    assert sent_messages == [("setCode", code, 0), ("variantComplete", None, 0)]
    assert next_called is False
    context.ws_comm.throw_error.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_local_dispatch_probes_once_across_two_eligible_calls(
    empty_registry, codex_configured
) -> None:
    """Two eligible calls on the same runner probe exactly once (cached)."""
    runner = FakeRunner(
        events=[
            CompleteEvent(
                result=GenerationResult(
                    provider="codex",
                    status="success",
                    primary_path="index.html",
                    text_files={"index.html": "<html>ok</html>"},
                )
            )
        ]
    )
    registry.register("codex", runner)

    middleware = CodeGenerationMiddleware()
    for _ in range(2):
        sent_messages: List[tuple[str, str | None, int]] = []
        context = _make_context(_extracted(), sent_messages)
        next_called = False

        async def next_func() -> None:
            nonlocal next_called
            next_called = True

        await middleware.process(context, next_func)
        assert next_called is False

    assert runner.probe_calls == 1
    assert runner.generate_calls == 2


@pytest.mark.asyncio
async def test_probe_runs_once_under_concurrent_requests(
    empty_registry, codex_configured
) -> None:
    """Concurrent eligible requests share a single probe (per-runner lock)."""
    runner = YieldingProbeRunner()
    registry.register("codex", runner)

    results = await asyncio.gather(
        _probe_runner_once(runner),
        _probe_runner_once(runner),
    )

    assert [r.available for r in results] == [True, True]
    assert runner.probe_calls == 1


@pytest.mark.asyncio
async def test_unavailable_probe_fails_closed_with_single_variant_error(
    empty_registry, codex_configured
) -> None:
    """An unavailable probe emits one safe variantError; no generate/fallback."""
    runner = FakeRunner(available=False, unavailable_reason="codex binary not found")
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await CodeGenerationMiddleware().process(context, next_func)

    assert runner.probe_calls == 1
    assert runner.generate_calls == 0
    # Exactly one variantError carrying the bounded reason; no variantComplete.
    assert [m[0] for m in sent_messages] == ["variantError"]
    assert sent_messages[0][1] == "codex binary not found"
    assert next_called is False
    context.ws_comm.throw_error.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_probe_without_reason_uses_safe_fallback(
    empty_registry, codex_configured
) -> None:
    """A probe failure with no reason yields the fixed safe fallback message."""
    runner = FakeRunner(available=False, unavailable_reason=None)
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    async def next_func() -> None:
        raise AssertionError("must not fall through to the cloud path")

    await CodeGenerationMiddleware().process(context, next_func)

    assert [m[0] for m in sent_messages] == ["variantError"]
    value = sent_messages[0][1]
    assert isinstance(value, str) and value.strip() != ""
    assert runner.generate_calls == 0


@pytest.mark.asyncio
async def test_unsupported_input_mode_fails_closed_without_probe(
    empty_registry, codex_configured
) -> None:
    """Image input on a configured local runner fails closed, no probe/generate."""
    bomb = BombRunner()
    registry.register("codex", bomb)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(input_mode="image"), sent_messages)

    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await CodeGenerationMiddleware().process(context, next_func)

    assert [m[0] for m in sent_messages] == ["variantError"]
    assert sent_messages[0][1] == UNSUPPORTED_MESSAGE
    assert bomb.probe_calls == 0
    assert bomb.generate_calls == 0
    assert next_called is False
    context.ws_comm.throw_error.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_unsupported_stack_fails_closed_without_probe(
    empty_registry, codex_configured
) -> None:
    """A stack outside RunnerStack fails closed, no probe/generate/fallback."""
    bomb = BombRunner()
    registry.register("codex", bomb)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(stack="bootstrap"), sent_messages)

    async def next_func() -> None:
        raise AssertionError("must not fall through to the cloud path")

    await CodeGenerationMiddleware().process(context, next_func)

    assert [m[0] for m in sent_messages] == ["variantError"]
    assert sent_messages[0][1] == UNSUPPORTED_MESSAGE
    assert bomb.probe_calls == 0
    assert bomb.generate_calls == 0


# --- CompleteEvent text_files -> setCode ordering ------------------------------


@pytest.mark.asyncio
async def test_complete_text_files_yield_sorted_setcode_before_variant_complete(
    empty_registry, codex_configured
) -> None:
    """CompleteEvent text_files become setCode messages in sorted path order."""
    runner = FakeRunner(
        events=[
            CompleteEvent(
                result=GenerationResult(
                    provider="codex",
                    status="success",
                    primary_path="b.html",
                    text_files={"b.html": "<html>B</html>", "a.html": "<html>A</html>"},
                )
            )
        ]
    )
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    async def next_func() -> None:
        raise AssertionError("must not pipeline into cloud/postprocessing")

    await CodeGenerationMiddleware().process(context, next_func)

    # Sorted by path: a.html then b.html, both before variantComplete.
    assert [m[0] for m in sent_messages] == [
        "setCode",
        "setCode",
        "variantComplete",
    ]
    assert [m[1] for m in sent_messages if m[0] == "setCode"] == [
        "<html>A</html>",
        "<html>B</html>",
    ]
    # No filename is ever sent to the client; only the code string payload.
    assert all("a.html" not in str(m[1]) for m in sent_messages)
    assert all("b.html" not in str(m[1]) for m in sent_messages)


@pytest.mark.asyncio
async def test_no_duplicate_setcode_when_runner_emits_both(
    empty_registry, codex_configured
) -> None:
    """A runner emitting SetCodeEvent + CompleteEvent sends the code once."""
    code = "<html>once</html>"
    runner = FakeRunner(
        events=[
            SetCodeEvent(file_name="index.html", code=code),
            CompleteEvent(
                result=GenerationResult(
                    provider="codex",
                    status="success",
                    primary_path="index.html",
                    text_files={"index.html": code},
                )
            ),
        ]
    )
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    async def next_func() -> None:
        raise AssertionError("must not pipeline into cloud/postprocessing")

    await CodeGenerationMiddleware().process(context, next_func)

    assert [m[0] for m in sent_messages] == ["setCode", "variantComplete"]
    assert sent_messages[0][1] == code


@pytest.mark.asyncio
async def test_error_event_raw_text_never_reaches_socket(
    empty_registry, codex_configured
) -> None:
    """Runner ErrorEvent raw text is replaced by a fixed client-safe message."""
    hostile = "codex subprocess failed in /tmp/job with model gpt-5 token secret-token"
    runner = FakeRunner(
        events=[ErrorEvent(message=hostile)],
        available=True,
    )
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    async def next_func() -> None:
        raise AssertionError("must not pipeline into cloud/postprocessing")

    await CodeGenerationMiddleware().process(context, next_func)

    assert [m[0] for m in sent_messages] == ["variantError"]
    value = sent_messages[0][1]
    assert isinstance(value, str) and value.strip() != ""
    assert value != hostile
    for token in ("/tmp/job", "codex", "gpt-5", "secret-token"):
        assert token not in value
    assert sum(1 for m in sent_messages if m[0] == "variantComplete") == 0


# --- GenerationInput construction ---------------------------------------------


@pytest.mark.asyncio
async def test_generation_input_is_server_owned_and_uses_full_text(
    empty_registry, codex_configured
) -> None:
    """GenerationInput: server request_id, full_text prompt, no image/prior files."""
    runner = FakeRunner(
        events=[
            CompleteEvent(
                result=GenerationResult(
                    provider="codex",
                    status="success",
                    primary_path="index.html",
                    text_files={"index.html": "<html>ok</html>"},
                )
            )
        ]
    )
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(
        _extracted(
            prompt={
                "text": "display text",
                "full_text": "full instruction text",
                "images": [],
                "videos": [],
            },
            design_system="Warm, friendly brand",
        ),
        sent_messages,
    )

    async def next_func() -> None:
        raise AssertionError("must not pipeline into cloud/postprocessing")

    await CodeGenerationMiddleware().process(context, next_func)

    assert len(runner.generation_inputs) == 1
    gen_input = runner.generation_inputs[0]
    assert gen_input.request_id.startswith("req_")
    assert gen_input.request_id != ""
    assert gen_input.prompt == "full instruction text"
    assert gen_input.stack == "html_tailwind"
    assert gen_input.generation_type == "create"
    assert gen_input.input_image is None
    assert gen_input.prior_files == []
    assert gen_input.design_system == "Warm, friendly brand"
    # The server-owned internals must never reach the client socket.
    assert all("req_" not in str(m[1]) for m in sent_messages)


@pytest.mark.asyncio
async def test_generation_input_falls_back_to_text_and_omits_client_paths(
    empty_registry, codex_configured
) -> None:
    """Update flow: no full_text -> text prompt; file_state path never used."""
    runner = FakeRunner(
        events=[
            CompleteEvent(
                result=GenerationResult(
                    provider="codex",
                    status="success",
                    primary_path="index.html",
                    text_files={"index.html": "<html>ok</html>"},
                )
            )
        ]
    )
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(
        _extracted(
            generation_type="update",
            prompt={"text": "Update the page", "images": [], "videos": []},
            file_state={
                "path": "/home/user/evil/index.html",
                "content": "<html>existing</html>",
            },
        ),
        sent_messages,
    )

    async def next_func() -> None:
        raise AssertionError("must not pipeline into cloud/postprocessing")

    await CodeGenerationMiddleware().process(context, next_func)

    assert len(runner.generation_inputs) == 1
    gen_input = runner.generation_inputs[0]
    assert gen_input.prompt == "Update the page"
    assert gen_input.generation_type == "update"
    # No caller/client filesystem paths are introduced in this slice.
    assert gen_input.prior_files == []
    assert gen_input.input_image is None
    assert "/home/user" not in str(gen_input)


# --- Legacy path preservation --------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_local_runner_preserves_legacy_cloud_path(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    """With no local runner configured, the legacy cloud path runs untouched."""
    monkeypatch.setattr(gen_code, "GENERATION_RUNNER", None)
    bomb = BombRunner()
    registry.register("codex", bomb)

    select_calls = 0

    class FakeModelSelector:
        def __init__(self, throw_error: Any) -> None:
            pass

        async def select_models(self, **kwargs: Any) -> List[Any]:
            nonlocal select_calls
            select_calls += 1
            return [MagicMock()]

    class FakeAgenticStage:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def process_variants(
            self, variant_models: List[Any], prompt_messages: List[Any]
        ) -> dict[int, str]:
            return {0: "legacy-code"}

    monkeypatch.setattr(gen_code, "ModelSelectionStage", FakeModelSelector)
    monkeypatch.setattr(gen_code, "AgenticGenerationStage", FakeAgenticStage)

    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)
    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await CodeGenerationMiddleware().process(context, next_func)

    # Legacy selection ran and the pipeline continued; local runner untouched.
    assert select_calls == 1
    assert next_called is True
    assert bomb.probe_calls == 0
    assert bomb.generate_calls == 0
    # No local dispatch messages were emitted by the middleware itself.
    assert all(m[0] not in ("setCode", "variantError") for m in sent_messages)


# --- PromptCreationMiddleware: local bypass --------------------------------


@pytest.mark.asyncio
async def test_prompt_creation_skipped_for_local_runner(
    empty_registry, codex_configured, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured local runner bypasses legacy prompt construction."""
    runner = FakeRunner(events=[])
    registry.register("codex", runner)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    class BombPromptCreationStage:
        def __init__(self, throw_error: Any) -> None:
            raise AssertionError("legacy prompt construction must be skipped")

    monkeypatch.setattr(gen_code, "PromptCreationStage", BombPromptCreationStage)

    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await PromptCreationMiddleware().process(context, next_func)

    assert next_called is True
    assert context.prompt_messages == []
    # Only a side-effect-free capability read may have happened; no probe.
    assert runner.probe_calls == 0


@pytest.mark.asyncio
async def test_prompt_creation_runs_for_legacy_path(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    """Without a local runner, prompt construction still runs untouched."""
    monkeypatch.setattr(gen_code, "GENERATION_RUNNER", None)
    sent_messages: List[tuple[str, str | None, int]] = []
    context = _make_context(_extracted(), sent_messages)

    built = False

    class FakePromptCreationStage:
        def __init__(self, throw_error: Any) -> None:
            pass

        async def build_prompt_messages(
            self, extracted_params: ExtractedParams
        ) -> List[Any]:
            nonlocal built
            built = True
            return [MagicMock()]

    monkeypatch.setattr(gen_code, "PromptCreationStage", FakePromptCreationStage)

    next_called = False

    async def next_func() -> None:
        nonlocal next_called
        next_called = True

    await PromptCreationMiddleware().process(context, next_func)

    assert built is True
    assert len(context.prompt_messages) == 1
    assert next_called is True


# --- _expand_complete_text_files unit behavior --------------------------------


@pytest.mark.asyncio
async def test_expand_wrapper_emits_sorted_setcode_events() -> None:
    """The wrapper yields SetCodeEvents (sorted) before the CompleteEvent."""

    async def source() -> AsyncIterator[GenerationEvent]:
        yield CompleteEvent(
            result=GenerationResult(
                provider="codex",
                status="success",
                primary_path="z.html",
                text_files={"z.html": "Z", "a.html": "A", "m.html": "M"},
            )
        )

    collected: List[GenerationEvent] = []
    async for event in _expand_complete_text_files(source()):
        collected.append(event)

    assert [type(e).__name__ for e in collected] == [
        "SetCodeEvent",
        "SetCodeEvent",
        "SetCodeEvent",
        "CompleteEvent",
    ]
    set_codes = [cast(SetCodeEvent, e) for e in collected if isinstance(e, SetCodeEvent)]
    assert [e.file_name for e in set_codes] == ["a.html", "m.html", "z.html"]
    assert [e.code for e in set_codes] == ["A", "M", "Z"]

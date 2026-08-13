"""RED test: WebSocket adapter for the local-runner vertical slice.

This test specifies the adapter boundary the next slice must implement. The
production symbol under test — ``routes.local_runner_adapter`` and its
``LocalRunnerWebSocketAdapter`` class — does not exist yet, so this test
currently fails at import with ``ModuleNotFoundError`` (strict TDD RED: the
failure is the *absence of the adapter*, not a malformed test setup).

Contract being specified (translation of the normalized ``GenerationEvent``
stream from ``runners.models`` into the existing
``WebSocketCommunicator.send_message`` calls for a single variant):

    StatusEvent(message)  -> send_message("status",  message, variant_index)
    ThinkingEvent(text)   -> send_message("thinking", text,    variant_index)
    AssistantEvent(text)  -> send_message("assistant", text,   variant_index)
    SetCodeEvent(code)    -> send_message("setCode",  code,    variant_index)
    CompleteEvent(...)    -> send_message("variantComplete", ...)  (exactly once)

The adapter must terminate the stream with exactly one ``variantComplete`` for
the variant, and the ``setCode`` message must carry the code from the
``SetCodeEvent`` (not the ``CompleteEvent`` result).
"""

from typing import Any, AsyncIterator, List, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from routes.generate_code import WebSocketCommunicator
from routes.local_runner_adapter import LocalRunnerWebSocketAdapter  # noqa: F401  (RED: not implemented yet)
from runners.models import (
    AssistantEvent,
    CompleteEvent,
    ErrorEvent,
    GenerationEvent,
    GenerationInput,
    GenerationResult,
    SetCodeEvent,
    StatusEvent,
    ThinkingEvent,
)


class FakeRunner:
    """Minimal ``Runner`` that replays a fixed, ordered event stream.

    Only ``generate()`` matters here — the adapter consumes its event stream;
    the remaining ``Runner`` protocol methods are unimplemented because the
    adapter under test must not call them.
    """

    def __init__(self, events: List[GenerationEvent]) -> None:
        self._events: List[GenerationEvent] = list(events)
        self.generate_calls = 0

    async def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        self.generate_calls += 1
        for event in self._events:
            yield event

    async def probe(self) -> Any:
        raise NotImplementedError("adapter must not probe the runner")

    async def cancel(self) -> None:
        raise NotImplementedError("adapter must not cancel the runner")

    def capabilities(self) -> Any:
        raise NotImplementedError("adapter must not read capabilities")


@pytest.mark.asyncio
async def test_adapter_translates_runner_events_to_websocket_messages() -> None:
    """A local runner's event stream becomes variant-0 WebSocket messages.

    The adapter must forward status/thinking/assistant messages, emit ``setCode``
    with the exact code from the ``SetCodeEvent``, and end with exactly one
    ``variantComplete`` for variant 0 — all through the existing
    ``WebSocketCommunicator.send_message``.
    """
    code = "<html><body>Hello from local runner</body></html>"

    events: List[GenerationEvent] = [
        StatusEvent(message="Generating code locally..."),
        ThinkingEvent(text="Planning the page structure"),
        AssistantEvent(text="Here is the generated page"),
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
    runner = FakeRunner(events)

    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    communicator = WebSocketCommunicator(cast(Any, websocket))

    adapter = LocalRunnerWebSocketAdapter(
        send_message=communicator.send_message,
        variant_index=0,
    )

    await adapter.stream(
        runner.generate(
            GenerationInput(
                request_id="req-1",
                prompt="Build a hello world page",
                stack="html_tailwind",
                generation_type="create",
            )
        )
    )

    # The adapter must have consumed the fake runner's stream.
    assert runner.generate_calls == 1

    payloads = [call.args[0] for call in websocket.send_json.await_args_list]

    # Message order and types, all for variant 0.
    assert [p["type"] for p in payloads] == [
        "status",
        "thinking",
        "assistant",
        "setCode",
        "variantComplete",
    ]
    assert all(p["variantIndex"] == 0 for p in payloads)

    status, thinking, assistant, set_code, variant_complete = payloads

    assert status["value"] == "Generating code locally..."
    assert thinking["value"] == "Planning the page structure"
    assert assistant["value"] == "Here is the generated page"

    # setCode must carry the exact code emitted by the SetCodeEvent.
    assert set_code["value"] == code

    # Exactly one variantComplete closes the stream (already enforced by the
    # type-order assertion above; asserted again explicitly for clarity).
    variant_complete_count = sum(
        1 for p in payloads if p["type"] == "variantComplete"
    )
    assert variant_complete_count == 1


@pytest.mark.asyncio
async def test_adapter_emits_variant_error_on_error_event() -> None:
    """An ``ErrorEvent`` becomes exactly one ``variantError`` and closes the stream."""

    events: List[GenerationEvent] = [
        StatusEvent(message="Starting local runner..."),
        ErrorEvent(message="Local runner failed to start"),
    ]
    runner = FakeRunner(events)

    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    communicator = WebSocketCommunicator(cast(Any, websocket))

    adapter = LocalRunnerWebSocketAdapter(
        send_message=communicator.send_message,
        variant_index=2,
    )

    await adapter.stream(
        runner.generate(
            GenerationInput(
                request_id="req-2",
                prompt="Build a page",
                stack="html_tailwind",
                generation_type="create",
            )
        )
    )

    payloads = [call.args[0] for call in websocket.send_json.await_args_list]

    # Status forwarded, then exactly one variantError carrying a fixed, generic
    # client-safe error (never the raw ErrorEvent message).
    assert [p["type"] for p in payloads] == ["status", "variantError"]
    assert all(p["variantIndex"] == 2 for p in payloads)
    assert isinstance(payloads[1]["value"], str)
    assert payloads[1]["value"].strip() != ""
    assert payloads[1]["value"] != "Local runner failed to start"
    assert sum(1 for p in payloads if p["type"] == "variantComplete") == 0


@pytest.mark.asyncio
async def test_adapter_fails_closed_on_nonterminal_stream_exhaustion() -> None:
    """A stream that ends without a terminal event fails closed with one variantError."""

    events: List[GenerationEvent] = [
        StatusEvent(message="Starting..."),
        ThinkingEvent(text="Planning..."),
    ]
    runner = FakeRunner(events)

    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    communicator = WebSocketCommunicator(cast(Any, websocket))

    adapter = LocalRunnerWebSocketAdapter(
        send_message=communicator.send_message,
        variant_index=1,
    )

    await adapter.stream(
        runner.generate(
            GenerationInput(
                request_id="req-3",
                prompt="Build a page",
                stack="html_tailwind",
                generation_type="create",
            )
        )
    )

    payloads = [call.args[0] for call in websocket.send_json.await_args_list]

    # All non-terminal events forwarded, then exactly one generic variantError.
    assert [p["type"] for p in payloads] == ["status", "thinking", "variantError"]
    assert all(p["variantIndex"] == 1 for p in payloads)
    assert sum(1 for p in payloads if p["type"] == "variantError") == 1
    assert sum(1 for p in payloads if p["type"] == "variantComplete") == 0
    # Generic message: non-empty, with no provider/model/path/credential detail.
    assert isinstance(payloads[-1]["value"], str)
    assert payloads[-1]["value"].strip() != ""


@pytest.mark.asyncio
async def test_adapter_ignores_events_after_terminal_event() -> None:
    """Events yielded after a terminal event are ignored (no trailing messages)."""

    code = "<html>done</html>"
    events: List[GenerationEvent] = [
        StatusEvent(message="Starting..."),
        CompleteEvent(
            result=GenerationResult(
                provider="codex",
                status="success",
                primary_path="index.html",
                text_files={"index.html": code},
            )
        ),
        ThinkingEvent(text="should be ignored"),
        SetCodeEvent(file_name="index.html", code=code),
    ]
    runner = FakeRunner(events)

    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    communicator = WebSocketCommunicator(cast(Any, websocket))

    adapter = LocalRunnerWebSocketAdapter(
        send_message=communicator.send_message,
        variant_index=0,
    )

    await adapter.stream(
        runner.generate(
            GenerationInput(
                request_id="req-4",
                prompt="Build a page",
                stack="html_tailwind",
                generation_type="create",
            )
        )
    )

    payloads = [call.args[0] for call in websocket.send_json.await_args_list]

    # Only messages up to and including the terminal variantComplete are sent.
    assert [p["type"] for p in payloads] == ["status", "variantComplete"]
    assert sum(1 for p in payloads if p["type"] == "variantComplete") == 1


@pytest.mark.asyncio
async def test_error_event_never_leaks_runner_details_to_client() -> None:
    """Raw ErrorEvent details (provider/model/path/credential) never reach the client.

    The adapter must replace the runner's error message with a fixed, generic,
    client-safe error regardless of how much sensitive detail the message carries.
    """
    hostile_message = (
        "codex subprocess failed in /tmp/job with model gpt-5 "
        "using token secret-token"
    )
    events: List[GenerationEvent] = [
        StatusEvent(message="Starting..."),
        ErrorEvent(message=hostile_message),
    ]
    runner = FakeRunner(events)

    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    communicator = WebSocketCommunicator(cast(Any, websocket))

    adapter = LocalRunnerWebSocketAdapter(
        send_message=communicator.send_message,
        variant_index=0,
    )

    await adapter.stream(
        runner.generate(
            GenerationInput(
                request_id="req-leak",
                prompt="Build a page",
                stack="html_tailwind",
                generation_type="create",
            )
        )
    )

    payloads = [call.args[0] for call in websocket.send_json.await_args_list]

    assert [p["type"] for p in payloads] == ["status", "variantError"]
    error_value = payloads[-1]["value"]
    assert isinstance(error_value, str)
    assert error_value.strip() != ""
    # The generic error must not contain any of the raw sensitive tokens.
    for token in ("/tmp/job", "codex", "gpt-5", "secret-token", hostile_message):
        assert token not in error_value
    assert sum(1 for p in payloads if p["type"] == "variantComplete") == 0


@pytest.mark.asyncio
async def test_adapter_fails_closed_when_iterator_raises() -> None:
    """An iterator that raises before a terminal event fails closed.

    ``stream()`` must emit exactly one generic ``variantError`` and return
    normally, never propagating the exception and never emitting ``variantComplete``.
    """

    async def raising_events() -> AsyncIterator[GenerationEvent]:
        yield StatusEvent(message="Starting...")
        raise RuntimeError("codex crashed in /tmp/job")

    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    communicator = WebSocketCommunicator(cast(Any, websocket))

    adapter = LocalRunnerWebSocketAdapter(
        send_message=communicator.send_message,
        variant_index=1,
    )

    # Must not raise; the exception is contained and reported generically.
    await adapter.stream(raising_events())

    payloads = [call.args[0] for call in websocket.send_json.await_args_list]

    assert [p["type"] for p in payloads] == ["status", "variantError"]
    assert all(p["variantIndex"] == 1 for p in payloads)
    assert sum(1 for p in payloads if p["type"] == "variantError") == 1
    assert sum(1 for p in payloads if p["type"] == "variantComplete") == 0

    error_value = payloads[-1]["value"]
    assert isinstance(error_value, str)
    assert error_value.strip() != ""
    # The generic error must not leak the exception's sensitive detail.
    for token in ("codex", "/tmp/job", "RuntimeError"):
        assert token not in error_value

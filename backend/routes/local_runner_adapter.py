"""WebSocket boundary for the local-runner vertical slice.

Translates the normalized ``GenerationEvent`` stream emitted by a local
coding-agent runner (see ``runners.models``) into the existing
``WebSocketCommunicator.send_message`` calls used by the rest of the app, for a
single variant.

This adapter is purely a translator: it never launches a subprocess, probes or
cancels a runner, reads capabilities, or inspects any credentials or
provider/process details. It must not leak provider names, model identifiers,
or raw artifact paths to the client.
"""

from __future__ import annotations

from typing import AsyncIterator, Awaitable, Callable, Literal

from runners.models import (
    AssistantEvent,
    CompleteEvent,
    ErrorEvent,
    GenerationEvent,
    SetCodeEvent,
    StatusEvent,
    ThinkingEvent,
)

# The subset of WebSocket message types this adapter emits. Deliberately
# narrower than ``routes.generate_code.MessageType`` so the adapter stays a
# thin, self-contained boundary; it is contravariantly safe to accept the real
# ``WebSocketCommunicator.send_message``.
AdapterMessageType = Literal[
    "status",
    "thinking",
    "assistant",
    "setCode",
    "variantComplete",
    "variantError",
]

SendMessage = Callable[[AdapterMessageType, str | None, int], Awaitable[None]]

# Safe, user-facing message for a stream that ended without a terminal event.
# Must not include provider names, models, artifact paths, or process details.
_GENERIC_TERMINATION_ERROR = "Local runner stream ended unexpectedly."


class LocalRunnerWebSocketAdapter:
    """Forward a runner's event stream to the WebSocket for one variant.

    A well-formed stream ends with exactly one ``CompleteEvent`` or
    ``ErrorEvent``; the adapter emits the matching terminal message exactly once
    and then returns, ignoring any events that follow. A stream that exhausts
    without a terminal event fails closed with a single generic
    ``variantError``.
    """

    def __init__(self, send_message: SendMessage, variant_index: int) -> None:
        self._send_message = send_message
        self._variant_index = variant_index

    async def stream(self, events: AsyncIterator[GenerationEvent]) -> None:
        async for event in events:
            if isinstance(event, StatusEvent):
                await self._send_message(
                    "status", event.message, self._variant_index
                )
            elif isinstance(event, ThinkingEvent):
                await self._send_message("thinking", event.text, self._variant_index)
            elif isinstance(event, AssistantEvent):
                await self._send_message(
                    "assistant", event.text, self._variant_index
                )
            elif isinstance(event, SetCodeEvent):
                await self._send_message("setCode", event.code, self._variant_index)
            elif isinstance(event, CompleteEvent):
                await self._send_message(
                    "variantComplete", None, self._variant_index
                )
                return
            elif isinstance(event, ErrorEvent):
                await self._send_message(
                    "variantError", event.message, self._variant_index
                )
                return
        # Non-terminal exhaustion: fail closed with exactly one generic error.
        await self._send_message(
            "variantError", _GENERIC_TERMINATION_ERROR, self._variant_index
        )

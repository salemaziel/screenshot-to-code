"""The ``Runner`` contract every local coding-agent runner must satisfy."""

from typing import AsyncIterator, Protocol

from runners.models import (
    GenerationEvent,
    GenerationInput,
    ProbeResult,
    RunnerCapabilities,
)


class Runner(Protocol):
    """A local coding-agent runner (e.g. a future Codex subprocess runner).

    Cancellation semantics: ``cancel()`` is best-effort and applies to the most
    recently started ``generate()``; it is a no-op when no generation is in
    flight and must never raise just because there is nothing to cancel.

    Result delivery: the stream yielded by ``generate()`` terminates with
    exactly one ``CompleteEvent`` (carrying the ``GenerationResult``) on
    success, or one ``ErrorEvent`` before the stream ends on failure. No
    events are yielded after the terminal event.

    Probing: ``probe()`` is the only operation allowed to launch or check a
    vendor CLI. ``capabilities()`` must be cheap and side-effect free so the
    registry can enumerate it without ever executing a binary. Consequently,
    the routine ``GET /api/capabilities`` endpoint (and any capability listing)
    must never invoke a CLI or trigger an OAuth flow; a selected runner can be
    probed later through an explicit, separate mechanism — never automatically
    as part of a capability read.
    """

    async def probe(self) -> ProbeResult:
        """Actually check whether this runner can run here (may be expensive)."""
        ...

    def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        """Stream normalized generation events for ``generation_input``."""
        ...

    async def cancel(self) -> None:
        """Best-effort cancellation of the in-flight generation (no-op if none)."""
        ...

    def capabilities(self) -> RunnerCapabilities:
        """Declared capabilities; must not launch or probe a vendor CLI."""
        ...

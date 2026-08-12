"""Local runner backed by the Codex CLI.

This is a deliberately narrow v0.1 slice: only the side-effect-free declared
``capabilities()`` and a safe, bounded ``probe()`` are implemented.
``generate()`` and ``cancel()`` are protocol-compatible stubs for later work.

No credentials or user-controlled executables/arguments are held here; the
probe uses a fixed literal argv and never launches a shell.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from custom_types import InputMode
from runners.models import (
    GenerationEvent,
    GenerationInput,
    ProbeResult,
    RunnerCapabilities,
    RunnerStack,
)

# Fixed probe argv — never user-controlled, never a shell.
_PROBE_ARGV = ("codex", "--version")

# Short, explicit bound on the probe subprocess.
_PROBE_TIMEOUT_SECONDS = 5.0

# Short, fixed bound on each cleanup wait step; keeps stop+reap bounded even if
# the child ignores SIGTERM (and SIGKILL).
_PROBE_CLEANUP_TIMEOUT_SECONDS = 1.0

# Short, generic, safe reasons — must never leak raw command output.
_MISSING_REASON = "codex binary not found"
_TIMEOUT_REASON = "codex probe timed out"
_NONZERO_REASON = "codex binary exited with an error"

# Declared capability shape: exactly the four local stacks and image+text input.
_SUPPORTED_STACKS: list[RunnerStack] = [
    "html_tailwind",
    "html_css",
    "react_tailwind",
    "vue_tailwind",
]
_SUPPORTED_INPUT_MODES: list[InputMode] = ["image", "text"]


class CodexCliRunner:
    """Skeleton local runner for the Codex CLI."""

    def capabilities(self) -> RunnerCapabilities:
        """Declared capabilities; never launches or checks the vendor CLI."""
        return RunnerCapabilities(
            runner_name="codex",
            available=False,
            supported_stacks=list(_SUPPORTED_STACKS),
            supported_input_modes=list(_SUPPORTED_INPUT_MODES),
        )

    async def probe(self) -> ProbeResult:
        """Check whether the Codex CLI is available (the sole check operation)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *_PROBE_ARGV,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return ProbeResult(available=False, unavailable_reason=_MISSING_REASON)

        try:
            await asyncio.wait_for(proc.communicate(), timeout=_PROBE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # Stop and reap the child before returning; never leak a live process.
            await self._terminate_and_reap(proc)
            return ProbeResult(available=False, unavailable_reason=_TIMEOUT_REASON)

        if proc.returncode == 0:
            return ProbeResult(available=True)
        return ProbeResult(available=False, unavailable_reason=_NONZERO_REASON)

    async def _terminate_and_reap(self, proc: asyncio.subprocess.Process) -> None:
        """Stop and reap an unresponsive probe child within a fixed bound.

        Escalates exactly ``terminate()`` -> bounded ``wait()`` -> ``kill()`` ->
        bounded ``wait()`` so a child that ignores SIGTERM can never hang
        cleanup. Only ``ProcessLookupError`` (child already gone) is swallowed;
        any other exception propagates.
        """
        try:
            proc.terminate()
        except ProcessLookupError:
            return  # already reaped elsewhere; nothing left to do

        try:
            await asyncio.wait_for(
                proc.wait(), timeout=_PROBE_CLEANUP_TIMEOUT_SECONDS
            )
            return
        except ProcessLookupError:
            return
        except asyncio.TimeoutError:
            pass  # SIGTERM ignored; escalate to SIGKILL

        try:
            proc.kill()
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(
                proc.wait(), timeout=_PROBE_CLEANUP_TIMEOUT_SECONDS
            )
        except (ProcessLookupError, asyncio.TimeoutError):
            # Already reaped, or the child ignores SIGKILL too — best effort done.
            pass

    def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        """Not implemented in this slice."""
        raise NotImplementedError("CodexCliRunner.generate is not implemented yet")

    async def cancel(self) -> None:
        """Best-effort no-op stub (no generation is implemented yet)."""
        return None

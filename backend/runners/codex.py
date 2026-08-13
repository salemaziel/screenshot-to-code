"""Local runner backed by the Codex CLI.

This is a deliberately narrow v0.1 slice: only the side-effect-free declared
``capabilities()`` and a safe, bounded ``probe()`` are implemented.
``generate()`` and ``cancel()`` are protocol-compatible stubs for later work.

No credentials or user-controlled executables/arguments are held here; the
probe uses a fixed literal argv and never launches a shell.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional

from custom_types import InputMode
from runners.models import (
    CompleteEvent,
    ErrorEvent,
    GenerationEvent,
    GenerationInput,
    GenerationResult,
    GenerationStatus,
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

# Bounds for generate() subprocess lifecycle.
_GENERATE_TIMEOUT_SECONDS = 300.0
_GENERATE_CLEANUP_TIMEOUT_SECONDS = 2.0

# Short, generic, safe reasons — must never leak raw command output.
_MISSING_REASON = "codex binary not found"
_TIMEOUT_REASON = "codex probe timed out"
_NONZERO_REASON = "codex binary exited with an error"
_GENERATION_ERROR_REASON = "codex generation failed"

# Declared capability shape: exactly the four local stacks and image+text input.
_SUPPORTED_STACKS: list[RunnerStack] = [
    "html_tailwind",
    "html_css",
    "react_tailwind",
    "vue_tailwind",
]
_SUPPORTED_INPUT_MODES: list[InputMode] = ["image", "text"]

# Provider key environment variable prefixes to strip from the child env.
_PROVIDER_KEY_PREFIXES = (
    "OPENAI_",
    "ANTHROPIC_",
    "GEMINI_",
    "GROQ_",
    "DEEPSEEK_",
    "REPLICATE_",
    "COHERE_",
    "MISTRAL_",
    "AZURE_",
    "AWS_",
    "GOOGLE_",
)


def _build_child_env() -> dict[str, str]:
    """Return a minimal, explicit environment for the codex child process.

    Preserves only PATH, HOME, and optionally CODEX_HOME; strips all known
    provider/API-key variables.
    """
    env: dict[str, str] = {}
    for key in ("PATH", "HOME"):
        if key in os.environ:
            env[key] = os.environ[key]
    if "CODEX_HOME" in os.environ:
        env["CODEX_HOME"] = os.environ["CODEX_HOME"]
    return env


def _build_output_schema() -> dict:
    """Return the output schema for codex exec, without minProperties."""
    return {
        "type": "object",
        "properties": {
            "primary_path": {"type": "string"},
            "files": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["primary_path", "files"],
        "additionalProperties": False,
    }


class CodexCliRunner:
    """Skeleton local runner for the Codex CLI."""

    def __init__(self) -> None:
        self._active_proc: Optional[asyncio.subprocess.Process] = None
        self._cancelled: bool = False

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

    async def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        """Run a single codex generation and yield exactly one terminal event.

        On success: yields one CompleteEvent and returns.
        On failure/cancel: yields one ErrorEvent and returns.
        """
        self._cancelled = False

        # Reject image input in this slice — not yet validated.
        if generation_input.input_image is not None:
            yield ErrorEvent(message="image input is not yet supported for codex")
            return

        # Build the workspace directory and output files.
        with tempfile.TemporaryDirectory(prefix="codex_ws_") as workspace:
            output_schema_path = os.path.join(workspace, "output_schema.json")
            output_file_path = os.path.join(workspace, "output.json")

            # Write the output schema.
            Path(output_schema_path).write_text(
                json.dumps(_build_output_schema())
            )

            # Build fixed argv for codex exec.
            argv: list[str] = [
                "codex",
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--json",
                "--output-schema",
                output_schema_path,
                "--output-last-message",
                output_file_path,
                "-C",
                workspace,
                "-",
            ]

            env = _build_child_env()

            try:
                self._active_proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            except FileNotFoundError:
                self._active_proc = None
                yield ErrorEvent(message=_MISSING_REASON)
                return
            except Exception:
                self._active_proc = None
                yield ErrorEvent(message=_GENERATION_ERROR_REASON)
                return

            try:
                # Send prompt via stdin, then wait for the child.
                try:
                    stdout_bytes, _ = await asyncio.wait_for(
                        self._active_proc.communicate(
                            generation_input.prompt.encode("utf-8")
                        ),
                        timeout=_GENERATE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    await self._terminate_generate_child()
                    self._active_proc = None
                    self._cancelled = False
                    yield ErrorEvent(message=_GENERATION_ERROR_REASON)
                    return

                # Check whether cancellation was requested during communicate.
                if self._cancelled:
                    await self._terminate_generate_child()
                    self._active_proc = None
                    self._cancelled = False
                    yield ErrorEvent(message=_GENERATION_ERROR_REASON)
                    return

                # Read and validate the output artifact.
                try:
                    event = self._parse_output_file(
                        output_file_path, generation_input.request_id
                    )
                except Exception:
                    event = ErrorEvent(message=_GENERATION_ERROR_REASON)

                self._active_proc = None
                yield event

            except Exception:
                self._active_proc = None
                yield ErrorEvent(message=_GENERATION_ERROR_REASON)

    async def _terminate_generate_child(self) -> None:
        """Stop and reap the active generate child with bounded escalation."""
        if self._active_proc is None:
            return
        proc = self._active_proc
        try:
            proc.terminate()
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(
                proc.wait(), timeout=_GENERATE_CLEANUP_TIMEOUT_SECONDS
            )
            return
        except ProcessLookupError:
            return
        except asyncio.TimeoutError:
            pass

        try:
            proc.kill()
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(
                proc.wait(), timeout=_GENERATE_CLEANUP_TIMEOUT_SECONDS
            )
        except (ProcessLookupError, asyncio.TimeoutError):
            pass

    def _parse_output_file(
        self, output_file_path: str, request_id: str
    ) -> CompleteEvent:
        """Parse and validate the --output-last-message artifact.

        Returns a CompleteEvent on success; raises on validation failure.
        """
        raw = Path(output_file_path).read_text(encoding="utf-8")
        data = json.loads(raw)

        primary_path = str(data.get("primary_path", ""))
        files_raw = data.get("files", {})

        if not isinstance(files_raw, dict):
            raise ValueError("files must be a JSON object")

        # Validate each file entry.
        text_files: dict[str, str] = {}
        seen_paths: set[str] = set()
        for key, content in files_raw.items():
            # Validate path safety.
            safe_key = self._validate_output_path(str(key))
            if safe_key in seen_paths:
                raise ValueError("duplicate output file path")
            seen_paths.add(safe_key)
            # Validate content.
            if not isinstance(content, str) or not content.strip():
                raise ValueError("output file content must be non-empty text")
            if "\x00" in content:
                raise ValueError("output file content must not contain NUL bytes")
            text_files[safe_key] = content

        if not text_files:
            raise ValueError("output must contain at least one file")

        # Validate primary_path.
        safe_primary = self._validate_output_path(primary_path) if primary_path else None
        if safe_primary is not None and safe_primary not in text_files:
            raise ValueError(
                "primary_path must be one of the output file keys"
            )

        status: GenerationStatus = "success"

        result = GenerationResult(
            provider="codex",
            primary_path=safe_primary,
            text_files=text_files,
            status=status,
        )
        return CompleteEvent(result=result)

    @staticmethod
    def _validate_output_path(value: str) -> str:
        """Validate a POSIX-style relative output file path.

        Rejects empty/whitespace, absolute paths, backslashes, drive letters,
        NUL bytes, and empty / '.' / '..' path segments.
        """
        if not isinstance(value, str):
            raise ValueError("output file path must be a string")
        if value != value.strip():
            raise ValueError("output file path must not have leading/trailing whitespace")
        if not value:
            raise ValueError("output file path must not be empty")
        if "\\" in value:
            raise ValueError("output file path must use POSIX separators ('/')")
        if value.startswith("/"):
            raise ValueError("output file path must be relative")
        if len(value) >= 2 and value[1] == ":":
            raise ValueError("output file path must be relative (no drive letter)")
        if "\x00" in value:
            raise ValueError("output file path must not contain NUL bytes")
        segments = value.split("/")
        if any(segment in ("", ".", "..") for segment in segments):
            raise ValueError(
                "output file path must not contain empty, '.', or '..' segments"
            )
        return value

    async def cancel(self) -> None:
        """Best-effort cancel: stop and reap the active child if any."""
        self._cancelled = True
        if self._active_proc is not None:
            await self._terminate_generate_child()
            self._active_proc = None
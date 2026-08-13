"""Unit tests for the Codex CLI local runner (v0.1 probe slice)."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest

from runners.codex import CodexCliRunner
from runners.models import (
    CompleteEvent,
    ErrorEvent,
    GenerationInput,
    GenerationResult,
    ProbeResult,
)


# --- capabilities() (side-effect free) ---------------------------------------


def test_capabilities_declares_codex_stacks_and_input_modes() -> None:
    caps = CodexCliRunner().capabilities()
    assert caps.runner_name == "codex"
    assert caps.available is False
    assert caps.supported_stacks == [
        "html_tailwind",
        "html_css",
        "react_tailwind",
        "vue_tailwind",
    ]
    assert caps.supported_input_modes == ["image", "text"]


def test_capabilities_never_probes_or_launches_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("capabilities() must not launch a vendor CLI")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", boom)

    caps = CodexCliRunner().capabilities()
    assert caps.available is False


# --- probe() ---------------------------------------------------------------


class _FakeProcess:
    """Minimal stand-in for asyncio.subprocess.Process."""

    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple:
        return self._stdout, self._stderr


class _HangingProcess:
    """A process whose communicate() never returns (for the timeout case)."""

    returncode: Optional[int] = None

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.awaited = False

    async def communicate(self) -> tuple:
        await asyncio.Event().wait()
        return b"", b""  # pragma: no cover - unreachable by design

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.awaited = True
        self.returncode = -15
        return self.returncode


async def test_probe_available_on_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(returncode=0, stdout=b"codex version 1.0.0\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await CodexCliRunner().probe()
    assert result == ProbeResult(available=True)
    assert result.unavailable_reason is None


async def test_probe_uses_fixed_literal_argv_and_no_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: List[tuple] = []
    shell_calls: List[tuple] = []

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        captured.append((args, kwargs))
        return _FakeProcess(returncode=0, stdout=b"codex version 1.0.0\n")

    async def fake_shell(*args: Any, **kwargs: Any) -> _FakeProcess:
        shell_calls.append((args, kwargs))
        return _FakeProcess(returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", fake_shell)

    result = await CodexCliRunner().probe()

    assert result.available is True
    # Fixed literal argv, no user-controlled executable or arguments.
    assert captured[0][0] == ("codex", "--version")
    # No shell, output captured via pipes.
    assert shell_calls == []
    assert captured[0][1].get("shell") is None
    assert captured[0][1].get("stdout") == asyncio.subprocess.PIPE
    assert captured[0][1].get("stderr") == asyncio.subprocess.PIPE


async def test_probe_missing_binary_returns_safe_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        raise FileNotFoundError("No such file or directory: 'codex'")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await CodexCliRunner().probe()
    assert result.available is False
    assert result.unavailable_reason is not None
    # Generic, safe, short — must not leak raw exception/output details.
    assert "No such file" not in result.unavailable_reason
    assert len(result.unavailable_reason) < 100


async def test_probe_timeout_returns_safe_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> _HangingProcess:
        return _HangingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("runners.codex._PROBE_TIMEOUT_SECONDS", 0.05)

    result = await CodexCliRunner().probe()
    assert result.available is False
    assert result.unavailable_reason is not None


async def test_probe_timeout_terminates_and_reaps_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On timeout the child must be stopped and reaped before returning."""

    proc = _HangingProcess()

    async def fake_exec(*args: Any, **kwargs: Any) -> _HangingProcess:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("runners.codex._PROBE_TIMEOUT_SECONDS", 0.05)

    result = await CodexCliRunner().probe()

    assert result.available is False
    assert result.unavailable_reason is not None
    # The timed-out child was terminated/killed and awaited (reaped).
    assert proc.terminated or proc.killed
    assert proc.awaited


class _SigtermIgnoringProcess:
    """A child whose communicate() hangs and which ignores SIGTERM.

    Its first wait() (after terminate) never returns, so cleanup must escalate
    to kill(); the second wait() (after kill) reaps with returncode -9.
    """

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False
        self.returncode: Optional[int] = None
        self.wait_calls = 0

    async def communicate(self) -> tuple:
        await asyncio.Event().wait()
        return b"", b""  # pragma: no cover - unreachable by design

    def terminate(self) -> None:
        self.terminated = True
        # SIGTERM is ignored: returncode stays None.

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            await asyncio.Event().wait()  # hang until killed
        assert self.returncode is not None
        return self.returncode


async def test_probe_timeout_escalates_to_kill_when_sigterm_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that ignores SIGTERM must be SIGKILLed and reaped, not hang."""

    proc = _SigtermIgnoringProcess()

    async def fake_exec(*args: Any, **kwargs: Any) -> _SigtermIgnoringProcess:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("runners.codex._PROBE_TIMEOUT_SECONDS", 0.05)

    # Outer bound so a missing escalation fails fast instead of hanging the
    # suite (the pre-fix code blocks forever on an unbounded await proc.wait()).
    result = await asyncio.wait_for(CodexCliRunner().probe(), timeout=2.0)

    assert result == ProbeResult(
        available=False, unavailable_reason="codex probe timed out"
    )
    # Escalation happened exactly: SIGTERM ignored -> SIGKILL -> eventual reap.
    assert proc.terminated
    assert proc.killed
    assert proc.wait_calls >= 2


async def test_probe_nonzero_exit_returns_safe_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(returncode=2, stderr=b"error: unknown flag --version")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    result = await CodexCliRunner().probe()
    assert result.available is False
    assert result.unavailable_reason is not None
    # Raw command output must never surface in the reason.
    assert "unknown flag" not in result.unavailable_reason


# --- generate() ---------------------------------------------------------------


class _GenerateFakeProcess:
    """Fake asyncio.subprocess.Process for generate() tests.

    Writes the supplied output JSON to a temporary file (simulating
    ``--output-last-message``), then returns code 0 via communicate().
    """

    def __init__(
        self,
        output_json: object,
        returncode: int = 0,
        stderr: bytes = b"",
    ):
        self.returncode: Optional[int] = None
        self._output_json = output_json
        self._retcode = returncode
        self._stderr = stderr
        self.terminated = False
        self.killed = False
        self._output_file: Optional[Path] = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        return b"", self._stderr

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._retcode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _fake_exec_success(
    output_schema_path: str,
) -> Any:
    """Return a factory that creates a successful fake process.

    Parses the output-schema path from the argv to write a dummy result file.
    """

    async def _factory(*args: Any, **kwargs: Any) -> _GenerateFakeProcess:
        # Extract the --output-schema and --output-last-message paths from argv.
        output_path = ""
        schema_path = ""
        for i, a in enumerate(args):
            if isinstance(a, str) and a == "--output-schema" and i + 1 < len(args):
                schema_path = str(args[i + 1])
            if isinstance(a, str) and a == "--output-last-message" and i + 1 < len(args):
                output_path = str(args[i + 1])

        # Write the output-schema file
        if schema_path:
            schema = {
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
            Path(schema_path).write_text(json.dumps(schema))

        # Write the output-last-message file with the result
        if output_path:
            result = {
                "primary_path": "index.html",
                "files": {"index.html": "<html><body>Hello</body></html>"},
            }
            Path(output_path).write_text(json.dumps(result))

        return _GenerateFakeProcess(output_json=result)

    return _factory


async def test_generate_success_emits_valid_complete_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful Codex run yields exactly one CompleteEvent with a valid result."""
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_exec_success("")
    )

    gen_input = GenerationInput(
        request_id="r1",
        prompt="Build a hello world page",
        stack="html_tailwind",
        generation_type="create",
    )

    runner = CodexCliRunner()
    events: list[object] = []
    async for event in runner.generate(gen_input):
        events.append(event)

    # Exactly one terminal event.
    assert len(events) == 1
    assert isinstance(events[0], CompleteEvent)
    cevent: CompleteEvent = events[0]
    assert cevent.result is not None
    assert cevent.result.provider == "codex"
    assert cevent.result.status == "success"
    assert cevent.result.primary_path == "index.html"
    assert "index.html" in cevent.result.text_files
    assert cevent.result.text_files["index.html"] == "<html><body>Hello</body></html>"


# --- cancel() ---------------------------------------------------------------


async def test_cancel_stops_active_child_and_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel() on an active generation terminates the child and generate yields ErrorEvent."""
    cancelled = asyncio.Event()

    class _HangingGenerateProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False
            self._done = asyncio.Event()

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            # Hang until cancelled signals.
            await cancelled.wait()
            return b"", b""

        async def wait(self) -> int:
            await self._done.wait()
            return -15

        def terminate(self) -> None:
            self.terminated = True
            self._done.set()
            cancelled.set()

        def kill(self) -> None:
            self.killed = True

    hanging_proc = _HangingGenerateProcess()

    async def fake_exec(*args: Any, **kwargs: Any) -> _HangingGenerateProcess:
        # Write valid output files so the parse step doesn't fail.
        schema_path = ""
        output_path = ""
        for i, a in enumerate(args):
            if a == "--output-schema" and i + 1 < len(args):
                schema_path = str(args[i + 1])
            if a == "--output-last-message" and i + 1 < len(args):
                output_path = str(args[i + 1])
        if schema_path:
            Path(schema_path).write_text(
                json.dumps(
                    {
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
                )
            )
        if output_path:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "primary_path": "index.html",
                        "files": {"index.html": "x"},
                    }
                )
            )
        return hanging_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    gen_input = GenerationInput(
        request_id="r1",
        prompt="test",
        stack="html_tailwind",
        generation_type="create",
    )

    runner = CodexCliRunner()
    events: list[object] = []

    async def _collect() -> None:
        async for event in runner.generate(gen_input):
            events.append(event)

    task = asyncio.create_task(_collect())
    # Give generate() time to reach the communicate() call.
    await asyncio.sleep(0.05)
    await runner.cancel()
    await task

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert hanging_proc.terminated


async def test_generate_fails_on_malformed_output_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A codex run producing unparseable output yields exactly one ErrorEvent."""

    async def fake_exec(*args: Any, **kwargs: Any) -> _GenerateFakeProcess:
        # Write the output file with invalid JSON.
        output_path = ""
        schema_path = ""
        for i, a in enumerate(args):
            if a == "--output-schema" and i + 1 < len(args):
                schema_path = str(args[i + 1])
            if a == "--output-last-message" and i + 1 < len(args):
                output_path = str(args[i + 1])
        if schema_path:
            Path(schema_path).write_text(
                json.dumps(
                    {
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
                )
            )
        if output_path:
            Path(output_path).write_text("not valid json {{{")
        return _GenerateFakeProcess(output_json={})

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    gen_input = GenerationInput(
        request_id="r1",
        prompt="test",
        stack="html_tailwind",
        generation_type="create",
    )

    runner = CodexCliRunner()
    events: list[object] = []
    async for event in runner.generate(gen_input):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    # Error message must be generic — never expose raw output.
    assert "not valid json" not in events[0].message

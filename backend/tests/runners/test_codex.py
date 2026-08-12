"""Unit tests for the Codex CLI local runner (v0.1 probe slice)."""

import asyncio
from typing import Any, List, Optional

import pytest

from runners.codex import CodexCliRunner
from runners.models import GenerationInput, ProbeResult


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


# --- protocol stubs ----------------------------------------------------------


async def test_cancel_is_a_noop_and_generate_is_a_stub() -> None:
    runner = CodexCliRunner()
    assert await runner.cancel() is None
    gen_input = GenerationInput(
        request_id="r",
        prompt="p",
        stack="html_tailwind",
        generation_type="create",
    )
    with pytest.raises(NotImplementedError):
        runner.generate(gen_input)

import importlib
from typing import AsyncIterator, Iterator

import pytest

import config
import preview_screenshot
import runners.capabilities as runner_caps
import runners.registry as registry
from routes.capabilities import get_capabilities
from runners.models import (
    GenerationEvent,
    GenerationInput,
    ProbeResult,
    RunnerCapabilities,
)


@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every test from the process-global registry state."""
    monkeypatch.setattr(registry, "_registry", {})
    yield


class FakeRunner:
    """Minimal structural runner for capability-state tests; never probes."""

    def __init__(self, name: str, available: bool = False) -> None:
        self._name = name
        self._available = available
        self.probe_calls = 0
        self.generate_calls = 0

    async def probe(self) -> ProbeResult:
        self.probe_calls += 1
        return ProbeResult(available=self._available)

    def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        self.generate_calls += 1
        raise NotImplementedError("not used in capability-state tests")

    async def cancel(self) -> None:
        return None

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            runner_name=self._name,
            available=self._available,
            supported_stacks=["html_tailwind"],
            supported_input_modes=["image"],
        )


class BombRunner(FakeRunner):
    """A runner whose probe/generate must never run during capability reads."""

    async def probe(self) -> ProbeResult:
        raise AssertionError("probe() must not be called by the capability endpoint")

    def generate(
        self, generation_input: GenerationInput
    ) -> AsyncIterator[GenerationEvent]:
        raise AssertionError("generate() must not be called by the capability endpoint")


# --- Backward compatibility: screenshot_preview -----------------------------


@pytest.mark.asyncio
async def test_capabilities_reports_screenshot_preview_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe() -> bool:
        return True

    monkeypatch.setattr(preview_screenshot, "probe_screenshot_preview", fake_probe)
    monkeypatch.setattr("routes.capabilities.probe_screenshot_preview", fake_probe)

    result = await get_capabilities()
    assert result.screenshot_preview is True


@pytest.mark.asyncio
async def test_capabilities_reports_screenshot_preview_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe() -> bool:
        return False

    monkeypatch.setattr("routes.capabilities.probe_screenshot_preview", fake_probe)

    result = await get_capabilities()
    assert result.screenshot_preview is False


# --- Local runner state on the endpoint --------------------------------------


@pytest.mark.asyncio
async def test_capabilities_reports_local_runner_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_probe() -> bool:
        return True

    monkeypatch.setattr("routes.capabilities.probe_screenshot_preview", fake_probe)
    monkeypatch.setattr(runner_caps, "GENERATION_RUNNER", None)

    result = await get_capabilities()
    # screenshot_preview output is unchanged.
    assert result.screenshot_preview is True
    # The new local-runner state is reported in a stable, safe shape.
    assert result.local_runner.enabled is False
    assert result.local_runner.available is False
    assert result.local_runner.name is None
    assert result.local_runner.unavailable_reason is not None


@pytest.mark.asyncio
async def test_capabilities_registered_runner_never_probes_or_generates(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    async def fake_probe() -> bool:
        return True

    monkeypatch.setattr("routes.capabilities.probe_screenshot_preview", fake_probe)
    registry.register("codex", BombRunner("codex"))
    monkeypatch.setattr(runner_caps, "GENERATION_RUNNER", "codex")

    result = await get_capabilities()
    # Declared capabilities are reported without any probe/generate execution.
    assert result.local_runner.enabled is True
    assert result.local_runner.name == "codex"
    assert result.local_runner.supported_stacks == ["html_tailwind"]
    assert result.local_runner.supported_input_modes == ["image"]


# --- local_runner_state() helper --------------------------------------------


def test_local_runner_state_disabled_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_caps, "GENERATION_RUNNER", None)

    state = runner_caps.local_runner_state()
    assert state.enabled is False
    assert state.available is False
    assert state.name is None
    assert state.unavailable_reason is not None


def test_local_runner_state_configured_but_runner_missing_from_registry(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    monkeypatch.setattr(runner_caps, "GENERATION_RUNNER", "codex")

    state = runner_caps.local_runner_state()
    assert state.enabled is True
    assert state.available is False
    assert state.name == "codex"
    assert "not registered" in (state.unavailable_reason or "")


def test_local_runner_state_configured_unknown_runner_never_invokes(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    """An invalid/unknown configured name yields a safe reason, no invocation."""
    registry.register("codex", BombRunner("codex"))
    monkeypatch.setattr(runner_caps, "GENERATION_RUNNER", "does-not-exist")

    state = runner_caps.local_runner_state()
    assert state.enabled is True
    assert state.available is False
    assert state.unavailable_reason is not None


def test_local_runner_state_registered_runner_reports_declared_capabilities(
    monkeypatch: pytest.MonkeyPatch, empty_registry
) -> None:
    runner = FakeRunner("codex")
    registry.register("codex", runner)
    monkeypatch.setattr(runner_caps, "GENERATION_RUNNER", "codex")

    state = runner_caps.local_runner_state()
    assert state.enabled is True
    assert state.name == "codex"
    # No probe happened: availability stays at the runner's declared default.
    assert state.available is False
    assert runner.probe_calls == 0
    assert state.supported_stacks == ["html_tailwind"]
    assert state.supported_input_modes == ["image"]


# --- Config ------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["1", "true", "TRUE", "yes", "on", " True ", " 1 "],
)
def test_parse_truthy_env_accepts_explicit_truthy_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GENERATION_RUNNER_ENABLED", value)
    assert config.parse_truthy_env("GENERATION_RUNNER_ENABLED") is True


@pytest.mark.parametrize(
    "value",
    ["", "0", "false", "FALSE", "no", "off", "maybe", "2", "null"],
)
def test_parse_truthy_env_rejects_everything_else(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GENERATION_RUNNER_ENABLED", value)
    assert config.parse_truthy_env("GENERATION_RUNNER_ENABLED") is False


def test_generation_runner_config_defaults_to_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENERATION_RUNNER", raising=False)
    importlib.reload(config)

    assert config.GENERATION_RUNNER is None


@pytest.mark.parametrize("value", ["", "   ", "\t"])
def test_generation_runner_config_treats_blank_as_unconfigured(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GENERATION_RUNNER", value)
    importlib.reload(config)

    assert config.GENERATION_RUNNER is None


def test_generation_runner_config_reads_configured_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATION_RUNNER", " codex_cli ")
    importlib.reload(config)

    assert config.GENERATION_RUNNER == "codex_cli"

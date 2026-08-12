"""Unit tests for the local coding-agent runner registry."""

from typing import AsyncIterator, Iterator, Optional

import pytest

import runners.registry as registry
from runners.models import (
    GenerationEvent,
    GenerationInput,
    ProbeResult,
    RunnerCapabilities,
)


class FakeRunner:
    """Minimal structural runner for registry tests; never probes anything."""

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
        raise NotImplementedError("not used in registry tests")

    async def cancel(self) -> None:
        return None

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            runner_name=self._name,
            available=self._available,
            supported_stacks=["html_tailwind", "react_tailwind"],
            supported_input_modes=["image"],
        )


@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate every test from the process-global registry state."""
    monkeypatch.setattr(registry, "_registry", {})
    yield


def test_registry_starts_empty(empty_registry) -> None:
    assert registry.names() == []
    assert registry.is_registered("codex") is False
    assert registry.get("codex") is None
    assert registry.list_capabilities() == []


def test_register_and_lookup(empty_registry) -> None:
    runner = FakeRunner("codex")
    registry.register("codex", runner)
    assert registry.is_registered("codex") is True
    assert registry.get("codex") is runner
    assert registry.names() == ["codex"]


def test_register_rejects_blank_name(empty_registry) -> None:
    runner = FakeRunner("codex")
    for bad_name in ("", "   ", "\t\n"):
        with pytest.raises(ValueError):
            registry.register(bad_name, runner)


def test_register_rejects_duplicate_name(empty_registry) -> None:
    registry.register("codex", FakeRunner("codex"))
    with pytest.raises(ValueError):
        registry.register("codex", FakeRunner("codex-2"))


def test_unregister_removes_runner_and_unknown_is_noop(empty_registry) -> None:
    registry.register("codex", FakeRunner("codex"))
    registry.unregister("codex")
    assert registry.get("codex") is None
    assert registry.names() == []
    # Unregistering a never-registered name must not raise.
    registry.unregister("never-registered")


def test_list_capabilities_enumerates_declared_reports_without_probing(
    empty_registry,
) -> None:
    codex = FakeRunner("codex", available=False)
    opencode = FakeRunner("opencode", available=True)
    registry.register("codex", codex)
    registry.register("opencode", opencode)

    reports = registry.list_capabilities()
    # Deterministic order by registration name.
    assert [report.runner_name for report in reports] == ["codex", "opencode"]
    codex_report = reports[0]
    assert codex_report.available is False
    assert codex_report.supported_stacks == ["html_tailwind", "react_tailwind"]
    assert codex_report.supported_input_modes == ["image"]
    # Enumerating capabilities must never probe (or generate).
    assert codex.probe_calls == 0
    assert codex.generate_calls == 0
    assert opencode.probe_calls == 0


def test_get_missing_runner_returns_none_safely(empty_registry) -> None:
    assert registry.get("codex") is None
    assert registry.get("") is None


@pytest.mark.parametrize("name", [None, "", "   ", "never-registered"])
def test_select_returns_none_safely_for_unconfigured_or_unknown(
    empty_registry, name: Optional[str]
) -> None:
    """``select`` never raises: None for unset, blank, or unknown names."""
    assert registry.select(name) is None


def test_select_returns_registered_runner(empty_registry) -> None:
    runner = FakeRunner("codex")
    registry.register("codex", runner)
    assert registry.select("codex") is runner
    # Surrounding whitespace in the configured name is tolerated.
    assert registry.select("  codex  ") is runner

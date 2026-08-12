"""Unit tests for the provider-neutral runner data models.

These encode the *corrected* contract: results identify a ``provider``, report
elapsed time in milliseconds, carry an unambiguous ``status``, and validate
output paths/content; inputs constrain ``stack`` to the exact four local-runner
stacks rather than the broader upstream ``Stack`` alias.
"""

from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError

from prompts.prompt_types import Stack
from runners.models import (
    AssistantEvent,
    CompleteEvent,
    ErrorEvent,
    GenerationEvent,
    GenerationInput,
    GenerationResult,
    GenerationStatus,
    GenerationType,
    LocalRunnerState,
    ProbeResult,
    RunnerCapabilities,
    RunnerStack,
    SetCodeEvent,
    StatusEvent,
    ThinkingEvent,
)


# --- GenerationInput --------------------------------------------------------


def test_generation_input_accepts_local_image_path(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    gen_input = GenerationInput(
        request_id="req-1",
        prompt="Build a landing page",
        stack="html_tailwind",
        generation_type="create",
        input_image=image,
    )
    assert gen_input.request_id == "req-1"
    assert gen_input.prompt == "Build a landing page"
    assert gen_input.stack == "html_tailwind"
    assert gen_input.generation_type == "create"
    assert gen_input.input_image == image
    assert gen_input.prior_files == []
    assert gen_input.design_system is None


def test_generation_input_accepts_each_supported_stack() -> None:
    for stack in ("html_tailwind", "html_css", "react_tailwind", "vue_tailwind"):
        gen_input = GenerationInput(
            request_id="r", prompt="p", stack=stack, generation_type="update"
        )
        assert gen_input.stack == stack


def test_generation_input_rejects_unknown_stack() -> None:
    with pytest.raises(ValidationError):
        GenerationInput(
            request_id="req-1",
            prompt="x",
            stack=cast(RunnerStack, "not_a_stack"),
            generation_type="create",
        )


def test_generation_input_rejects_stacks_from_broader_upstream_alias() -> None:
    """The local-runner contract allows exactly four stacks, even though the
    upstream ``Stack`` alias additionally includes ``bootstrap`` and
    ``ionic_tailwind``."""
    assert "bootstrap" in set(Stack.__args__)  # type: ignore[attr-defined]
    for rejected in ("bootstrap", "ionic_tailwind"):
        with pytest.raises(ValidationError):
            GenerationInput(
                request_id="r",
                prompt="p",
                stack=cast(RunnerStack, rejected),
                generation_type="create",
            )


def test_generation_input_rejects_unknown_generation_type() -> None:
    with pytest.raises(ValidationError):
        GenerationInput(
            request_id="req-1",
            prompt="x",
            stack="html_tailwind",
            generation_type=cast(GenerationType, "delete"),
        )


def test_generation_input_rejects_blank_request_id() -> None:
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValidationError):
            GenerationInput(
                request_id=bad,
                prompt="p",
                stack="html_tailwind",
                generation_type="create",
            )


def test_generation_input_rejects_blank_prompt() -> None:
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValidationError):
            GenerationInput(
                request_id="r",
                prompt=bad,
                stack="html_tailwind",
                generation_type="create",
            )


def test_generation_input_rejects_blank_input_image() -> None:
    with pytest.raises(ValidationError):
        GenerationInput(
            request_id="r",
            prompt="p",
            stack="html_tailwind",
            generation_type="create",
            input_image="",  # type: ignore[arg-type]
        )


def test_generation_input_rejects_blank_prior_file() -> None:
    with pytest.raises(ValidationError):
        GenerationInput(
            request_id="r",
            prompt="p",
            stack="html_tailwind",
            generation_type="create",
            prior_files=[""],  # type: ignore[list-item]
        )


def test_generation_input_accepts_nonblank_design_system() -> None:
    gen_input = GenerationInput(
        request_id="r",
        prompt="p",
        stack="html_tailwind",
        generation_type="create",
        design_system="Use the brand colors",
    )
    assert gen_input.design_system == "Use the brand colors"


def test_generation_input_rejects_blank_design_system() -> None:
    for bad in ("", "   ", "\t\n"):
        with pytest.raises(ValidationError):
            GenerationInput(
                request_id="r",
                prompt="p",
                stack="html_tailwind",
                generation_type="create",
                design_system=bad,
            )


# --- GenerationResult -------------------------------------------------------


def test_generation_result_defaults_and_requires_provider_and_status() -> None:
    result = GenerationResult(provider="codex", status="success")
    assert result.provider == "codex"
    assert result.runner_version is None
    assert result.elapsed_ms is None
    assert result.primary_path is None
    assert result.text_files == {}
    assert result.status == "success"


def test_generation_result_rejects_blank_provider() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(provider="   ", status="success")


def test_generation_result_requires_status() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(provider="codex")  # type: ignore[call-arg]


def test_generation_result_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(
            provider="codex",
            status=cast(GenerationStatus, "pending"),
        )


def test_generation_result_rejects_negative_elapsed_ms() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(provider="codex", status="success", elapsed_ms=-1)


def test_generation_result_accepts_full_shape() -> None:
    result = GenerationResult(
        provider="codex",
        runner_version="0.1.0",
        elapsed_ms=12500,
        primary_path="index.html",
        text_files={"index.html": "<html></html>"},
        status="success",
    )
    assert result.provider == "codex"
    assert result.runner_version == "0.1.0"
    assert result.elapsed_ms == 12500
    assert result.primary_path == "index.html"
    assert result.text_files["index.html"] == "<html></html>"
    assert result.status == "success"


# --- Output path / content validation ---------------------------------------


def test_generation_result_accepts_nested_relative_path() -> None:
    result = GenerationResult(
        provider="codex",
        status="success",
        primary_path="src/components/App.tsx",
        text_files={"src/components/App.tsx": "export default App"},
    )
    assert result.primary_path == "src/components/App.tsx"


@pytest.mark.parametrize(
    "primary_path",
    ["/etc/passwd", "/index.html", "C:/index.html", "index.html\\..\\secret"],
)
def test_generation_result_rejects_absolute_or_windows_primary_path(
    primary_path: str,
) -> None:
    with pytest.raises(ValidationError):
        GenerationResult(
            provider="codex",
            status="success",
            primary_path=primary_path,
            text_files={primary_path: "x"},
        )


@pytest.mark.parametrize(
    "bad_key",
    [
        "",  # empty
        "   ",  # whitespace only
        "/abs/path.txt",  # absolute
        "a/../b.txt",  # parent traversal
        "../b.txt",
        "a/./b.txt",  # dot segment
        "./b.txt",
        "a//b.txt",  # empty segment
        "a\\b.txt",  # backslash separator
        "C:foo.txt",  # drive letter
    ],
)
def test_generation_result_rejects_invalid_text_file_keys(bad_key: str) -> None:
    with pytest.raises(ValidationError):
        GenerationResult(
            provider="codex",
            status="success",
            text_files={bad_key: "content"},
        )


def test_generation_result_rejects_nul_byte_in_content() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(
            provider="codex",
            status="success",
            text_files={"index.html": "a\x00b"},
        )


def test_generation_result_rejects_primary_path_not_in_text_files() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(
            provider="codex",
            status="success",
            primary_path="index.html",
            text_files={"style.css": "body {}"},
        )


def test_generation_result_rejects_nul_byte_in_primary_path() -> None:
    with pytest.raises(ValidationError):
        GenerationResult(
            provider="codex",
            status="success",
            primary_path="a\x00b.html",
            text_files={"a\x00b.html": "x"},
        )


# --- Events -----------------------------------------------------------------


def test_generation_event_union_parses_each_type() -> None:
    adapter: TypeAdapter[GenerationEvent] = TypeAdapter(GenerationEvent)

    status = adapter.validate_python({"type": "status", "message": "running"})
    assert isinstance(status, StatusEvent)
    assert status.message == "running"

    thinking = adapter.validate_python({"type": "thinking", "text": "hmm"})
    assert isinstance(thinking, ThinkingEvent)

    assistant = adapter.validate_python({"type": "assistant", "text": "hi"})
    assert isinstance(assistant, AssistantEvent)

    set_code = adapter.validate_python(
        {"type": "setCode", "file_name": "index.html", "code": "<html>"}
    )
    assert isinstance(set_code, SetCodeEvent)
    assert set_code.file_name == "index.html"

    error = adapter.validate_python({"type": "error", "message": "boom"})
    assert isinstance(error, ErrorEvent)
    assert error.message == "boom"

    complete = adapter.validate_python({"type": "complete"})
    assert isinstance(complete, CompleteEvent)
    assert complete.result is None


def test_generation_event_union_rejects_unknown_type() -> None:
    adapter: TypeAdapter[GenerationEvent] = TypeAdapter(GenerationEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "teleport"})


def test_complete_event_carries_result() -> None:
    result = GenerationResult(
        provider="codex",
        status="success",
        primary_path="index.html",
        text_files={"index.html": "<html></html>"},
    )
    event = CompleteEvent(result=result)
    assert event.result == result
    assert event.result is not None
    assert event.result.primary_path == "index.html"


@pytest.mark.parametrize(
    "bad_name",
    [
        "",  # empty
        "   ",  # whitespace only
        "/abs/index.html",  # absolute
        "a/../b.txt",  # parent traversal
        "a/./b.txt",  # dot segment
        "a//b.txt",  # empty segment
        "a\\b.txt",  # backslash separator
        "C:/index.html",  # drive letter
    ],
)
def test_set_code_event_rejects_unsafe_file_name(bad_name: str) -> None:
    with pytest.raises(ValidationError):
        SetCodeEvent(file_name=bad_name, code="<html>")


def test_set_code_event_rejects_nul_byte_in_file_name() -> None:
    with pytest.raises(ValidationError):
        SetCodeEvent(file_name="a\x00b.html", code="<html>")


def test_set_code_event_rejects_nul_byte_in_code() -> None:
    with pytest.raises(ValidationError):
        SetCodeEvent(file_name="index.html", code="a\x00b")


# --- Capabilities / probe / state ------------------------------------------


def test_runner_capabilities_defaults_to_unavailable() -> None:
    caps = RunnerCapabilities(runner_name="codex")
    assert caps.available is False
    assert caps.supported_stacks == []
    assert caps.supported_input_modes == []
    assert caps.unavailable_reason is None


def test_runner_capabilities_rejects_blank_name() -> None:
    with pytest.raises(ValidationError):
        RunnerCapabilities(runner_name="")


def test_runner_capabilities_accepts_declared_shape() -> None:
    caps = RunnerCapabilities(
        runner_name="codex",
        available=False,
        supported_stacks=["html_tailwind", "react_tailwind"],
        supported_input_modes=["image"],
        unavailable_reason="binary not found",
    )
    assert caps.supported_stacks == ["html_tailwind", "react_tailwind"]
    assert caps.supported_input_modes == ["image"]
    assert caps.unavailable_reason == "binary not found"


def test_runner_capabilities_rejects_stacks_outside_exact_set() -> None:
    with pytest.raises(ValidationError):
        RunnerCapabilities(
            runner_name="codex",
            supported_stacks=[cast(RunnerStack, "bootstrap")],
        )


def test_probe_result_shape() -> None:
    ok = ProbeResult(available=True)
    assert ok.available is True
    assert ok.unavailable_reason is None

    blocked = ProbeResult(available=False, unavailable_reason="missing binary")
    assert blocked.available is False
    assert blocked.unavailable_reason == "missing binary"


def test_local_runner_state_defaults_are_safe_and_disabled() -> None:
    state = LocalRunnerState(enabled=False)
    assert state.enabled is False
    assert state.name is None
    assert state.available is False
    assert state.supported_stacks == []
    assert state.supported_input_modes == []

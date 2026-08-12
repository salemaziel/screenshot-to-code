"""Provider-neutral data models for local coding-agent runners.

These define the input/output contract shared by every local runner (e.g. a
future Codex subprocess runner) so the rest of the app can select and query
runners without knowing how a specific vendor CLI works. The models carry no
API keys, credentials, or vendor-specific fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Union, cast

from pydantic import BaseModel, Field, field_validator, model_validator

from custom_types import InputMode

# The exact set of stacks the local-runner contract supports. Deliberately
# narrower than ``prompts.prompt_types.Stack``: local runners support only
# these four, regardless of the broader upstream alias (which also includes
# ``bootstrap`` and ``ionic_tailwind``).
RunnerStack = Literal[
    "html_tailwind",
    "html_css",
    "react_tailwind",
    "vue_tailwind",
]

GenerationType = Literal["create", "update"]

GenerationStatus = Literal["success", "failure", "cancelled"]


def _empty_path_list() -> List[Path]:
    return []


def _empty_string_dict() -> Dict[str, str]:
    return {}


def _empty_stack_list() -> List[RunnerStack]:
    return []


def _empty_input_mode_list() -> List[InputMode]:
    return []


def _validate_relative_path(value: str) -> str:
    """Validate a POSIX-style relative output file key.

    Rejects empty/whitespace-only paths, absolute paths, Windows separators
    and drive letters, NUL bytes, and any empty/``.``/``..`` path segment.
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


def _validate_text_content(value: str) -> str:
    """Reject unsafe/binary-like text file content (at minimum NUL bytes)."""
    if "\x00" in value:
        raise ValueError("text file content must not contain NUL bytes")
    return value


def _reject_blank_entries(entries: object) -> None:
    """Reject blank string entries in a raw list/tuple of server-side paths."""
    if not isinstance(entries, (list, tuple)):
        return
    for item in cast(Sequence[object], entries):
        if isinstance(item, str) and not item.strip():
            raise ValueError("prior_files entries must not be blank")


class GenerationInput(BaseModel):
    """Everything a runner needs to generate code for a single user turn.

    ``input_image`` is a server-owned local filesystem path: the local-runner
    contract never carries remote URLs or upload identifiers, so callers must
    materialize the image locally before invoking a runner. ``prior_files``
    lists the local files that already exist (for update turns);
    ``design_system`` is the optional design-system text to apply. The image
    and prior-file paths are server-side values, so no filesystem existence
    check is performed here (such a check is not a security boundary).
    """

    request_id: str
    prompt: str
    stack: RunnerStack
    generation_type: GenerationType
    input_image: Optional[Path] = None
    prior_files: List[Path] = Field(default_factory=_empty_path_list)
    design_system: Optional[str] = None

    @field_validator("request_id")
    @classmethod
    def _request_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request_id must not be blank")
        return stripped

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("input_image", mode="before")
    @classmethod
    def _input_image_not_blank(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            raise ValueError("input_image must not be blank")
        return value

    @field_validator("prior_files", mode="before")
    @classmethod
    def _prior_files_not_blank(cls, value: Any) -> Any:
        _reject_blank_entries(value)
        return value


class GenerationResult(BaseModel):
    """Validated outcome of a completed generation.

    ``text_files`` maps a relative, POSIX-style output path to its text
    content; ``primary_path`` is the main generated file when one exists (a
    text-only result may have none) and, when supplied, must be one of the
    ``text_files`` keys. Output paths are validated to stay inside the result
    (relative, no traversal). ``status`` is the unambiguous outcome of the
    run.
    """

    provider: str
    runner_version: Optional[str] = None
    elapsed_ms: Optional[int] = None
    primary_path: Optional[str] = None
    text_files: Dict[str, str] = Field(default_factory=_empty_string_dict)
    status: GenerationStatus

    @field_validator("provider")
    @classmethod
    def _provider_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("provider must not be blank")
        return stripped

    @field_validator("elapsed_ms")
    @classmethod
    def _elapsed_ms_non_negative(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("elapsed_ms must not be negative")
        return value

    @field_validator("text_files")
    @classmethod
    def _validate_text_files(cls, value: Dict[str, str]) -> Dict[str, str]:
        validated: Dict[str, str] = {}
        for key, content in value.items():
            safe_key = _validate_relative_path(key)
            validated[safe_key] = _validate_text_content(content)
        return validated

    @field_validator("primary_path")
    @classmethod
    def _validate_primary_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def _primary_path_is_output_key(self) -> "GenerationResult":
        if self.primary_path is not None and self.primary_path not in self.text_files:
            raise ValueError("primary_path must be one of the text_files keys")
        return self


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    message: str


class AssistantEvent(BaseModel):
    type: Literal["assistant"] = "assistant"
    text: str


class ThinkingEvent(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str


class SetCodeEvent(BaseModel):
    type: Literal["setCode"] = "setCode"
    file_name: str
    code: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


class CompleteEvent(BaseModel):
    type: Literal["complete"] = "complete"
    result: Optional[GenerationResult] = None


GenerationEvent = Annotated[
    Union[
        StatusEvent,
        AssistantEvent,
        ThinkingEvent,
        SetCodeEvent,
        ErrorEvent,
        CompleteEvent,
    ],
    Field(discriminator="type"),
]
"""Normalized stream event yielded by a runner's ``generate()``.

A generation stream terminates with exactly one ``CompleteEvent`` (carrying the
``GenerationResult``) on success, or one ``ErrorEvent`` before the stream ends
on failure.
"""


class RunnerCapabilities(BaseModel):
    """Declared capabilities of a local runner, safe to report to callers.

    ``available`` is the last known probe outcome and defaults to False
    (unavailable) until a probe succeeds; constructing this model never
    launches a vendor CLI. ``unavailable_reason`` is a short, safe,
    user-facing explanation for why the runner is unavailable; it must not
    leak raw command output or secrets.
    """

    runner_name: str
    available: bool = False
    supported_stacks: List[RunnerStack] = Field(default_factory=_empty_stack_list)
    supported_input_modes: List[InputMode] = Field(
        default_factory=_empty_input_mode_list
    )
    unavailable_reason: Optional[str] = None

    @field_validator("runner_name")
    @classmethod
    def _runner_name_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("runner_name must not be blank")
        return stripped

    @field_validator("unavailable_reason")
    @classmethod
    def _reason_is_short_and_clean(
        cls, value: Optional[str]
    ) -> Optional[str]:
        """Trim and bound the reason so it stays a safe, short message."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        return stripped[:500]


class ProbeResult(BaseModel):
    """Outcome of an actual availability probe.

    Unlike ``RunnerCapabilities``, this reflects a real check and may launch a
    vendor CLI. ``unavailable_reason`` follows the same short/safe rules as
    ``RunnerCapabilities``.
    """

    available: bool
    unavailable_reason: Optional[str] = None


class LocalRunnerState(BaseModel):
    """Configured local-runner state reported by ``GET /api/capabilities``.

    Computed from configuration and the registry only — never by launching a
    runner binary. ``enabled`` reflects whether a local runner has been
    configured/selected; ``available`` is False until a registered runner has
    been probed; declared ``supported_stacks``/``supported_input_modes`` are
    only populated when a runner is registered.
    """

    enabled: bool
    name: Optional[str] = None
    available: bool = False
    unavailable_reason: Optional[str] = None
    supported_stacks: List[RunnerStack] = Field(default_factory=_empty_stack_list)
    supported_input_modes: List[InputMode] = Field(
        default_factory=_empty_input_mode_list
    )

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import traceback
from typing import Callable, Awaitable
from fastapi import APIRouter, WebSocket
import openai
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from config import (
    ANTHROPIC_API_KEY,
    GENERATION_RUNNER,
    GEMINI_API_KEY,
    IS_DEBUG_ENABLED,
    IS_PROD,
    NUM_VARIANTS,
    NUM_VARIANTS_VIDEO,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    REPLICATE_API_KEY,
)
from custom_types import InputMode
from llm import (
    Llm,
)
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    Literal,
    cast,
    get_args,
)
from weakref import WeakKeyDictionary
from openai.types.chat import ChatCompletionMessageParam

from utils import print_prompt_preview

# WebSocket message types
MessageType = Literal[
    "chunk",
    "status",
    "setCode",
    "error",
    "variantComplete",
    "variantError",
    "variantCount",
    "variantModels",
    "thinking",
    "assistant",
    "toolStart",
    "toolResult",
]
from prompts.pipeline import build_prompt_messages
from prompts.request_parsing import parse_prompt_content, parse_prompt_history
from prompts.prompt_types import PromptHistoryMessage, Stack, UserTurnInput
from uploaded_assets import (
    append_uploaded_asset_ids_to_history,
    append_uploaded_asset_ids_to_prompt,
    infer_local_asset_base_url,
)
from agent.runner import Agent
from fs_logging.agent_runs import AgentRunRecorder
from routes.local_runner_adapter import LocalRunnerWebSocketAdapter
from runners.base import Runner
from runners.codex import CodexCliRunner
from runners.models import (
    CompleteEvent,
    GenerationEvent,
    GenerationInput,
    ProbeResult,
    RunnerStack,
    SetCodeEvent,
)
from runners.registry import (
    is_registered as runner_is_registered,
    register as runner_register,
    select as runner_select,
)
from routes.model_choice_sets import (
    ALL_KEYS_MODELS_DEFAULT,
    ALL_KEYS_MODELS_TEXT_CREATE,
    ALL_KEYS_MODELS_UPDATE,
    ANTHROPIC_ONLY_MODELS,
    GEMINI_ANTHROPIC_MODELS,
    GEMINI_OPENAI_MODELS,
    GEMINI_ONLY_MODELS,
    OPENAI_ANTHROPIC_MODELS,
    OPENAI_ONLY_MODELS,
    VIDEO_VARIANT_MODELS,
)

# from utils import pprint_prompt
from ws.constants import APP_ERROR_WEB_SOCKET_CODE  # type: ignore


router = APIRouter()


# Local-runner dispatch (v0.1): only the codex runner is ever registered and
# selected. Fixed, client-safe failure messages — never interpolate runner
# output, provider names, model identifiers, or artifact paths into these.
_LOCAL_RUNNER_NAME = "codex"
_LOCAL_UNSUPPORTED_MESSAGE = (
    "This request is not supported by the configured local runner."
)
_LOCAL_RUNNER_UNAVAILABLE_FALLBACK = "The local runner is not available."
_MAX_UNAVAILABLE_REASON_LENGTH = 500


@dataclass
class PipelineContext:
    """Context object that carries state through the pipeline"""

    websocket: WebSocket
    ws_comm: "WebSocketCommunicator | None" = None
    params: Dict[str, Any] = field(default_factory=dict)
    extracted_params: "ExtractedParams | None" = None
    prompt_messages: List[ChatCompletionMessageParam] = field(default_factory=list)
    variant_models: List[Llm] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    variant_completions: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def send_message(self):
        assert self.ws_comm is not None
        return self.ws_comm.send_message

    @property
    def throw_error(self):
        assert self.ws_comm is not None
        return self.ws_comm.throw_error


class Middleware(ABC):
    """Base class for all pipeline middleware"""

    @abstractmethod
    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        """Process the context and call the next middleware"""
        pass


class Pipeline:
    """Pipeline for processing WebSocket code generation requests"""

    def __init__(self):
        self.middlewares: List[Middleware] = []

    def use(self, middleware: Middleware) -> "Pipeline":
        """Add a middleware to the pipeline"""
        self.middlewares.append(middleware)
        return self

    async def execute(self, websocket: WebSocket) -> None:
        """Execute the pipeline with the given WebSocket"""
        context = PipelineContext(websocket=websocket)

        # Build the middleware chain
        async def start(ctx: PipelineContext):
            pass  # End of pipeline

        chain = start
        for middleware in reversed(self.middlewares):
            chain = self._wrap_middleware(middleware, chain)

        await chain(context)

    def _wrap_middleware(
        self,
        middleware: Middleware,
        next_func: Callable[[PipelineContext], Awaitable[None]],
    ) -> Callable[[PipelineContext], Awaitable[None]]:
        """Wrap a middleware with its next function"""

        async def wrapped(context: PipelineContext) -> None:
            await middleware.process(context, lambda: next_func(context))

        return wrapped


class WebSocketCommunicator:
    """Handles WebSocket communication with consistent error handling"""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.is_closed = False

    async def accept(self) -> None:
        """Accept the WebSocket connection"""
        await self.websocket.accept()
        print("Incoming websocket connection...")

    async def send_message(
        self,
        type: MessageType,
        value: str | None,
        variantIndex: int,
        data: Dict[str, Any] | None = None,
        eventId: str | None = None,
    ) -> None:
        """Send a message to the client with debug logging"""
        if self.is_closed:
            return

        # Print for debugging on the backend
        if type == "error":
            print(f"Error (variant {variantIndex + 1}): {value}")
        elif type == "status":
            print(f"Status (variant {variantIndex + 1}): {value}")
        elif type == "variantComplete":
            print(f"Variant {variantIndex + 1} complete")
        elif type == "variantError":
            print(f"Variant {variantIndex + 1} error: {value}")

        try:
            payload: Dict[str, Any] = {"type": type, "variantIndex": variantIndex}
            if value is not None:
                payload["value"] = value
            if data is not None:
                payload["data"] = data
            if eventId is not None:
                payload["eventId"] = eventId
            await self.websocket.send_json(payload)
        except (
            ConnectionClosedOK,
            ConnectionClosedError,
            RuntimeError,
            WebSocketDisconnect,
        ):
            print(f"WebSocket closed by client, skipping message: {type}")
            self.is_closed = True

    async def throw_error(self, message: str) -> None:
        """Send an error message and close the connection"""
        print(message)
        if not self.is_closed:
            try:
                await self.websocket.send_json({"type": "error", "value": message})
                await self.websocket.close(APP_ERROR_WEB_SOCKET_CODE)
            except (
                ConnectionClosedOK,
                ConnectionClosedError,
                RuntimeError,
                WebSocketDisconnect,
            ):
                print("WebSocket already closed by client")
            self.is_closed = True

    async def receive_params(self) -> Dict[str, Any]:
        """Receive parameters from the client"""
        try:
            params: Dict[str, Any] = await self.websocket.receive_json()
        except WebSocketDisconnect:
            self.is_closed = True
            raise
        print("Received params")
        return params

    async def close(self) -> None:
        """Close the WebSocket connection"""
        if not self.is_closed:
            try:
                await self.websocket.close()
            except (
                ConnectionClosedOK,
                ConnectionClosedError,
                RuntimeError,
                WebSocketDisconnect,
            ):
                pass  # Already closed by client
            self.is_closed = True


@dataclass
class ExtractedParams:
    stack: Stack
    input_mode: InputMode
    should_generate_images: bool
    openai_api_key: str | None
    anthropic_api_key: str | None
    gemini_api_key: str | None
    replicate_api_key: str | None
    openai_base_url: str | None
    generation_type: Literal["create", "update"]
    prompt: UserTurnInput
    history: List[PromptHistoryMessage]
    file_state: Dict[str, str] | None
    option_codes: List[str]
    should_extract_assets: bool = True
    asset_base_url: str = ""
    design_system: str | None = None


class ParameterExtractionStage:
    """Handles parameter extraction and validation from WebSocket requests"""

    def __init__(
        self,
        throw_error: Callable[[str], Coroutine[Any, Any, None]],
        asset_base_url: str = "",
    ):
        self.throw_error = throw_error
        self.asset_base_url = asset_base_url

    async def extract_and_validate(self, params: Dict[str, Any]) -> ExtractedParams:
        """Extract and validate all parameters from the request"""
        # Read the code config settings (stack) from the request.
        generated_code_config = params.get("generatedCodeConfig", "")
        if generated_code_config not in get_args(Stack):
            await self.throw_error(
                f"Invalid generated code config: {generated_code_config}"
            )
            raise ValueError(f"Invalid generated code config: {generated_code_config}")
        validated_stack = cast(Stack, generated_code_config)

        # Validate the input mode
        input_mode = params.get("inputMode")
        if input_mode not in get_args(InputMode):
            await self.throw_error(f"Invalid input mode: {input_mode}")
            raise ValueError(f"Invalid input mode: {input_mode}")
        validated_input_mode = cast(InputMode, input_mode)

        openai_api_key = self._get_from_settings_dialog_or_env(
            params, "openAiApiKey", OPENAI_API_KEY
        )

        # If neither is provided, we throw an error later only if Claude is used.
        anthropic_api_key = self._get_from_settings_dialog_or_env(
            params, "anthropicApiKey", ANTHROPIC_API_KEY
        )
        gemini_api_key = self._get_from_settings_dialog_or_env(
            params, "geminiApiKey", GEMINI_API_KEY
        )
        replicate_api_key = self._get_from_settings_dialog_or_env(
            params, "replicateApiKey", REPLICATE_API_KEY
        )

        # Base URL for OpenAI API
        openai_base_url: str | None = None
        # Disable user-specified OpenAI Base URL in prod
        if not IS_PROD:
            openai_base_url = self._get_from_settings_dialog_or_env(
                params, "openAiBaseURL", OPENAI_BASE_URL
            )
        if not openai_base_url:
            print("Using official OpenAI URL")

        # Feature preferences default to enabled for older clients.
        should_generate_images = bool(params.get("isImageGenerationEnabled", True))
        should_extract_assets = bool(params.get("isAssetExtractionEnabled", True))

        # Extract and validate generation type
        generation_type = params.get("generationType", "create")
        if generation_type not in ["create", "update"]:
            await self.throw_error(f"Invalid generation type: {generation_type}")
            raise ValueError(f"Invalid generation type: {generation_type}")
        generation_type = cast(Literal["create", "update"], generation_type)

        # Extract prompt content
        prompt: UserTurnInput = parse_prompt_content(params.get("prompt"))

        # Extract history (default to empty list)
        history: List[PromptHistoryMessage] = parse_prompt_history(
            params.get("history")
        )

        prompt = append_uploaded_asset_ids_to_prompt(prompt, self.asset_base_url)
        history = append_uploaded_asset_ids_to_history(history, self.asset_base_url)

        # Extract file state for agent edits
        raw_file_state = params.get("fileState")
        file_state: Dict[str, str] | None = None
        if isinstance(raw_file_state, dict):
            content = raw_file_state.get("content")
            if isinstance(content, str) and content.strip():
                path = raw_file_state.get("path") or "index.html"
                file_state = {"path": path, "content": content}

        raw_option_codes = params.get("optionCodes")
        option_codes: List[str] = []
        if isinstance(raw_option_codes, list):
            for entry in raw_option_codes:
                if isinstance(entry, str):
                    option_codes.append(entry)
                elif entry is None:
                    option_codes.append("")
                else:
                    option_codes.append(str(entry))

        raw_design_system = params.get("designSystem")
        design_system = (
            raw_design_system.strip()
            if isinstance(raw_design_system, str) and raw_design_system.strip()
            else None
        )

        return ExtractedParams(
            stack=validated_stack,
            input_mode=validated_input_mode,
            should_generate_images=should_generate_images,
            should_extract_assets=should_extract_assets,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            gemini_api_key=gemini_api_key,
            replicate_api_key=replicate_api_key,
            openai_base_url=openai_base_url,
            generation_type=generation_type,
            prompt=prompt,
            history=history,
            file_state=file_state,
            option_codes=option_codes,
            asset_base_url=self.asset_base_url,
            design_system=design_system,
        )

    def _get_from_settings_dialog_or_env(
        self, params: dict[str, Any], key: str, env_var: str | None
    ) -> str | None:
        """Get value from client settings or environment variable"""
        value = params.get(key)
        if value:
            print(f"Using {key} from client-side settings dialog")
            return value

        if env_var:
            print(f"Using {key} from environment variable")
            return env_var

        return None


class ModelSelectionStage:
    """Handles selection of variant models based on available API keys and generation type"""

    def __init__(self, throw_error: Callable[[str], Coroutine[Any, Any, None]]):
        self.throw_error = throw_error

    async def select_models(
        self,
        generation_type: Literal["create", "update"],
        input_mode: InputMode,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        gemini_api_key: str | None = None,
    ) -> List[Llm]:
        """Select appropriate models based on available API keys"""
        try:
            num_variants = 2 if generation_type == "update" else NUM_VARIANTS
            variant_models = self._get_variant_models(
                generation_type,
                input_mode,
                num_variants,
                openai_api_key,
                anthropic_api_key,
                gemini_api_key,
            )

            # Print the variant models (one per line)
            print("Variant models:")
            for index, model in enumerate(variant_models):
                print(f"Variant {index + 1}: {model.value}")

            return variant_models
        except Exception:
            await self.throw_error(
                "No OpenAI, Anthropic, or Gemini API key found. Please add the environment variable "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY to backend/.env or in the settings dialog. "
                "If you add it to .env, make sure to restart the backend server."
            )
            raise Exception("No API key")

    def _get_variant_models(
        self,
        generation_type: Literal["create", "update"],
        input_mode: InputMode,
        num_variants: int,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        gemini_api_key: str | None,
    ) -> List[Llm]:
        """Simple model cycling that scales with num_variants"""

        # Video mode requires Gemini - 2 variants for comparison
        if input_mode == "video":
            if not gemini_api_key:
                raise Exception(
                    "Video mode requires a Gemini API key. "
                    "Please add GEMINI_API_KEY to backend/.env or in the settings dialog"
                )
            return list(VIDEO_VARIANT_MODELS)

        # Define models based on available API keys
        if gemini_api_key and anthropic_api_key and openai_api_key:
            if input_mode == "text" and generation_type == "create":
                models = list(ALL_KEYS_MODELS_TEXT_CREATE)
            elif generation_type == "update":
                models = list(ALL_KEYS_MODELS_UPDATE)
            else:
                models = list(ALL_KEYS_MODELS_DEFAULT)
        elif gemini_api_key and anthropic_api_key:
            models = list(GEMINI_ANTHROPIC_MODELS)
        elif gemini_api_key and openai_api_key:
            models = list(GEMINI_OPENAI_MODELS)
        elif openai_api_key and anthropic_api_key:
            models = list(OPENAI_ANTHROPIC_MODELS)
        elif gemini_api_key:
            models = list(GEMINI_ONLY_MODELS)
        elif anthropic_api_key:
            models = list(ANTHROPIC_ONLY_MODELS)
        elif openai_api_key:
            models = list(OPENAI_ONLY_MODELS)
        else:
            raise Exception("No OpenAI or Anthropic key")

        # Cycle through models: [A, B] with num=5 becomes [A, B, A, B, A]
        selected_models: List[Llm] = []
        for i in range(num_variants):
            selected_models.append(models[i % len(models)])

        return selected_models


class PromptCreationStage:
    """Handles prompt assembly for code generation"""

    def __init__(self, throw_error: Callable[[str], Coroutine[Any, Any, None]]):
        self.throw_error = throw_error

    async def build_prompt_messages(
        self,
        extracted_params: ExtractedParams,
    ) -> List[ChatCompletionMessageParam]:
        """Create prompt messages"""
        try:
            prompt_messages = await build_prompt_messages(
                stack=extracted_params.stack,
                input_mode=extracted_params.input_mode,
                generation_type=extracted_params.generation_type,
                prompt=extracted_params.prompt,
                history=extracted_params.history,
                file_state=extracted_params.file_state,
                image_generation_enabled=extracted_params.should_generate_images,
                design_system=extracted_params.design_system,
            )
            print_prompt_preview(prompt_messages)

            return prompt_messages
        except Exception:
            await self.throw_error(
                "Error assembling prompt. Contact support at support@getwhimsyworks.com"
            )
            raise


class PostProcessingStage:
    """Handles post-processing after code generation completes"""

    def __init__(self):
        pass

    async def process_completions(
        self,
        completions: List[str],
        websocket: WebSocket,
    ) -> None:
        """Process completions and perform cleanup."""
        return None


class AgenticGenerationStage:
    """Handles agent tool-calling generation for each variant."""

    def __init__(
        self,
        send_message: Callable[[MessageType, str | None, int, Dict[str, Any] | None, str | None], Coroutine[Any, Any, None]],
        openai_api_key: str | None,
        openai_base_url: str | None,
        anthropic_api_key: str | None,
        gemini_api_key: str | None,
        replicate_api_key: str | None,
        should_generate_images: bool,
        file_state: Dict[str, str] | None,
        asset_base_url: str,
        option_codes: List[str] | None,
        should_extract_assets: bool = True,
        generation_id: str | None = None,
        stack: str | None = None,
        input_mode: str | None = None,
        generation_type: str | None = None,
    ):
        self.send_message = send_message
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.replicate_api_key = replicate_api_key
        self.should_generate_images = should_generate_images
        self.should_extract_assets = should_extract_assets
        self.file_state = file_state
        self.asset_base_url = asset_base_url
        self.option_codes = option_codes or []
        self.generation_id = (
            generation_id
            or f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        self.stack = stack
        self.input_mode = input_mode
        self.generation_type = generation_type

    async def process_variants(
        self,
        variant_models: List[Llm],
        prompt_messages: List[ChatCompletionMessageParam],
    ) -> Dict[int, str]:
        tasks: List[asyncio.Task[str]] = []
        for index, model in enumerate(variant_models):
            tasks.append(
                asyncio.create_task(
                    self._run_variant(index, model, prompt_messages)
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        variant_completions: Dict[int, str] = {}
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                print(f"Variant {index + 1} failed: {result}")
                continue
            if result:
                variant_completions[index] = result

        return variant_completions

    async def _run_variant(
        self,
        index: int,
        model: Llm,
        prompt_messages: List[ChatCompletionMessageParam],
    ) -> str:
        try:
            async def send_runner_message(
                type: str,
                value: str | None,
                variant_index: int,
                data: Dict[str, Any] | None,
                event_id: str | None,
            ) -> None:
                await self.send_message(
                    cast(MessageType, type),
                    value,
                    variant_index,
                    data,
                    event_id,
                )

            recorder = AgentRunRecorder(
                generation_id=self.generation_id,
                variant_index=index,
                entry_point="websocket",
                stack=self.stack,
                input_mode=self.input_mode,
                generation_type=self.generation_type,
            )
            runner = Agent(
                send_message=send_runner_message,
                variant_index=index,
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url,
                anthropic_api_key=self.anthropic_api_key,
                gemini_api_key=self.gemini_api_key,
                replicate_api_key=self.replicate_api_key,
                should_generate_images=self.should_generate_images,
                should_extract_assets=self.should_extract_assets,
                asset_base_url=self.asset_base_url,
                initial_file_state=self.file_state,
                option_codes=self.option_codes,
                recorder=recorder,
            )
            completion = await runner.run(model, prompt_messages)
            if completion:
                await self.send_message("setCode", completion, index, None, None)
            await self.send_message(
                "variantComplete",
                "Variant generation complete",
                index,
                None,
                None,
            )
            return completion
        except openai.AuthenticationError as e:
            print(f"[VARIANT {index + 1}] OpenAI Authentication failed", e)
            error_message = (
                "Incorrect OpenAI key. Please make sure your OpenAI API key is correct, "
                "or create a new OpenAI API key on your OpenAI dashboard."
                + (
                    " Alternatively, you can purchase code generation credits directly on this website."
                    if IS_PROD
                    else ""
                )
            )
            await self.send_message("variantError", error_message, index, None, None)
            return ""
        except openai.NotFoundError as e:
            print(f"[VARIANT {index + 1}] OpenAI Model not found", e)
            error_message = (
                e.message
                + ". Please make sure you have followed the instructions correctly to obtain "
                "an OpenAI key with GPT vision access: "
                "https://github.com/abi/screenshot-to-code/blob/main/Troubleshooting.md"
                + (
                    " Alternatively, you can purchase code generation credits directly on this website."
                    if IS_PROD
                    else ""
                )
            )
            await self.send_message("variantError", error_message, index, None, None)
            return ""
        except openai.RateLimitError as e:
            print(f"[VARIANT {index + 1}] OpenAI Rate limit exceeded", e)
            error_message = (
                "OpenAI error - 'You exceeded your current quota, please check your plan and billing details.'"
                + (
                    " Alternatively, you can purchase code generation credits directly on this website."
                    if IS_PROD
                    else ""
                )
            )
            await self.send_message("variantError", error_message, index, None, None)
            return ""
        except Exception as e:
            print(f"Error in variant {index + 1}: {e}")
            traceback.print_exception(type(e), e, e.__traceback__)
            await self.send_message("variantError", str(e), index, None, None)
            return ""


# --- Local-runner registration, selection, and dispatch helpers ---------------


def _ensure_configured_runner_registered() -> None:
    """Register the configured Codex runner exactly once per process.

    Only ``GENERATION_RUNNER == "codex"`` triggers registration (the config
    value is already normalized by ``config``). Registration is side-effect
    free: it never probes, launches a CLI, or reads credentials, and repeated
    calls are idempotent so imports/reloads/tests are safe.
    """
    if GENERATION_RUNNER != _LOCAL_RUNNER_NAME:
        return
    if not runner_is_registered(_LOCAL_RUNNER_NAME):
        runner_register(_LOCAL_RUNNER_NAME, CodexCliRunner())


def _select_local_runner() -> Runner | None:
    """Resolve the configured local runner, or None when not selected.

    Returns None for unset/blank/unknown ``GENERATION_RUNNER`` values so the
    legacy cloud path is preserved. Never probes or executes a CLI.
    """
    if GENERATION_RUNNER != _LOCAL_RUNNER_NAME:
        return None
    _ensure_configured_runner_registered()
    return runner_select(GENERATION_RUNNER)


@dataclass
class LocalDispatchDecision:
    """Whether the current request dispatches to a local runner.

    ``runner`` is set whenever a local runner is configured (regardless of
    request eligibility); ``eligible`` is True only when the request is within
    the v0.1 local contract (text input only, stack declared by the runner).
    """

    runner: Runner | None = None
    eligible: bool = False


def _is_local_request_eligible(
    extracted_params: ExtractedParams, runner: Runner
) -> bool:
    """v0.1 eligibility: text input only, stack declared by the runner.

    Reads only the runner's declared ``capabilities()``, which is side-effect
    free by contract — it never probes or launches a vendor CLI.
    """
    if extracted_params.input_mode != "text":
        return False
    if extracted_params.stack not in get_args(RunnerStack):
        return False
    caps = runner.capabilities()
    if extracted_params.stack not in caps.supported_stacks:
        return False
    if extracted_params.input_mode not in caps.supported_input_modes:
        return False
    return True


def _local_dispatch_decision(
    extracted_params: ExtractedParams,
) -> LocalDispatchDecision:
    """Compute the local-dispatch decision for a request."""
    runner = _select_local_runner()
    if runner is None:
        return LocalDispatchDecision()
    return LocalDispatchDecision(
        runner=runner,
        eligible=_is_local_request_eligible(extracted_params, runner),
    )


# Probe outcomes cached per runner instance (weakly referenced, so test fakes
# and module reloads never leak): the first eligible request probes exactly
# once and later requests reuse the cached outcome.
_local_runner_probe_cache: WeakKeyDictionary[Runner, ProbeResult] = (
    WeakKeyDictionary()
)

# Per-runner locks serializing the probe so concurrent eligible requests on the
# same runner still observe a single probe (the cache alone is racy: two
# coroutines can both read a miss before either writes a result). Weakly keyed
# for the same reason as the cache, and a fresh lock is created per runner
# instance, so no lock is ever shared across event loops.
_local_runner_probe_locks: WeakKeyDictionary[Runner, asyncio.Lock] = (
    WeakKeyDictionary()
)


async def _probe_runner_once(runner: Runner) -> ProbeResult:
    """Probe ``runner`` at most once per instance, caching the outcome.

    A probe that raises is treated as unavailable with a fixed safe fallback;
    the exception is never propagated into the socket path. The probe section
    is guarded by a per-runner lock (with a re-check inside it) so concurrent
    eligible requests observe a single probe.
    """
    cached = _local_runner_probe_cache.get(runner)
    if cached is not None:
        return cached

    lock = _local_runner_probe_locks.get(runner)
    if lock is None:
        lock = asyncio.Lock()
        _local_runner_probe_locks[runner] = lock

    async with lock:
        cached = _local_runner_probe_cache.get(runner)
        if cached is not None:
            return cached
        try:
            result = await runner.probe()
        except Exception:
            result = ProbeResult(available=False)
        _local_runner_probe_cache[runner] = result
        return result


def _safe_unavailable_reason(reason: str | None) -> str:
    """Bound and clean a probe's unavailable reason; fall back when unusable."""
    if reason:
        cleaned = reason.strip()
        if cleaned:
            return cleaned[:_MAX_UNAVAILABLE_REASON_LENGTH]
    return _LOCAL_RUNNER_UNAVAILABLE_FALLBACK


def _build_generation_input(
    extracted_params: ExtractedParams, request_id: str
) -> GenerationInput:
    """Build a server-owned GenerationInput for local dispatch.

    Never carries user/client filesystem paths: ``request_id`` is a
    server-created UUID, ``input_image`` is always None, and ``prior_files``
    is intentionally empty in this slice (a client-supplied file_state path is
    never forwarded).
    """
    prompt_text = (
        extracted_params.prompt.get("full_text")
        or extracted_params.prompt.get("text")
        or ""
    )
    return GenerationInput(
        request_id=request_id,
        prompt=prompt_text,
        stack=cast(RunnerStack, extracted_params.stack),
        generation_type=extracted_params.generation_type,
        design_system=extracted_params.design_system,
    )


async def _expand_complete_text_files(
    events: AsyncIterator[GenerationEvent],
) -> AsyncIterator[GenerationEvent]:
    """Expand a terminal CompleteEvent's text_files into SetCodeEvents.

    The Codex runner delivers its generated source only inside the final
    ``CompleteEvent.result.text_files``, but the frontend receives source via
    ``setCode`` messages. This wrapper emits one ``SetCodeEvent`` per file in
    deterministic (sorted) path order immediately before the terminal event,
    so the adapter's ``variantComplete`` lands after every ``setCode``. A file
    already emitted as an explicit ``SetCodeEvent`` is never emitted twice.
    """
    emitted_files: set[str] = set()
    async for event in events:
        if isinstance(event, SetCodeEvent):
            emitted_files.add(event.file_name)
            yield event
        elif isinstance(event, CompleteEvent) and event.result is not None:
            for path in sorted(event.result.text_files):
                if path not in emitted_files:
                    yield SetCodeEvent(
                        file_name=path, code=event.result.text_files[path]
                    )
            yield event
        else:
            yield event


# Pipeline Middleware Implementations


class WebSocketSetupMiddleware(Middleware):
    """Handles WebSocket setup and teardown"""

    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        # Create and setup WebSocket communicator
        context.ws_comm = WebSocketCommunicator(context.websocket)
        await context.ws_comm.accept()

        try:
            await next_func()
        finally:
            # Always close the WebSocket
            await context.ws_comm.close()


class ParameterExtractionMiddleware(Middleware):
    """Handles parameter extraction and validation"""

    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        # Receive parameters
        assert context.ws_comm is not None
        context.params = await context.ws_comm.receive_params()

        # Extract and validate
        param_extractor = ParameterExtractionStage(
            context.throw_error,
            infer_local_asset_base_url(context.websocket),
        )
        context.extracted_params = await param_extractor.extract_and_validate(
            context.params
        )

        # Log what we're generating
        print(
            f"Generating {context.extracted_params.stack} code in {context.extracted_params.input_mode} mode"
        )

        await next_func()


class StatusBroadcastMiddleware(Middleware):
    """Sends initial status messages to all variants"""

    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        # A configured local runner announces exactly one variant (v0.1 local
        # dispatch streams only variant 0). Unsupported local requests fail
        # closed downstream with a variantError under the same single-variant
        # contract. The legacy multi-variant announcement is untouched when no
        # local runner is selected.
        assert context.extracted_params is not None
        if _local_dispatch_decision(context.extracted_params).runner is not None:
            await context.send_message("variantCount", "1", 0)
            await context.send_message("status", "Generating code...", 0)
            await next_func()
            return

        # Determine variant count based on input mode and generation type.
        # Edit/update flows use two variants to keep latency and cost down.
        is_video_mode = context.extracted_params.input_mode == "video"
        is_update = context.extracted_params.generation_type == "update"
        num_variants = (
            NUM_VARIANTS_VIDEO if is_video_mode else 2 if is_update else NUM_VARIANTS
        )

        # Tell frontend how many variants we're using
        await context.send_message("variantCount", str(num_variants), 0)

        for i in range(num_variants):
            await context.send_message("status", "Generating code...", i)

        await next_func()


class PromptCreationMiddleware(Middleware):
    """Handles prompt creation"""

    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        # A configured local runner bypasses legacy prompt construction: the
        # local path builds its own GenerationInput and never consumes
        # context.prompt_messages, so assembling the legacy prompt here is
        # wasted work and can fail closed (closing the socket) before the
        # local dispatch emits its own client-safe outcome.
        assert context.extracted_params is not None
        if _local_dispatch_decision(context.extracted_params).runner is not None:
            await next_func()
            return

        prompt_creator = PromptCreationStage(context.throw_error)
        context.prompt_messages = await prompt_creator.build_prompt_messages(
            context.extracted_params,
        )
        await next_func()


class CodeGenerationMiddleware(Middleware):
    """Handles the main code generation logic"""

    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        assert context.extracted_params is not None

        # Configured local runner: the request is handled entirely by the
        # local path, before any legacy model selection. Ineligible or
        # unavailable requests fail closed with a client-safe variantError and
        # never fall back to the cloud; the local path never reads API keys,
        # builds an Agent, or calls the next middleware.
        local_decision = _local_dispatch_decision(context.extracted_params)
        if local_decision.runner is not None:
            await self._process_local_generation(context, local_decision)
            return

        try:
            # Select models (handles video mode internally)
            model_selector = ModelSelectionStage(context.throw_error)
            context.variant_models = await model_selector.select_models(
                generation_type=context.extracted_params.generation_type,
                input_mode=context.extracted_params.input_mode,
                openai_api_key=context.extracted_params.openai_api_key,
                anthropic_api_key=context.extracted_params.anthropic_api_key,
                gemini_api_key=context.extracted_params.gemini_api_key,
            )
            if IS_DEBUG_ENABLED:
                await context.send_message(
                    "variantModels",
                    None,
                    0,
                    {"models": [model.value for model in context.variant_models]},
                    None,
                )

            generation_stage = AgenticGenerationStage(
                send_message=context.send_message,
                openai_api_key=context.extracted_params.openai_api_key,
                openai_base_url=context.extracted_params.openai_base_url,
                anthropic_api_key=context.extracted_params.anthropic_api_key,
                gemini_api_key=context.extracted_params.gemini_api_key,
                replicate_api_key=context.extracted_params.replicate_api_key,
                should_generate_images=context.extracted_params.should_generate_images,
                should_extract_assets=context.extracted_params.should_extract_assets,
                file_state=context.extracted_params.file_state,
                asset_base_url=context.extracted_params.asset_base_url,
                option_codes=context.extracted_params.option_codes,
                stack=str(context.extracted_params.stack),
                input_mode=str(context.extracted_params.input_mode),
                generation_type=context.extracted_params.generation_type,
            )

            context.variant_completions = await generation_stage.process_variants(
                variant_models=context.variant_models,
                prompt_messages=context.prompt_messages,
            )

            # Check if all variants failed
            if len(context.variant_completions) == 0:
                await context.throw_error(
                    "Error generating code. Please contact support."
                )
                return  # Don't continue the pipeline

            # Convert to list format
            context.completions = []
            for i in range(len(context.variant_models)):
                if i in context.variant_completions:
                    context.completions.append(context.variant_completions[i])
                else:
                    context.completions.append("")

        except Exception as e:
            print(f"[GENERATE_CODE] Unexpected error: {e}")
            await context.throw_error(f"An unexpected error occurred: {str(e)}")
            return  # Don't continue the pipeline

        await next_func()

    async def _process_local_generation(
        self,
        context: PipelineContext,
        decision: LocalDispatchDecision,
    ) -> None:
        """Run an eligible local generation for variant 0, or fail closed.

        The runner is probed at most once per instance (cached); an
        unsupported or unavailable request emits exactly one client-safe
        ``variantError`` and returns without generating, without cloud
        fallback, and without postprocessing. Unexpected transport errors from
        the sender are deliberately not caught here, so they propagate instead
        of being masked.
        """
        assert context.extracted_params is not None
        assert decision.runner is not None
        runner = decision.runner

        if not decision.eligible:
            await context.send_message("variantError", _LOCAL_UNSUPPORTED_MESSAGE, 0)
            return

        try:
            generation_input = _build_generation_input(
                context.extracted_params, f"req_{uuid.uuid4().hex}"
            )
        except Exception:
            await context.send_message("variantError", _LOCAL_UNSUPPORTED_MESSAGE, 0)
            return

        probe = await _probe_runner_once(runner)
        if not probe.available:
            await context.send_message(
                "variantError", _safe_unavailable_reason(probe.unavailable_reason), 0
            )
            return

        events = _expand_complete_text_files(runner.generate(generation_input))
        adapter = LocalRunnerWebSocketAdapter(
            send_message=context.send_message, variant_index=0
        )
        await adapter.stream(events)


class PostProcessingMiddleware(Middleware):
    """Handles post-processing and logging"""

    async def process(
        self, context: PipelineContext, next_func: Callable[[], Awaitable[None]]
    ) -> None:
        post_processor = PostProcessingStage()
        await post_processor.process_completions(
            context.completions, context.websocket
        )

        await next_func()


@router.websocket("/generate-code")
async def stream_code(websocket: WebSocket):
    """Handle WebSocket code generation requests using a pipeline pattern"""
    pipeline = Pipeline()

    # Configure the pipeline
    pipeline.use(WebSocketSetupMiddleware())
    pipeline.use(ParameterExtractionMiddleware())
    pipeline.use(StatusBroadcastMiddleware())
    pipeline.use(PromptCreationMiddleware())
    pipeline.use(CodeGenerationMiddleware())
    pipeline.use(PostProcessingMiddleware())

    # Execute the pipeline
    await pipeline.execute(websocket)

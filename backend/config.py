import os

NUM_VARIANTS = 4
NUM_VARIANTS_VIDEO = 2


def parse_truthy_env(name: str) -> bool:
    """Read a boolean env var strictly.

    Only the explicit truthy spellings ``1``/``true``/``yes``/``on`` (case-
    and whitespace-insensitive) count as enabled; everything else — including
    the string ``"false"`` — is disabled. This avoids the classic
    ``bool("false") is True`` footgun.
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

# LLM-related
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)

# Image generation (optional)
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY", None)

# Debugging-related
IS_DEBUG_ENABLED = bool(os.environ.get("IS_DEBUG_ENABLED", False))
DEBUG_DIR = os.environ.get("DEBUG_DIR", "")

# When enabled, every LLM request is written to run_logs/prompt_reports as a
# JSON report viewable at /evals/prompt-reports.
# Hard per-generation spend ceiling; a run that would continue past this is
# aborted. Applies per variant/eval run. Unpriced models are not bounded.
GENERATION_MAX_COST_USD = 3.0

PROMPT_REPORTS_ENABLED = os.environ.get(
    "PROMPT_REPORTS_ENABLED", ""
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_ASSET_DIR = os.environ.get(
    "LOCAL_ASSET_DIR", os.path.join(os.path.dirname(__file__), "local_assets")
)
# Base URL the backend serves /local-assets from. The live (websocket) path
# infers this per-request; the evals path has no request, so it uses this.
LOCAL_ASSET_BASE_URL = os.environ.get("LOCAL_ASSET_BASE_URL", "http://127.0.0.1:7001")

# Set to True when running in production (on the hosted version)
# Used as a feature flag to enable or disable certain features
IS_PROD = os.environ.get("IS_PROD", False)

# Local coding-agent runner (e.g. a future Codex subprocess runner), selected
# by the GENERATION_RUNNER env var. Unset/blank = disabled and unconfigured
# (the safe default). When set, the value is a runner name looked up in the
# runner registry; an unknown name selects nothing and yields a safe
# "unavailable" reason at the capability endpoint. Setting this never reads
# or requires an API key, and never launches a vendor CLI.
GENERATION_RUNNER = os.environ.get("GENERATION_RUNNER", "").strip() or None

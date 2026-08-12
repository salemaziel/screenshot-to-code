"""Configured local-runner state for the ``/api/capabilities`` endpoint.

This module is deliberately probe-free: it only reads configuration and the
runner registry. Determining whether a runner actually works (launching a
vendor CLI) is the runner's ``probe()`` job and is never done here, so calling
this never executes a binary.
"""

from config import GENERATION_RUNNER
from runners.models import LocalRunnerState
from runners.registry import select

_DISABLED_REASON = "local runner disabled"
_MISSING_REASON = "configured runner is not registered"


def local_runner_state() -> LocalRunnerState:
    """Current configured local-runner state, safe to expose over the API.

    Returns ``available=False`` with a short reason whenever the runner is
    unconfigured or not registered in the registry. When a runner is
    registered, its declared capabilities are reported without probing it, so
    no vendor CLI (or OAuth flow) is ever launched by this function.
    """
    if not GENERATION_RUNNER:
        return LocalRunnerState(
            enabled=False,
            name=None,
            available=False,
            unavailable_reason=_DISABLED_REASON,
        )
    runner = select(GENERATION_RUNNER)
    if runner is None:
        return LocalRunnerState(
            enabled=True,
            name=GENERATION_RUNNER,
            available=False,
            unavailable_reason=_MISSING_REASON,
        )
    caps = runner.capabilities()
    return LocalRunnerState(
        enabled=True,
        name=GENERATION_RUNNER,
        available=caps.available,
        unavailable_reason=caps.unavailable_reason,
        supported_stacks=caps.supported_stacks,
        supported_input_modes=caps.supported_input_modes,
    )

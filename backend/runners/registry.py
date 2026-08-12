"""Registry of local coding-agent runners.

Starts empty: no runner is registered by default, and importing this module has
no side effects — a runner is only ever added via an explicit ``register``
call. Enumerating capabilities never probes a vendor CLI; it only reads each
runner's declared ``capabilities()`` report (see ``Runner`` in ``base``).
"""

from typing import Dict, List, Optional

from runners.base import Runner
from runners.models import RunnerCapabilities

_registry: Dict[str, Runner] = {}


def register(name: str, runner: Runner) -> None:
    """Register ``runner`` under ``name``.

    Raises ``ValueError`` for a blank/whitespace-only name or a name that is
    already registered, so failures are predictable and easy to test.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("runner name must not be blank")
    if cleaned in _registry:
        raise ValueError(f"runner {cleaned!r} is already registered")
    _registry[cleaned] = runner


def unregister(name: str) -> None:
    """Remove the runner registered under ``name``.

    Unregistering a name that was never registered is a no-op.
    """
    _registry.pop(name, None)


def get(name: str) -> Optional[Runner]:
    """Return the runner registered under ``name``, or None when missing."""
    return _registry.get(name)


def select(name: Optional[str]) -> Optional[Runner]:
    """Safely resolve a configured runner name to a registered runner.

    Returns None for an unset/blank name or a name that was never registered,
    and never raises. This is the single selection helper used to turn the
    configured ``GENERATION_RUNNER`` value into a runner (or into a safe
    "unavailable" outcome).
    """
    if name is None:
        return None
    cleaned = name.strip()
    if not cleaned:
        return None
    return _registry.get(cleaned)


def is_registered(name: str) -> bool:
    """Whether a runner is registered under ``name``."""
    return name in _registry


def names() -> List[str]:
    """Registered runner names, sorted for deterministic output."""
    return sorted(_registry.keys())


def list_capabilities() -> List[RunnerCapabilities]:
    """Declared capability reports for every registered runner.

    Safe by contract: this only calls each runner's ``capabilities()``, which
    must never launch or probe a vendor CLI. Order is deterministic (sorted by
    registration name).
    """
    return [_registry[name].capabilities() for name in sorted(_registry)]

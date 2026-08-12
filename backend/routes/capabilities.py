from fastapi import APIRouter
from pydantic import BaseModel

from preview_screenshot import probe_screenshot_preview
from runners.capabilities import local_runner_state
from runners.models import LocalRunnerState

router = APIRouter()


class Capabilities(BaseModel):
    screenshot_preview: bool
    local_runner: LocalRunnerState


@router.get("/api/capabilities", response_model=Capabilities)
async def get_capabilities() -> Capabilities:
    """Backend feature availability for the frontend to reflect in settings.

    The local-runner state is computed from configuration and the runner
    registry's declared ``capabilities()`` only; it never invokes ``probe()``
    or ``generate()`` and therefore never launches a vendor CLI.
    """
    return Capabilities(
        screenshot_preview=await probe_screenshot_preview(),
        local_runner=local_runner_state(),
    )

"""Prompt management endpoints."""

from fastapi import APIRouter, Depends

from pydantic import BaseModel

from api.dependencies import get_prompt_registry
from src.models.prompt import PromptVersion
from src.prompt_service.registry import PromptRegistry

router = APIRouter()


class RegisterPromptRequest(BaseModel):
    version_id: str
    template: str
    grade: str = ""
    variant: str = "standard"


@router.post("/prompts/register")
async def register_prompt(
    request: RegisterPromptRequest,
    registry: PromptRegistry = Depends(get_prompt_registry),
):
    """Register a new prompt version."""
    version = PromptVersion(
        version_id=request.version_id,
        template=request.template,
        grade=request.grade,
        variant=request.variant,
    )
    await registry.register(version)
    return {"status": "registered", "version_id": request.version_id}


@router.get("/prompts/active")
async def list_active_prompts(
    grade: str = "",
    variant: str = "",
    registry: PromptRegistry = Depends(get_prompt_registry),
):
    """List active prompt versions."""
    versions = await registry.list_active(grade=grade, variant=variant)
    return {"prompts": [v.model_dump() for v in versions]}

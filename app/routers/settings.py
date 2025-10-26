from __future__ import annotations

from fastapi import APIRouter, Depends, Form

from ..dependencies import get_app_settings
from ..models import RunConfig

router = APIRouter()


@router.get("/settings/status")
def get_settings_status(settings=Depends(get_app_settings)):
    return {"keys": settings.keys_status()}


class _SetReq(RunConfig.model_construct().__class__):
    pass


@router.post("/settings")
async def post_settings(
    gemini_api_key: str | None = Form(default=None),
    openai_api_key: str | None = Form(default=None),
    persist: bool = Form(default=False),
    settings=Depends(get_app_settings),
):
    status = settings.set_keys(gemini=gemini_api_key, openai=openai_api_key, persist=persist)
    return {"ok": True, "keys": status, "persisted": persist}


__all__ = ["router"]

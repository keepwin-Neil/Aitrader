"""
LLM 管理: provider 密钥 (API key / CLI 模式 / 模型名)。
本机单用户运行, 不做登录鉴权。
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import core_engine as _eng
from core_store import SYSTEM_ACTOR

router = APIRouter()


class LLMKeyInput(BaseModel):
    api_key: Optional[str] = ""
    use_cli: Optional[bool] = None
    cli_command: Optional[str] = None
    model_name: Optional[str] = None


# ── Provider 密钥 ────────────────────────────────────────────────────────

@router.get("/llm/keys")
def list_llm_keys():
    items = _eng.list_llm_key_settings()
    return {"success": True, "count": len(items), "items": items}


@router.post("/llm/keys/{provider}")
def save_llm_key(provider: str, payload: LLMKeyInput):
    try:
        item = _eng.save_llm_key_setting(
            provider,
            payload.api_key or "",
            actor_user=SYSTEM_ACTOR,
            use_cli=payload.use_cli,
            cli_command=payload.cli_command,
            model_name=payload.model_name,
        )
        return {"success": True, "item": item}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/llm/keys/{provider}")
def clear_llm_key(provider: str):
    try:
        item = _eng.clear_llm_key_setting(provider)
        return {"success": True, "item": item}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

"""
策略组合: 保存 / 读取一组策略, 供组合优化等研究工具引用。
本机单用户运行, 不区分归属。
"""
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import core_store as _store

router = APIRouter()


class StrategyComboCreateInput(BaseModel):
    name: str
    strategies: list[str]
    llm_owners: Optional[list[str]] = None
    note: Optional[str] = None


class StrategyComboUpdateInput(BaseModel):
    name: Optional[str] = None
    strategies: Optional[list[str]] = None
    llm_owners: Optional[list[str]] = None
    note: Optional[str] = None


@router.get("/strategy-combos")
def list_strategy_combos():
    items = _store.list_strategy_combos()
    return {"success": True, "count": len(items), "items": items}


@router.get("/strategy-combos/{combo_id}")
def get_strategy_combo(combo_id: int):
    item = _store.get_strategy_combo(combo_id)
    if not item:
        raise HTTPException(status_code=404, detail="组合不存在")
    return {"success": True, "combo": item}


@router.post("/strategy-combos")
def create_strategy_combo(payload: StrategyComboCreateInput):
    try:
        item = _store.create_strategy_combo(
            name=payload.name,
            strategies=payload.strategies or [],
            llm_owners=payload.llm_owners or [],
            note=payload.note,
        )
        return {"success": True, "combo": item}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/strategy-combos/{combo_id}")
def update_strategy_combo(combo_id: int, payload: StrategyComboUpdateInput):
    changes = payload.model_dump(exclude_none=True) if hasattr(payload, "model_dump") \
        else payload.dict(exclude_none=True)
    try:
        return {"success": True, "combo": _store.update_strategy_combo(combo_id, **changes)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/strategy-combos/{combo_id}")
def delete_strategy_combo(combo_id: int):
    if not _store.delete_strategy_combo(combo_id):
        raise HTTPException(status_code=404, detail="组合不存在")
    return {"success": True}

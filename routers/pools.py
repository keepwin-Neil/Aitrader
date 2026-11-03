"""
Pools router: /pools, /pools/{name}, /pools/hot-stocks/*
"""
from fastapi import APIRouter, HTTPException

import main as _m

router = APIRouter()


@router.get("/pools")
def get_pools():
    """获取所有股票池"""
    return _m._load_pools()


@router.post("/pools")
def save_pool(body: dict):
    """创建或更新股票池
    body: {"name": "US Tech", "market": "US", "symbols": ["AAPL", "NVDA"]}

    系统托管的热门池（HOT_POOL_NAME）不允许通过此端点修改——只能通过
    /pools/hot-stocks/refresh 由系统自动刷新。这样避免普通用户的 chip UI
    last-write-wins 覆盖掉共享的热门池。
    """
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "股票池名称不能为空")
    # 系统托管池只读：防止普通用户通过 chip UI 覆盖共享的热门池数据。
    if name == getattr(_m, "HOT_POOL_NAME", "🔥 US Hot Stocks"):
        raise HTTPException(403, "系统托管的热门池不可手动修改，请使用热门股刷新")
    market = body.get("market", "US").upper()
    symbols = body.get("symbols", [])
    if not symbols:
        raise HTTPException(400, "股票池不能为空")
    symbols = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))

    def _apply(pools):
        prev = pools.get(name) or {}
        if not isinstance(prev, dict):   # 兼容历史上手改出来的 list 形态
            prev = {"symbols": prev if isinstance(prev, list) else []}
    # 保留池上的其它字段 (note 等); symbols 顺序 = 用户自定义顺序, 原样存
        _RESERVED = ("name", "market", "symbols", "pinned")
        entry = {k: v for k, v in prev.items() if k not in _RESERVED}
        # 改名等场景会把整份池带过来 → body 里的其它字段 (note 等) 也要收下,
        # 否则 prev 为空时它们会被静默丢掉
        entry.update({k: v for k, v in body.items() if k not in _RESERVED})
        entry["market"] = market
        entry["symbols"] = symbols
        # pinned: 置顶列表 (右键 📌), 只能是池内 symbol; 传了才改, 不传保留 (去掉已不在池里的)
        pinned = body.get("pinned")
        if pinned is None:
            pinned = prev.get("pinned") or []
        if not isinstance(pinned, list):
            raise HTTPException(400, "pinned 必须是列表")
        pinned = [str(s).strip().upper() for s in pinned if str(s).strip()]
        pinned = [s for s in dict.fromkeys(pinned) if s in symbols]
        if pinned:
            entry["pinned"] = pinned
        pools[name] = entry
        return entry

    saved = _m._mutate_pools(_apply)
    return {"ok": True, "name": name, "pool": saved}


@router.delete("/pools/{name}")
def delete_pool(name: str):
    """删除股票池——系统托管的热门池不可删除。"""
    if name == getattr(_m, "HOT_POOL_NAME", "🔥 US Hot Stocks"):
        raise HTTPException(403, "系统托管的热门池不可删除")
    def _apply(pools):
        if name not in pools:
            raise HTTPException(404, f"股票池 '{name}' 不存在")
        del pools[name]
    _m._mutate_pools(_apply)
    return {"ok": True, "deleted": name}


@router.post("/pools/hot-stocks/refresh")
def refresh_hot_stocks():
    """手动刷新热门美股股票池"""
    data = _m._update_hot_pool()
    if data is None:
        raise HTTPException(500, "热门股更新失败，请稍后重试")
    return {"ok": True, "count": len(data["symbols"]), "symbols": data["symbols"],
            "updated_at": data["updated_at"]}


@router.get("/pools/hot-stocks/details")
def get_hot_stocks_details():
    """获取热门股详情（名称/涨跌幅/成交量/市值）"""
    cache = _m._load_hot_cache()
    if not cache:
        return {"details": {}, "updated_at": None}
    return cache

"""
Eval Chain Router — LLM 策略评估链 API 端点
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import threading
import time
import logging

import main as _m

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

# 全局进度追踪
# 容器用 LockedDict, 登记走 _m.register_job —— 这是 8 张后台任务表里最后一张
# 还在裸赋值的。裸赋值的后果不是锁(runner 自己把 progress 建成了 LockedDict),
# 而是**从不回收**: 每跑一次评估链就往里塞一条, 条目里挂着全部 reviews
# (每条带 LLM 打分 + 完整回测指标, limit 默认 500) + summary + actions, 永不释放。
# 另外 7 张表都有 JOB_REGISTRY_MAX 的上限。
_chain_jobs = _m.LockedDict()  # {job_id: EvalChainRunner.progress}


@router.post("/eval-chain/run")
def start_eval_chain(body: dict):
    """启动完整评估链（后台线程）。返回 job_id 用于轮询进度。"""
    from eval_chain import EvalChainRunner

    config = {
        "pool": body.get("pool", "MYPOOL"),
        "interval": body.get("interval", "1d"),
        "limit": body.get("limit", 500),
        "provider": body.get("provider", "claude"),
        # 不填就交给 EvalChainRunner 按 provider 取默认 (统一出处: LLM_PROVIDER_DEFAULTS)
        "model": body.get("model") or "",
        "dry_run": body.get("dry_run", True),
        "force_eval": body.get("force_eval", False),
        "start_stage": body.get("start_stage", "evaluate"),
        "signal_type": body.get("signal_type", "combined"),
    }

    runner = EvalChainRunner(config)
    job_id = runner.round_id
    # 传的是 runner 自己的 LockedDict 本体, register_job 会原样登记(不重新包),
    # 这样 runner 后续的写才能被轮询读到。
    _m.register_job(_chain_jobs, job_id, runner.progress)

    def _run():
        try:
            runner.run()
        except Exception as e:
            runner.progress["status"] = "error"
            runner.progress["error"] = str(e)[:500]
        # 不用再重新登记: 上面登记的就是同一个对象, runner 的写已经生效。

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "config": config,
        "message": f"评估链已启动: {config['pool']} | {config['provider']} | dry_run={config['dry_run']}",
    }


@router.get("/eval-chain/status/{job_id}")
async def get_chain_status(job_id: str):
    """轮询评估链进度。(async: 别跟重活抢线程池)"""
    prog = _chain_jobs.get(job_id)
    if not prog:
        raise HTTPException(404, f"Job '{job_id}' not found")
    # 先脱离写侧: runner 在后台线程里持续改这个 dict
    prog = _m.snapshot_of(prog)

    # 返回精简版（不含 reviews 完整数据，减少传输量）
    return _m.job_json_response({
        "job_id": job_id,
        "stage": prog.get("stage", "unknown"),
        "status": prog.get("status", "unknown"),
        "done": prog.get("done", 0),
        "total": prog.get("total", 0),
        "current": prog.get("current", ""),
        # 这三个是 EvalChainRunner 早就在写、前端也早就在读的字段, 之前被这个
        # 白名单挡掉了 —— 所以评估链的面板永远没有单位后缀, 也不显示并发那一行
        "unit": prog.get("unit", ""),
        "workers": prog.get("workers", 0),
        "running": list(prog.get("running") or []),
        "sub_stage": prog.get("sub_stage", ""),
        "error": prog.get("error"),
        "summary": prog.get("summary", {}),
        "actions": prog.get("actions", {}),
        "report_path": prog.get("report_path", ""),
    })


@router.get("/eval-chain/results/{job_id}")
def get_chain_results(job_id: str):
    """获取完整评估结果（含全部 reviews）。"""
    prog = _chain_jobs.get(job_id)
    if not prog:
        raise HTTPException(404, f"Job '{job_id}' not found")
    prog = _m.snapshot_of(prog)
    if prog.get("status") != "done":
        return {"status": prog.get("status"), "message": "Chain 尚未完成"}
    return _m.job_json_response({
        "job_id": job_id,
        "status": "done",
        # raw_response 是整段 LLM 原文, 前端只挑 7 个字段, 而它已经另存进 DB
        # 和 llm_reviews/*.json —— HTTP 这一份(N × 整段原文, N 默认 500)纯浪费。
        "reviews": [{k: v for k, v in (r or {}).items() if k != "raw_response"}
                    if isinstance(r, dict) else r
                    for r in (prog.get("reviews") or [])],
        "summary": prog.get("summary", {}),
        "actions": prog.get("actions", {}),
        "report_path": prog.get("report_path", ""),
    })


@router.post("/eval-chain/analyze-single/{strategy_name}")
def analyze_single_strategy(strategy_name: str, body: dict = None):
    """同步：对单个策略调用 LLM 分析。需要传入该策略的 metrics 数据。"""
    body = body or {}
    from llm_evaluator import evaluate_strategy
    from core_store import create_llm_review

    metrics = body.get("metrics", {})
    if not metrics:
        raise HTTPException(400, "缺少 metrics 数据")

    provider = body.get("provider", "claude")
    from core_engine import default_model_for
    model = body.get("model") or default_model_for(provider)
    pool_name = body.get("pool", "")
    signal_type = body.get("signal_type", "combined")

    wh = _m._load_warehouse()
    fav_set = set(wh.get("favourites", []))
    active_set = set(wh.get("active", []))
    status = "active" if strategy_name in active_set else "archived"

    result = evaluate_strategy(
        strategy_name=strategy_name,
        metrics=metrics,
        warehouse_status=status,
        is_favourite=strategy_name in fav_set,
        pool_name=pool_name,
        stock_count=metrics.get("stock_count", 0),
        interval=body.get("interval", "1d"),
        rank=metrics.get("rank", 0),
        total_strategies=metrics.get("total_strategies", 0),
        signal_type=signal_type,
        provider=provider,
        model=model,
    )

    # 写入数据库
    round_id = f"single_{int(time.time())}"
    try:
        create_llm_review(
            strategy_name=strategy_name,
            review_round=round_id,
            signal_type=signal_type,
            eval_pool=pool_name,
            provider=provider,
            model=model,
            verdict=result["verdict"],
            confidence=result["confidence"],
            overall_score=result.get("overall_score", 0),
            reasoning=result.get("reasoning"),
            risk_warnings=result.get("risk_warnings"),
            suggestions=result.get("suggestions"),
            input_metrics=metrics,
            raw_response=result.get("raw_response"),
            prompt_hash=result.get("prompt_hash"),
            response_hash=result.get("response_hash"),
            input_tokens=result.get("token_usage", {}).get("input_tokens", 0),
            output_tokens=result.get("token_usage", {}).get("output_tokens", 0),
            duration_ms=result.get("duration_ms", 0),
        )
    except Exception as e:
        logger.warning(f"[EvalChain] DB write failed: {e}")

    return result


@router.get("/eval-chain/reviews")
def list_reviews(strategy_name: Optional[str] = None,
                 review_round: Optional[str] = None,
                 verdict: Optional[str] = None,
                 limit: int = 100):
    """查询 LLM review 历史记录。"""
    from core_store import list_llm_reviews
    return {"reviews": list_llm_reviews(
        strategy_name=strategy_name,
        review_round=review_round,
        verdict=verdict,
        limit=limit,
    )}


@router.get("/eval-chain/reviews/latest/{strategy_name}")
def get_latest_strategy_review(strategy_name: str):
    """获取单策略最新 LLM review。"""
    from core_store import get_latest_review
    review = get_latest_review(strategy_name)
    if not review:
        raise HTTPException(404, f"No review found for '{strategy_name}'")
    return review


@router.get("/eval-chain/summary")
def get_reviews_summary(review_round: Optional[str] = None):
    """汇总统计：verdict 分布、平均 confidence/score、token 用量。"""
    from core_store import list_llm_reviews

    reviews = list_llm_reviews(review_round=review_round, limit=500)
    if not reviews:
        return {"total": 0, "message": "No reviews found"}

    activate = sum(1 for r in reviews if r["verdict"] == "ACTIVATE")
    archive = sum(1 for r in reviews if r["verdict"] == "ARCHIVE")
    watch = sum(1 for r in reviews if r["verdict"] == "WATCH")
    avg_conf = round(sum(r["confidence"] for r in reviews) / len(reviews), 1)
    avg_score = round(sum(r.get("overall_score", 0) for r in reviews) / len(reviews), 1)
    total_in = sum(r.get("input_tokens", 0) for r in reviews)
    total_out = sum(r.get("output_tokens", 0) for r in reviews)

    # 按 round 分组
    rounds = {}
    for r in reviews:
        rd = r["review_round"]
        if rd not in rounds:
            rounds[rd] = {"round": rd, "count": 0, "created": r["created_at"]}
        rounds[rd]["count"] += 1

    return {
        "total": len(reviews),
        "activate_count": activate,
        "archive_count": archive,
        "watch_count": watch,
        "avg_confidence": avg_conf,
        "avg_score": avg_score,
        "total_tokens": {"input": total_in, "output": total_out, "total": total_in + total_out},
        "rounds": sorted(rounds.values(), key=lambda x: x["created"], reverse=True)[:10],
    }


@router.get("/eval-chain/providers")
def list_llm_providers():
    """列出已配置的 LLM provider（不暴露 key），前端用于 LLM 选择弹窗。"""
    import core_engine as _eng
    items = _eng.list_llm_key_settings()
    return {"providers": [
        {
            "owner": it["owner"],
            "provider": it["provider"],
            "model_name": it.get("model_name"),
            "configured": it.get("configured", False),
            "config_source": it.get("config_source"),  # "saved" | "env" | "cli"
            "use_cli": it.get("use_cli", False),
        }
        for it in items if it.get("configured")
    ]}


@router.get("/eval-chain/opt-history")
def list_opt_history(pool_name: Optional[str] = None, signal_type: Optional[str] = None,
                     limit: int = 30):
    """查询策略优化历史记录（不含 per_stock 大数据，轻量返回）。"""
    from core_store import list_optimization_runs
    return {"runs": list_optimization_runs(pool_name=pool_name, signal_type=signal_type, limit=limit)}


@router.get("/eval-chain/opt-history/{run_id}")
def get_opt_history_detail(run_id: str):
    """获取单次优化的完整记录（含 per_stock 明细）。"""
    from core_store import get_optimization_run
    run = get_optimization_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run

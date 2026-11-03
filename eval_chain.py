"""
Eval Chain Orchestrator — 连接 Evaluate → LLM Analysis → Action → Report
支持完整链路运行，也支持从任意 stage 启动
"""
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("uvicorn.error")

from locked_dict import LockedDict

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path(__file__).parent / "strategy_reports"
REPORTS_DIR.mkdir(exist_ok=True)
REVIEWS_DIR = Path(__file__).parent / "llm_reviews"
REVIEWS_DIR.mkdir(exist_ok=True)


class EvalChainRunner:
    """评估链编排器 — 串联各 pipeline stage"""

    def __init__(self, config: dict):
        self.pool = config.get("pool", "MYPOOL")
        self.interval = config.get("interval", "1d")
        self.limit = config.get("limit", 500)
        self.provider = config.get("provider", "claude")
        # 默认模型名统一从 core_engine.LLM_PROVIDER_DEFAULTS 取, 不在本文件写死
        from core_engine import default_model_for
        self.model = config.get("model") or default_model_for(self.provider)
        self.dry_run = config.get("dry_run", True)
        self.force_eval = config.get("force_eval", False)
        self.start_stage = config.get("start_stage", "evaluate")  # evaluate / llm_analyze / apply
        self.signal_type = config.get("signal_type", "combined")

        self.round_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_file = CACHE_DIR / f"eval_chain_{self.round_id}.json"

        # 进度追踪
        # LockedDict: 这个 dict 是后台线程(_run)在写、HTTP 轮询在读的共享状态。
        # 写点有 26 处(self.progress[...] = / .update), 读侧要遍历 summary/actions/
        # reviews —— 撞上就 RuntimeError: dictionary changed size during iteration
        # → HTTP 500 → 前端判定评估链失败并停止跟踪, 而后台其实还在跑。
        # 容器自己持锁, 写点一行不用改。
        self.progress = LockedDict({
            "stage": "init",
            "done": 0,
            "total": 0,
            "current": "",
            "status": "running",
            "error": None,
            "round_id": self.round_id,
            "reviews": [],
            "summary": {},
            "report_path": "",
            "actions": {},
        })

    def run(self) -> dict:
        """同步执行完整链路。返回 progress 状态。"""
        try:
            # Stage 1: Evaluate
            ranking = []
            if self.start_stage == "evaluate":
                self.progress["stage"] = "evaluate"
                ranking = self._stage_evaluate()
                self._save_checkpoint("evaluate_done", {"ranking": ranking})
            else:
                # 从 checkpoint 恢复
                cp = self._load_checkpoint()
                ranking = cp.get("ranking", [])

            if not ranking:
                self.progress["status"] = "error"
                self.progress["error"] = "评估阶段无结果"
                return self.progress

            # Stage 2: LLM Analyze
            self.progress["stage"] = "llm_analyze"
            self.progress["total"] = len(ranking)
            self.progress["done"] = 0
            reviews, summary = self._stage_llm_analyze(ranking)
            self._save_checkpoint("llm_done", {"ranking": ranking, "reviews": reviews, "summary": summary})

            # Stage 3: Apply Actions
            self.progress["stage"] = "apply_actions"
            actions = self._stage_apply_actions(reviews)
            self.progress["actions"] = actions

            # Stage 4: Report
            self.progress["stage"] = "report"
            report_path = self._stage_report(reviews, summary)
            self.progress["report_path"] = report_path

            # 顺序要紧: /eval-chain/results 是先看 status=="done" 再读
            # reviews/summary 的。如果先把 status 翻成 done, 轮询正好落在这两行
            # 之间就会拿到"已完成但 reviews=[] summary={}"的空结果, 前端直接
            # 渲染成一份空报告。所以数据先落, status 最后落, 而且一次 update
            # 写完 —— LockedDict 保证单次 update 是原子的。
            self.progress["reviews"] = reviews
            self.progress["summary"] = summary
            self.progress.update({"stage": "done", "status": "done"})

        except Exception as e:
            logger.error(f"[EvalChain] 运行失败: {e}", exc_info=True)
            self.progress["status"] = "error"
            self.progress["error"] = str(e)[:500]

        return self.progress

    # ─── Stage 1: Evaluate ───────────────────────────────────

    def _stage_evaluate(self) -> list:
        """复用 /strategies/evaluate 的逻辑获取策略排名。"""
        import pickle
        import subprocess
        from backtest import get_ohlcv
        from strategies import STRATEGY_WAREHOUSE

        logger.info(f"[EvalChain] Stage: evaluate | pool={self.pool} interval={self.interval}")
        self.progress["current"] = f"评估中 — {self.pool}"

        pool_path = Path(__file__).parent / "stock_pools.json"
        if not pool_path.exists():
            raise RuntimeError("stock_pools.json not found")
        with open(pool_path, encoding="utf-8") as f:
            pools = json.load(f)
        pool_info = pools.get(self.pool)
        if not pool_info:
            raise RuntimeError(f"Pool '{self.pool}' not found")

        symbols = pool_info["symbols"]
        market = pool_info.get("market", "US")

        import main as _m
        wh = _m._load_warehouse()
        active_set = set(wh.get("active", []))
        deleted_set = {d["name"] if isinstance(d, dict) else d for d in wh.get("deleted", [])}
        fav_set = set(wh.get("favourites", []))

        from strategies import list_strategies, STRATEGY_REGISTRY
        all_strats = [s for s in set(list(STRATEGY_REGISTRY.keys()) + list(STRATEGY_WAREHOUSE.keys()))
                      if s not in deleted_set]

        _period_map = {"1d": "max", "1wk": "max", "1mo": "max", "1h": "2y"}
        dl_period = _period_map.get(self.interval, "2y")

        datasets = {}
        # 之前这个阶段进度恒显 0/0 (total 从没设过), 而它恰恰是最慢的。
        # 下载按股票报, 回测阶段下面从 eval_worker 的进度文件里镜像过来。
        self.progress.update({"done": 0, "total": len(symbols), "unit": " 只",
                              "sub_stage": "download", "workers": 1})
        for _i, sym in enumerate(symbols):
            self.progress.update({"done": _i, "current": f"拉取行情 — {sym}", "running": [sym]})
            try:
                df = get_ohlcv(sym, period=dl_period, interval=self.interval, market=market)
                if df is not None and len(df) > 30:
                    if self.limit and len(df) > self.limit:
                        df = df.iloc[-self.limit:]
                    datasets[sym] = pickle.dumps(df)
            except Exception:
                pass
        self.progress.update({"done": len(symbols), "running": []})

        if not datasets:
            return []

        # 写入任务文件，调用 eval_worker 子进程
        task_file = Path(__file__).parent / "_eval_tasks.pkl"
        result_file = Path(__file__).parent / "_eval_results.pkl"
        with open(task_file, "wb") as f:
            pickle.dump({"strategies": all_strats, "datasets": datasets}, f)

        venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        from eval_worker import PROGRESS_FILE as prog_file, read_progress as _read_prog, \
            write_progress as _write_prog
        # 写上自己的 run_id 再开跑: 这个文件是跟「UI 评测排名」共用的,
        # 不打标记的话两边会互相把对方的 done/total 画到自己的面板上
        _write_prog(run_id=f"chain_{self.job_id}", stage="start", done=0, total=0,
                    sym_done=0, sym_total=0, current="", error=None)

        # 原来是 subprocess.run 一路阻塞到底, 所以这一整段(最慢的一段)进度恒 0/0。
        # eval_worker 本来就在往 _eval_progress.json 写真实 done/total, 这里边等边读。
        self.progress.update({"sub_stage": "backtest", "done": 0, "total": 0, "unit": " 组"})
        # 落文件而不是 PIPE: 下面这个 while 循环在子进程跑完之前不会去读管道
        # (communicate() 在循环之后), 而 eval_worker 底下还挂着 N 个 ProcessPool
        # 子进程共用这个 stderr —— 任何 pandas/numpy 警告或 traceback 累计写满
        # 管道缓冲区(约 64KB), 子进程就永久阻塞在 write 上, poll() 永远不返回,
        # 这里空转满 600s 然后报一个假的"eval_worker 超时"。
        # 文件没有缓冲区上限, 出错时再读尾部即可。
        _log = CACHE_DIR / f"eval_worker_{self.job_id}.log"
        _logf = open(_log, "wb")
        try:
            proc = subprocess.Popen(
                [python_cmd, str(Path(__file__).parent / "eval_worker.py")],
                cwd=str(Path(__file__).parent),
                stdout=_logf, stderr=subprocess.STDOUT,
            )
        except Exception:
            # Popen 自己抛(python_cmd 不存在、cwd 有问题)时这个 fd 会一直挂着 ——
            # 正常路径和超时路径都 close 了, 只有这条没有。
            _logf.close()
            raise
        _deadline = time.time() + 600
        while proc.poll() is None:
            if time.time() > _deadline:
                proc.kill()
                _logf.close()
                raise RuntimeError("eval_worker 超时 (600s)")
            try:
                wp = _read_prog()
                if wp:
                    if wp.get("total"):
                        self.progress.update({
                            "done": wp.get("done", 0), "total": wp.get("total", 0),
                            "workers": wp.get("workers", 0), "unit": " 组",
                            "current": f"回测 — {wp.get('current', '')}",
                        })
            except Exception:
                pass
            time.sleep(0.5)
        proc.wait()
        _logf.close()
        if proc.returncode != 0:
            try:
                _tail = _log.read_text(encoding="utf-8", errors="replace")[-300:]
            except Exception:
                _tail = "(日志不可读)"
            raise RuntimeError(f"eval_worker 失败: {_tail}")

        with open(result_file, "rb") as f:
            results = pickle.load(f)

        # 构建排名
        import numpy as np
        ranking = []
        for sname in all_strats:
            scores_data = results.get(sname, [])
            if not scores_data:
                continue
            scores = [d[0] for d in scores_data]
            returns = [d[1] for d in scores_data]
            sharpes = [d[2] for d in scores_data]
            drawdowns = [d[3] for d in scores_data]
            winrates = [d[4] for d in scores_data]
            pfs = [d[5] for d in scores_data]
            trades_list = [d[6] for d in scores_data]
            wins_list = [d[7] for d in scores_data]

            n = len(scores)
            trim = max(1, n // 10)
            sorted_scores = sorted(scores)
            trimmed = sorted_scores[trim:-trim] if n > 2 * trim else sorted_scores

            positive_count = sum(1 for r in returns if r > 0)
            total_trades = sum(trades_list)
            total_wins = sum(wins_list)

            status = "active" if sname in active_set else ("new" if sname not in deleted_set else "archived")

            ranking.append({
                "name": sname,
                "score": round(float(np.mean(scores)), 4),
                "trimmed_score": round(float(np.mean(trimmed)), 4),
                "avg_return": round(float(np.mean(returns)), 2),
                "sharpe": round(float(np.mean(sharpes)), 2),
                "max_drawdown": round(float(np.mean(drawdowns)), 2),
                "win_rate": round(float(np.mean(winrates)), 1),
                "profit_factor": round(float(np.mean(pfs)), 2),
                "avg_trades": round(total_trades / max(1, n), 1),
                "positive_stocks": positive_count,
                "positive_pct": round(positive_count / max(1, n) * 100, 0),
                "stock_count": n,
                "status": status,
                "favourite": sname in fav_set,
                "llm_owner": _m._get_strategy_llm_owner(sname, _m._load_strategy_llm_registry()),
            })

        ranking.sort(key=lambda x: x["trimmed_score"], reverse=True)
        for i, r in enumerate(ranking):
            r["rank"] = i + 1

        self.progress["total"] = len(ranking)
        self.progress["done"] = len(ranking)
        logger.info(f"[EvalChain] Evaluate 完成: {len(ranking)} 策略")
        return ranking

    # ─── Stage 2: LLM Analyze ────────────────────────────────

    def _stage_llm_analyze(self, ranking: list) -> tuple:
        """调用 llm_evaluator 批量分析。"""
        from llm_evaluator import evaluate_batch
        from core_store import create_llm_review

        logger.info(f"[EvalChain] Stage: llm_analyze | {len(ranking)} 策略 | provider={self.provider}")

        strategies = []
        for r in ranking:
            strategies.append({
                "name": r["name"],
                "metrics": r,
                "status": r.get("status", "active"),
                "is_favourite": r.get("favourite", False),
                "rank": r.get("rank", 0),
            })

        pool_name = self.pool
        stock_count = ranking[0]["stock_count"] if ranking else 0

        def _on_progress(done, total, name, verdict):
            self.progress["done"] = done
            self.progress["current"] = f"{name} → {verdict}"

        result = evaluate_batch(
            strategies=strategies,
            pool_name=pool_name,
            stock_count=stock_count,
            interval=self.interval,
            signal_type=self.signal_type,
            provider=self.provider,
            model=self.model,
            max_concurrent=4,
            on_progress=_on_progress,
        )

        reviews = result["reviews"]
        summary = result["summary"]

        # 写入数据库
        for rev in reviews:
            try:
                create_llm_review(
                    strategy_name=rev["strategy_name"],
                    review_round=self.round_id,
                    signal_type=self.signal_type,
                    eval_pool=pool_name,
                    eval_interval=self.interval,
                    provider=self.provider,
                    model=self.model,
                    verdict=rev["verdict"],
                    confidence=rev["confidence"],
                    overall_score=rev.get("overall_score", 0),
                    reasoning=rev.get("reasoning"),
                    risk_warnings=rev.get("risk_warnings"),
                    suggestions=rev.get("suggestions"),
                    input_metrics=next((r for r in ranking if r["name"] == rev["strategy_name"]), {}),
                    raw_response=rev.get("raw_response"),
                    prompt_hash=rev.get("prompt_hash"),
                    response_hash=rev.get("response_hash"),
                    input_tokens=rev.get("token_usage", {}).get("input_tokens", 0),
                    output_tokens=rev.get("token_usage", {}).get("output_tokens", 0),
                    duration_ms=rev.get("duration_ms", 0),
                )
            except Exception as e:
                logger.warning(f"[EvalChain] DB write failed for {rev['strategy_name']}: {e}")

        # 保存 JSON 备份
        review_file = REVIEWS_DIR / f"{self.round_id}_reviews.json"
        with open(review_file, "w", encoding="utf-8") as f:
            json.dump({"reviews": reviews, "summary": summary, "round_id": self.round_id,
                        "pool": pool_name, "provider": self.provider, "model": self.model},
                      f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[EvalChain] LLM Analyze 完成: A={summary['activate_count']} "
                     f"R={summary['archive_count']} W={summary['watch_count']}")
        return reviews, summary

    # ─── Stage 3: Apply Actions ──────────────────────────────

    def _stage_apply_actions(self, reviews: list) -> dict:
        """根据 verdict 更新 strategy_warehouse.json"""
        import main as _m

        actions = {"activated": [], "archived": [], "watched": [], "skipped": []}

        if self.dry_run:
            for rev in reviews:
                v = rev["verdict"]
                if v == "ACTIVATE":
                    actions["activated"].append(rev["strategy_name"])
                elif v == "ARCHIVE":
                    actions["archived"].append(rev["strategy_name"])
                else:
                    actions["watched"].append(rev["strategy_name"])
            logger.info(f"[EvalChain] Dry-run: activate={len(actions['activated'])}, "
                         f"archive={len(actions['archived'])}, watch={len(actions['watched'])}")
            return actions

        # 实际执行
        wh = _m._load_warehouse()
        active_set = set(wh.get("active", []))
        fav_set = set(wh.get("favourites", []))
        archived = wh.get("archived", [])
        archived_names = {a["name"] if isinstance(a, dict) else a for a in archived}

        changed = False
        for rev in reviews:
            name = rev["strategy_name"]
            verdict = rev["verdict"]

            if verdict == "ACTIVATE":
                if name not in active_set:
                    active_set.add(name)
                    # 如果在 archived 中，移除
                    archived = [a for a in archived
                                if (a["name"] if isinstance(a, dict) else a) != name]
                    changed = True
                actions["activated"].append(name)

            elif verdict == "ARCHIVE":
                if name in fav_set:
                    actions["skipped"].append(name)
                    continue
                if name in active_set:
                    active_set.discard(name)
                    archived.append({"name": name, "archived_at": datetime.now().strftime("%Y-%m-%d"),
                                     "reason": f"LLM eval: {rev.get('reasoning', '')[:100]}"})
                    changed = True
                actions["archived"].append(name)

            else:  # WATCH
                actions["watched"].append(name)

        if changed:
            wh["active"] = sorted(active_set)
            wh["archived"] = archived
            wh["history"] = wh.get("history", [])
            wh["history"].append({
                "action": "llm_eval_chain",
                "round_id": self.round_id,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "summary": {k: len(v) for k, v in actions.items()},
            })
            _m._save_warehouse(wh)
            logger.info(f"[EvalChain] Warehouse 已更新")

        return actions

    # ─── Stage 4: Report ─────────────────────────────────────

    def _stage_report(self, reviews: list, summary: dict) -> str:
        """生成 Markdown 报告。"""
        from llm_evaluator import generate_report

        md = generate_report(
            reviews=reviews,
            summary=summary,
            pool_name=self.pool,
            round_id=self.round_id,
            provider=self.provider,
            model=self.model,
        )

        report_path = REPORTS_DIR / f"eval_chain_{self.round_id}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"[EvalChain] 报告已保存: {report_path.name}")
        return str(report_path)

    # ─── Checkpoint ──────────────────────────────────────────

    def _save_checkpoint(self, stage: str, data: dict):
        cp = {"stage": stage, "round_id": self.round_id, "config": {
            "pool": self.pool, "interval": self.interval, "provider": self.provider,
            "model": self.model, "dry_run": self.dry_run,
        }, **data}
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(cp, f, indent=2, ensure_ascii=False, default=str)

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

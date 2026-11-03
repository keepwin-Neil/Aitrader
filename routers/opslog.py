"""Ops-log admin panel backend — surfaces internal observability to the
9000 admin UI.

Exposes read-only views of runtime log sources that are otherwise
scattered across the project directory:

  1. **Scan audit JSONL** — .cache/scan_audit/YYYY-MM-DD/*.jsonl
  2. **Baseline snapshots** — baseline_YYYYMMDD.json + compare_baseline diff
  3. **Unhandled-error log** — <AITRADER_LOG_DIR>/unhandled_errors.log (+ rotations)

All routes are admin-gated via ``admin_auth.require_admin`` at the router
level — when ``AITRADER_ADMIN_TOKEN`` env is set, any call to /opslog/*
requires matching ``X-Admin-Token`` header. Env unset → no-op gate (same
backward-compat contract as /policy/reload).

Security constraints:
  * Date params validated against YYYY-MM-DD regex (no path traversal).
  * File names validated against allowlist regex (no `..`, no `/`).
  * Subprocess calls bounded by timeout; no shell=True.
  * Output sizes capped to keep admin UI responsive.

Scope (Stage 1): read-only surfacing + one write op (trigger new baseline).
Stage 2 (future): test-script runner with sandboxed execution.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

import scan_audit

from admin_auth import require_admin

router = APIRouter(
    prefix="/opslog",
    tags=["opslog"],
    dependencies=[Depends(require_admin)],
)

# --------------------------------------------------------------------------
# Constants / whitelists
# --------------------------------------------------------------------------

_SCAN_AUDIT_ROOT = scan_audit._AUDIT_DIR
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FILE_RE = re.compile(r"^[a-zA-Z0-9_\-.]+\.jsonl$")
_BASELINE_FILE_RE = re.compile(r"^baseline_\d{8}\.json$")

# scan_audit JSONL tail cap — any one file can get chunky.
_SCAN_TAIL_MAX_LINES = 1000
# subprocess timeouts (seconds).
_BASELINE_RUN_TIMEOUT = 120
_COMPARE_TIMEOUT = 15

# Project root — assumes this file lives at <root>/routers/opslog.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _validate_date(date_str: str) -> None:
    if not _DATE_RE.match(date_str):
        raise HTTPException(400, f"date must be YYYY-MM-DD, got {date_str!r}")


def _validate_scan_audit_filename(name: str) -> None:
    if not _FILE_RE.match(name):
        raise HTTPException(
            400,
            f"filename must match ^[a-zA-Z0-9_\\-.]+\\.jsonl$ "
            f"(no slashes, no '..'); got {name!r}",
        )


def _validate_baseline_filename(name: str) -> None:
    if not _BASELINE_FILE_RE.match(name):
        raise HTTPException(
            400,
            f"baseline filename must match ^baseline_\\d{{8}}\\.json$; "
            f"got {name!r}",
        )


# --------------------------------------------------------------------------
# Route: unhandled-error log
#
# 为什么挂在这儿: main.py 的全局异常处理器把未捕获堆栈落到
# <AITRADER_LOG_DIR>/unhandled_errors.log (带轮转)。这个文件被 .gitignore 挡着,
# 也没有任何路由 —— 等于只有 SSH 上机器才看得到。而本文件正是项目里"给管理员
# 看运行时日志"的那个面。不接进来, 要么这份日志根本没人读, 要么将来再写一个
# 读取器, 把下面这套路径校验/截断逻辑再抄一遍。
# --------------------------------------------------------------------------

_ERR_TAIL_MAX_LINES = 2000
_ERR_FILE_RE = re.compile(r"^unhandled_errors\.log(\.\d+)?$")


def _err_log_dir() -> Path:
    """跟 main.LOG_DIR 同源, 但不 import main (opslog 是薄路由层, 不该拖起整个 app)。"""
    return Path(os.getenv("AITRADER_LOG_DIR", str(Path(__file__).resolve().parent.parent / "logs")))


def _tail_lines(fp: Path, n: int) -> list:
    """只读文件尾部的 n 行。

    原来是 f.readlines()[-n:] —— 把整个文件读进内存再切。轮转配的是
    maxBytes=10MB, 实测 tracemalloc 峰值 17.9MB(一次请求), 而真正要的只有
    最后几百行。管理面板多开几个标签就是几十 MB 的无谓峰值。
    从尾部往前按块读, 够 n 行就停。
    """
    block = 64 * 1024
    with open(fp, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        buf = b""
        pos = end
        while pos > 0 and buf.count(b"\n") <= n:
            step = min(block, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    text = buf.decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


@router.get("/errors")
def unhandled_errors(
    lines: int = Query(200, ge=1, le=_ERR_TAIL_MAX_LINES),
    file: str = Query("unhandled_errors.log"),
):
    """未捕获异常日志的尾部。file 只允许 unhandled_errors.log[.N] (轮转产物)。"""
    if not _ERR_FILE_RE.match(file):
        raise HTTPException(
            400,
            f"file must match ^unhandled_errors\\.log(\\.\\d+)?$ "
            f"(no slashes, no '..'); got {file!r}",
        )
    d = _err_log_dir()
    fp = d / file
    # 再确认一次落在目录内 —— 正则已经挡住了 ../, 这是第二道
    try:
        fp.resolve().relative_to(d.resolve())
    except (ValueError, OSError):
        raise HTTPException(400, "path escapes log dir")
    if not fp.exists():
        return {"dir": str(d), "file": file, "exists": False,
                "lines": [], "size_bytes": 0,
                "available": sorted(p.name for p in d.glob("unhandled_errors.log*")) if d.exists() else []}
    try:
        tail = _tail_lines(fp, lines)
    except OSError as e:
        raise HTTPException(500, f"read failed: {e}")
    return {
        "dir": str(d),
        "file": file,
        "exists": True,
        "size_bytes": fp.stat().st_size,
        "mtime": datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc).isoformat(),
        "returned_lines": len(tail),
        "lines": tail,
        "available": sorted(p.name for p in d.glob("unhandled_errors.log*")),
    }


# --------------------------------------------------------------------------
# Route: scan_audit — list JSONL files for a given date
# --------------------------------------------------------------------------


@router.get("/scan-audit")
def list_scan_audit(date: Optional[str] = Query(None)):
    """List scan_audit JSONL files for a given date (default: today UTC).

    Returns: list of {name, size_bytes, mtime_utc, line_count_approx}.
    ``line_count_approx`` is a fast estimate from byte count / 200; real
    count requires reading the file (expensive for large JSONL).
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _validate_date(date)

    day_dir = _SCAN_AUDIT_ROOT / date
    if not day_dir.exists():
        return {"date": date, "dir": str(day_dir), "exists": False, "files": []}

    files = []
    for p in sorted(day_dir.glob("*.jsonl")):
        try:
            stat = p.stat()
            size = stat.st_size
            mtime = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            # rough estimate — each JSONL line typically 150-250 bytes
            approx_lines = size // 200 if size else 0
            files.append({
                "name": p.name,
                "size_bytes": size,
                "mtime_utc": mtime,
                "line_count_approx": approx_lines,
            })
        except OSError:
            continue

    return {
        "date": date,
        "dir": str(day_dir),
        "exists": True,
        "file_count": len(files),
        "files": files,
    }


# --------------------------------------------------------------------------
# Route: scan_audit — tail one JSONL file
# --------------------------------------------------------------------------


@router.get("/scan-audit/{date}/{file}")
def tail_scan_audit_file(
    date: str,
    file: str,
    tail: int = Query(100, ge=1, le=_SCAN_TAIL_MAX_LINES),
):
    """Return the last N JSON lines of a scan_audit JSONL file.

    Safe against path traversal: date must be YYYY-MM-DD, file must match
    ^[a-zA-Z0-9_\\-.]+\\.jsonl$ (no slashes, no '..'). Reads are bounded
    by tail<=_SCAN_TAIL_MAX_LINES.
    """
    _validate_date(date)
    _validate_scan_audit_filename(file)

    target = _SCAN_AUDIT_ROOT / date / file
    # Resolve + verify still under _SCAN_AUDIT_ROOT (defense in depth).
    try:
        target_real = target.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise HTTPException(404, f"scan_audit file not found: {date}/{file}")
    if _SCAN_AUDIT_ROOT.resolve() not in target_real.parents:
        raise HTTPException(403, "path escape detected")

    # Efficient tail: read last ~chunk bytes, split, take last N lines.
    try:
        size = target_real.stat().st_size
        # Read enough bytes to likely cover `tail` lines (avg ~200 bytes/line).
        read_bytes = min(size, max(tail * 400, 8192))
        with open(target_real, "rb") as fh:
            fh.seek(-read_bytes, 2 if size > read_bytes else 0)
            chunk = fh.read()
    except OSError as exc:
        raise HTTPException(500, f"read failed: {exc!r}")

    text = chunk.decode("utf-8", errors="replace")
    raw_lines = [ln for ln in text.splitlines() if ln.strip()]
    raw_lines = raw_lines[-tail:]

    # Best-effort parse — surface parse errors per-line so a bad entry
    # doesn't nuke the whole response.
    parsed = []
    for ln in raw_lines:
        try:
            parsed.append(json.loads(ln))
        except json.JSONDecodeError as exc:
            parsed.append({
                "_parse_error": str(exc),
                "_raw": ln[:300],
            })

    return {
        "date": date,
        "file": file,
        "path": str(target_real),
        "size_bytes": size,
        "tail_requested": tail,
        "lines_returned": len(parsed),
        "entries": parsed,
    }


# --------------------------------------------------------------------------
# Route: baseline snapshots — list
# --------------------------------------------------------------------------


@router.get("/baselines")
def list_baselines():
    """Enumerate baseline_YYYYMMDD.json files in the project root.

    Returns ordered newest-first for UI convenience.
    """
    out = []
    for p in sorted(
        _PROJECT_ROOT.glob("baseline_*.json"), reverse=True
    ):
        if not _BASELINE_FILE_RE.match(p.name):
            continue
        try:
            stat = p.stat()
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            out.append({
                "name": p.name,
                "error": f"read failed: {exc!r}",
            })
            continue
        out.append({
            "name": p.name,
            "size_bytes": stat.st_size,
            "measured_at_utc": data.get("measured_at_utc"),
            "group_count": data.get("group_count"),
            "unique_symbols_total": data.get("unique_symbols_total"),
            "stock_overlap_ratio": data.get("stock_overlap_ratio"),
            "unique_strategies_total": data.get("unique_strategies_total"),
            "strategy_overlap_ratio": data.get("strategy_overlap_ratio"),
            "llm_calls_yesterday": data.get("llm_calls_yesterday"),
            "duplicate_llm_calls_yesterday": data.get(
                "duplicate_llm_calls_yesterday"
            ),
        })
    return {"count": len(out), "baselines": out}


# --------------------------------------------------------------------------
# Route: baseline snapshots — run new snapshot now
# --------------------------------------------------------------------------


@router.post("/baselines/run")
def run_baseline():
    """Trigger scripts/baseline_overlap.py synchronously.

    Runs in-process-tree via subprocess (separate Python interpreter) so
    import-time side-effects in core_engine/main don't collide with the
    FastAPI worker's already-loaded modules.

    Blocks up to _BASELINE_RUN_TIMEOUT seconds. Admin UI shows a spinner.
    """
    script = _PROJECT_ROOT / "scripts" / "baseline_overlap.py"
    if not script.exists():
        raise HTTPException(500, f"baseline script missing: {script}")

    cmd = [sys.executable, str(script)]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=_BASELINE_RUN_TIMEOUT,
            check=False,
            # Inherit env — needs DATABASE_URL etc.
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            504,
            f"baseline script timeout after {_BASELINE_RUN_TIMEOUT}s",
        )

    # Find the newest baseline file — should be the one we just wrote.
    newest = None
    for p in sorted(_PROJECT_ROOT.glob("baseline_*.json"), reverse=True):
        if _BASELINE_FILE_RE.match(p.name):
            newest = p.name
            break

    return {
        "rc": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-50:],
        "stderr_tail": result.stderr.splitlines()[-30:],
        "newest_file": newest,
        "ran_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


# --------------------------------------------------------------------------
# Route: baseline snapshots — diff two snapshots
# --------------------------------------------------------------------------


@router.get("/baselines/diff")
def diff_baselines(
    before: str = Query(...),
    after: str = Query(...),
):
    """Run scripts/compare_baseline.py before.json after.json and return
    the captured stdout (formatted diff report).

    Both filenames must match ^baseline_\\d{8}\\.json$; both files must
    exist in the project root.
    """
    _validate_baseline_filename(before)
    _validate_baseline_filename(after)

    before_path = _PROJECT_ROOT / before
    after_path = _PROJECT_ROOT / after
    for p in (before_path, after_path):
        if not p.exists():
            raise HTTPException(404, f"baseline not found: {p.name}")

    script = _PROJECT_ROOT / "scripts" / "compare_baseline.py"
    if not script.exists():
        raise HTTPException(500, f"compare script missing: {script}")

    cmd = [sys.executable, str(script), str(before_path), str(after_path)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_COMPARE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            504,
            f"compare script timeout after {_COMPARE_TIMEOUT}s",
        )

    return {
        "before": before,
        "after": after,
        "rc": result.returncode,
        "stdout": result.stdout,
        "stderr_tail": result.stderr.splitlines()[-10:],
    }


# --------------------------------------------------------------------------
# Route: baseline file — single snapshot content
# --------------------------------------------------------------------------


@router.get("/baselines/{name}")
def get_baseline_file(name: str):
    """Return the full content of a single baseline_*.json file.

    Filename must match ^baseline_\\d{8}\\.json$ — no traversal.
    """
    _validate_baseline_filename(name)
    path = _PROJECT_ROOT / name
    if not path.exists():
        raise HTTPException(404, f"baseline not found: {name}")
    try:
        return {
            "name": name,
            "path": str(path),
            "content": json.loads(path.read_text(encoding="utf-8")),
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"read/parse failed: {exc!r}")

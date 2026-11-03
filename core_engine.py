from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
import pandas as pd

import scan_center
import scan_audit


LLM_PROVIDER_DEFAULTS = {
    "OpenAI": {"provider": "openai", "model_name": "gpt-4.1"},
    "Claude": {"provider": "claude", "model_name": "claude-sonnet-4-20250514"},
    "Grok": {"provider": "grok", "model_name": "grok-3"},
    "Gemini": {"provider": "gemini", "model_name": "gemini-2.5-flash"},
    "DeepSeek": {"provider": "deepseek", "model_name": "deepseek-r1"},
    "Llama": {"provider": "llama", "model_name": "llama-4-maverick"},
    "Mistral": {"provider": "mistral", "model_name": "mistral-large-latest"},
    "Qwen": {"provider": "qwen", "model_name": "qwen3-235b-a22b"},
    "Cohere": {"provider": "cohere", "model_name": "command-a-03-2025"},
    "Yi": {"provider": "yi", "model_name": "yi-lightning"},
    "Doubao": {"provider": "doubao", "model_name": "doubao-1.5-pro-256k"},
    "Zhipu": {"provider": "zhipu", "model_name": "glm-4-plus"},
    "MiniMax": {"provider": "minimax", "model_name": "MiniMax-M1"},
    "Moonshot": {"provider": "moonshot", "model_name": "moonshot-v1-auto"},
    "StepFun": {"provider": "stepfun", "model_name": "step-2-16k"},
}


# ── 默认模型名的唯一出处 ─────────────────────────────────────────────────────
# 以前这些字符串散在 9 个文件里 (eval_chain / llm_evaluator / news_scorer /
# opinion_cache / routers.eval_chain / 三个测试脚本 / 前端 HTML), 各写各的,
# 连同一个 provider 在不同文件里的默认值都不一样
# (news_scorer 写 claude-sonnet-4-6, 别处写 claude-sonnet-4-20250514)。
# 改一处漏八处的典型。现在统一从 LLM_PROVIDER_DEFAULTS 取。
#
# 注意这只是**兜底默认**: 真正生效的是管理后台保存的 model_name
# (list_llm_key_settings / _resolve_model_name 都是先读 saved 再回落到这里)。
def default_model_for(provider: str, fallback: str = "") -> str:
    """按 provider 取默认模型名。provider 不认识时返回 fallback。"""
    want = _normalize_provider(provider or "")
    for defaults in LLM_PROVIDER_DEFAULTS.values():
        if _normalize_provider(defaults.get("provider", "")) == want:
            return defaults.get("model_name") or fallback
    return fallback


# ── Market trading hours (local time) ──────────────────────────────
# Used to gate signal processing — signals are only processed during market hours.
from zoneinfo import ZoneInfo

_per_market_cache: dict = {"mtime": 0.0, "data": None}


MARKET_TRADING_HOURS: dict[str, dict] = {
    "US": {
        "tz": ZoneInfo("America/New_York"),
        "open_hour": 9, "open_minute": 30,
        "close_hour": 16, "close_minute": 0,
    },
    "CN": {
        "tz": ZoneInfo("Asia/Shanghai"),
        "open_hour": 9, "open_minute": 30,
        "close_hour": 15, "close_minute": 0,
    },
    "HK": {
        "tz": ZoneInfo("Asia/Hong_Kong"),
        "open_hour": 9, "open_minute": 30,
        "close_hour": 16, "close_minute": 0,
    },
    "UK": {
        "tz": ZoneInfo("Europe/London"),
        "open_hour": 8, "open_minute": 0,
        "close_hour": 16, "close_minute": 30,
    },
}


def _is_market_open(market: str) -> bool:
    """Check if the given market is currently within trading hours (weekday + time range)."""
    mkt = MARKET_TRADING_HOURS.get(market.upper())
    if not mkt:
        return True  # unknown market → allow (don't block)
    now = datetime.now(mkt["tz"])
    # Weekend check (Monday=0 … Sunday=6)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=mkt["open_hour"], minute=mkt["open_minute"], second=0, microsecond=0)
    close_time = now.replace(hour=mkt["close_hour"], minute=mkt["close_minute"], second=0, microsecond=0)
    return open_time <= now <= close_time
LLM_REQUEST_TIMEOUT_SEC = float(os.getenv("AITRADER_LLM_TIMEOUT_SEC", "25"))
OPENAI_COMPATIBLE_PROVIDERS = {
    "openai", "grok", "deepseek",
    "llama", "mistral", "qwen", "cohere",
    "yi", "doubao", "zhipu", "minimax", "moonshot", "stepfun",
}
LLM_API_CONFIG = {
    "openai": {
        "key_envs": ["OPENAI_API_KEY"],
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "chat_path": "/chat/completions",
    },
    "grok": {
        "key_envs": ["XAI_API_KEY", "GROK_API_KEY"],
        "base_url_env": "XAI_BASE_URL",
        "default_base_url": "https://api.x.ai/v1",
        "chat_path": "/chat/completions",
    },
    "deepseek": {
        "key_envs": ["DEEPSEEK_API_KEY"],
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "chat_path": "/chat/completions",
    },
    "claude": {
        "key_envs": ["ANTHROPIC_API_KEY"],
        "default_base_url": "https://api.anthropic.com/v1/messages",
    },
    "gemini": {
        "key_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta/models",
    },
    "llama": {
        "key_envs": ["LLAMA_API_KEY"],
        "base_url_env": "LLAMA_BASE_URL",
        "default_base_url": "https://api.llama.com/v1",
        "chat_path": "/chat/completions",
    },
    "mistral": {
        "key_envs": ["MISTRAL_API_KEY"],
        "base_url_env": "MISTRAL_BASE_URL",
        "default_base_url": "https://api.mistral.ai/v1",
        "chat_path": "/chat/completions",
    },
    "qwen": {
        "key_envs": ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],
        "base_url_env": "QWEN_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_path": "/chat/completions",
    },
    "cohere": {
        "key_envs": ["COHERE_API_KEY", "CO_API_KEY"],
        "base_url_env": "COHERE_BASE_URL",
        "default_base_url": "https://api.cohere.com/v2",
        "chat_path": "/chat/completions",
    },
    "yi": {
        "key_envs": ["YI_API_KEY"],
        "base_url_env": "YI_BASE_URL",
        "default_base_url": "https://api.lingyiwanwu.com/v1",
        "chat_path": "/chat/completions",
    },
    "doubao": {
        "key_envs": ["ARK_API_KEY", "DOUBAO_API_KEY"],
        "base_url_env": "DOUBAO_BASE_URL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "chat_path": "/chat/completions",
    },
    "zhipu": {
        "key_envs": ["ZHIPU_API_KEY"],
        "base_url_env": "ZHIPU_BASE_URL",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat_path": "/chat/completions",
    },
    "minimax": {
        "key_envs": ["MINIMAX_API_KEY"],
        "base_url_env": "MINIMAX_BASE_URL",
        "default_base_url": "https://api.minimax.chat/v1",
        "chat_path": "/text/chatcompletion_v2",
    },
    "moonshot": {
        "key_envs": ["MOONSHOT_API_KEY"],
        "base_url_env": "MOONSHOT_BASE_URL",
        "default_base_url": "https://api.moonshot.cn/v1",
        "chat_path": "/chat/completions",
    },
    "stepfun": {
        "key_envs": ["STEPFUN_API_KEY"],
        "base_url_env": "STEPFUN_BASE_URL",
        "default_base_url": "https://api.stepfun.com/v1",
        "chat_path": "/chat/completions",
    },
}
# ── CLI mode defaults (provider → command template) ──
# Placeholders: {system_prompt_file} {user_prompt_file} {model} {max_tokens}
def _find_cli_bin(name: str) -> str:
    """自动探测 CLI 工具路径，避免写死本地路径。"""
    import shutil
    # 优先检查 ~/.local/bin（claude CLI 默认安装位置）
    local_bin = Path.home() / ".local" / "bin" / name
    if local_bin.exists():
        return str(local_bin)
    # 检查系统 PATH
    found = shutil.which(name)
    if found:
        return found
    # 常见路径
    for p in ["/opt/homebrew/bin", "/usr/local/bin", "/snap/bin"]:
        candidate = Path(p) / name
        if candidate.exists():
            return str(candidate)
    return name  # fallback: 依赖 PATH

LLM_CLI_DEFAULTS = {
    "claude": {
        "bin": _find_cli_bin("claude"),
        # Display-only template (shown in admin UI). Execution uses LLM_CLI_ARGV_SPECS below.
        "template": '{bin} -p --system-prompt-file {system_prompt_file} --output-format json --model {model} < {user_prompt_file}',
    },
    "gemini": {
        "bin": _find_cli_bin("gemini"),
        "template": 'cat {user_prompt_file} | {bin} -p " " -m {model} -o json',
    },
    "openai": {
        "bin": _find_cli_bin("codex"),
        "template": 'cat {system_prompt_file} {user_prompt_file} | {bin} exec -m {model} -o {output_file} --dangerously-bypass-approvals-and-sandbox --ephemeral -',
    },
}

# ── CLI argv execution specs (SECURE: shell=False) ────────────────────────────
# Codex review 2026-04-22 [critical]: the previous implementation passed a
# user-editable template through `subprocess.run(..., shell=True)`, meaning any
# admin with HTTP access (or XSS in the admin UI) could achieve arbitrary host
# command execution. We now ignore client-supplied `cli_command` entirely and
# execute fixed, server-side argv specs without a shell.
#
# Each spec is `{"argv": [...], "stdin_files": [...]}`:
#   - `argv[i]` is format-substituted with {bin, model, system_prompt_file,
#     user_prompt_file, output_file, max_tokens}.
#   - `stdin_files` are concatenated and piped to the process stdin (replaces
#     the old `<` / `cat ... |` shell pipelines).
# `model_name` is whitelist-validated before substitution to close any
# residual shell-metacharacter paths that might sneak through argv.
LLM_CLI_ARGV_SPECS = {
    "claude": {
        "argv": [
            "{bin}", "-p",
            "--system-prompt-file", "{system_prompt_file}",
            "--output-format", "json",
            "--model", "{model}",
        ],
        "stdin_files": ["{user_prompt_file}"],
    },
    "gemini": {
        "argv": [
            "{bin}", "-p", " ",
            "-m", "{model}",
            "-o", "json",
        ],
        "stdin_files": ["{user_prompt_file}"],
    },
    "openai": {
        "argv": [
            "{bin}", "exec",
            "-m", "{model}",
            "-o", "{output_file}",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ephemeral",
            "-",
        ],
        "stdin_files": ["{system_prompt_file}", "{user_prompt_file}"],
    },
}
# Model name whitelist: alphanumeric + common separators only. Blocks shell
# metacharacters (`;`, `|`, backticks, `$()`, spaces, quotes, newlines, etc.).
import re as _cli_re
_CLI_MODEL_NAME_RE = _cli_re.compile(r"^[A-Za-z0-9_.\-:/]+$")
del _cli_re
DAEMON_STATUS_FILE = Path(__file__).parent / ".cache" / "scan_daemon_status.json"
LLM_PROVIDER_CONFIG_FILE = Path(__file__).parent / ".cache" / "llm_keys.json"
DAEMON_STATUS_FILE.parent.mkdir(exist_ok=True)
LLM_PROVIDER_CONFIG_FILE.parent.mkdir(exist_ok=True)
logger = logging.getLogger("uvicorn.error")
_DAEMON_LOCK = threading.RLock()
_LLM_CONFIG_LOCK = threading.RLock()


# ── CLI per-provider mutex (docs/llm_call_rules.md §2) ──────────────────────
# 强制同一 provider 的 CLI 调用串行（禁并发），不同 provider 之间互相独立。
# key = _normalize_provider(provider)；锁的获取/释放必须放 try/finally。
# 仅在 _call_via_cli 执行 subprocess 期间持锁，不要在持锁时做 HTTP/DB 调用。
# CLI 超时固定 120s（见 LLM_CLI_TIMEOUT_SEC），失败/超时由调用方
# (quant_llm / stock_report / llm_evaluator / news_scorer) 捕获并回退 API 模式。
LLM_CLI_TIMEOUT_SEC = 120.0

class _CliMutexRegistry:
    """Per-provider mutex registry for CLI subprocess serialization."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()

    def get(self, provider: str) -> threading.Lock:
        key = _normalize_provider(provider) or "_default"
        with self._meta_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock


_CLI_MUTEX = _CliMutexRegistry()


_DAEMON_THREAD: Optional[threading.Thread] = None
_DAEMON_STATE = {
    "running": False,
    "started_at": None,
    "last_tick_at": None,
    "last_run_at": None,
    "last_error": None,
}


def _main():
    import importlib

    return importlib.import_module("main")


def _utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _group_signature(interval: str, refresh_sec: int, signal_side: str, pool_names: list[str], strategies: list[str]) -> str:
    raw = json.dumps(
        {
            "interval": interval,
            "refresh_sec": refresh_sec,
            "signal_side": signal_side,
            "pool_names": sorted(pool_names),
            "strategies": sorted(strategies),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_provider(provider: Optional[str]) -> str:
    return str(provider or "").strip().lower()


def _provider_env_key(provider: str) -> Optional[str]:
    config = LLM_API_CONFIG.get((provider or "").lower(), {})
    for env_name in config.get("key_envs", []):
        value = os.getenv(env_name)
        if value:
            return value.strip()
    return None


def _read_llm_provider_configs() -> dict[str, dict]:
    with _LLM_CONFIG_LOCK:
        if not LLM_PROVIDER_CONFIG_FILE.exists():
            return {}
        try:
            data = json.loads(LLM_PROVIDER_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read LLM key config file", exc_info=True)
            return {}
    items = data if isinstance(data, dict) else {}
    cleaned: dict[str, dict] = {}
    for key, value in items.items():
        provider = _normalize_provider(key)
        if provider in LLM_API_CONFIG and isinstance(value, dict):
            cleaned[provider] = value
    return cleaned


def _write_llm_provider_configs(configs: dict[str, dict]) -> None:
    payload = {k: v for k, v in sorted(configs.items()) if k in LLM_API_CONFIG and isinstance(v, dict)}
    with _LLM_CONFIG_LOCK:
        LLM_PROVIDER_CONFIG_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _provider_saved_config(provider: str) -> dict:
    return dict(_read_llm_provider_configs().get(_normalize_provider(provider)) or {})


def _call_via_cli(provider: str, model_name: str, system_prompt: str,
                  user_prompt: str, max_tokens: int = 200) -> tuple[Optional[dict], str, dict]:
    """Call LLM via CLI command. Returns (parsed_json, raw_text, token_usage).

    Supports three CLI output patterns:
      1. Claude Code (--output-format json): {"type":"result","result":"..."}
      2. Codex (-o output_file): result written to file, stdout may be JSONL events
      3. Gemini / generic: raw text on stdout
    """
    import subprocess
    import tempfile

    # Codex review 2026-04-22 [critical]: execution no longer uses
    # client-editable templates or `shell=True`. We look up a fixed argv spec
    # and pipe stdin from the prompt files. The `cli_command` field in
    # admin config is kept only as a human-readable reference for the UI.
    norm = _normalize_provider(provider)
    argv_spec = LLM_CLI_ARGV_SPECS.get(norm)
    if not argv_spec:
        raise RuntimeError(
            f"CLI mode not supported for provider {provider!r}: "
            "no server-side argv spec registered"
        )

    # Whitelist model_name — blocks shell metacharacters even though argv
    # execution would already prevent most injection.
    model_name_str = str(model_name or "").strip()
    if not model_name_str:
        raise ValueError("CLI mode requires a non-empty model_name")
    if not _CLI_MODEL_NAME_RE.match(model_name_str):
        raise ValueError(
            f"invalid model_name for CLI execution: {model_name_str!r} "
            "(allowed: A-Za-z0-9, '_.-:/')"
        )

    defaults = LLM_CLI_DEFAULTS.get(norm, {})
    cli_bin = defaults.get("bin", provider)

    # Write prompts to temp files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as sf:
        sf.write(system_prompt)
        sys_file = sf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as uf:
        uf.write(user_prompt)
        usr_file = uf.name

    # Output file for CLIs that use -o (e.g. Codex)
    out_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    out_file.close()
    out_path = out_file.name

    try:
        # Build argv from spec, then read stdin files into a single payload.
        _fmt_mapping = {
            "bin": cli_bin,
            "system_prompt_file": sys_file,
            "user_prompt_file": usr_file,
            "output_file": out_path,
            "model": model_name_str,
            "max_tokens": str(max_tokens),
        }
        argv = [str(arg).format(**_fmt_mapping) for arg in argv_spec["argv"]]
        stdin_payload = ""
        for sf_template in (argv_spec.get("stdin_files") or []):
            _path = str(sf_template).format(**_fmt_mapping)
            try:
                stdin_payload += Path(_path).read_text(encoding="utf-8")
            except Exception as _e:
                raise RuntimeError(f"failed to read CLI stdin file {_path!r}: {_e}") from _e

        # CLI tools mostly use their own login session — no API key needed.
        # Ensure common tool paths are in PATH (node, claude, etc.)
        cli_env = {**os.environ}
        _extra_paths = "/opt/homebrew/bin:/usr/local/bin:" + str(Path.home() / ".local" / "bin")
        cli_env["PATH"] = f"{_extra_paths}:{cli_env.get('PATH', '')}"
        if norm == "gemini":
            cli_env["GEMINI_SYSTEM_PROMPT"] = system_prompt
            # Gemini CLI 的 settings.json 若 selectedType=api-key，需要 GEMINI_API_KEY。
            # 若缺这个 env 会报 "Invalid auth method selected" (exit 41)。
            # 从本地存储的 provider config 读 api_key，同时填 GOOGLE_API_KEY 作 fallback。
            try:
                _saved_gem = _provider_saved_config(provider)
                _gem_key = (_saved_gem.get("api_key") or "").strip()
                if _gem_key:
                    cli_env.setdefault("GEMINI_API_KEY", _gem_key)
                    cli_env.setdefault("GOOGLE_API_KEY", _gem_key)
            except Exception:
                pass

        logger.info(
            "[CLI:%s] exec argv[0]=%s model=%s stdin_bytes=%d",
            provider, argv[0], model_name_str, len(stdin_payload),
        )
        # docs/llm_call_rules.md §3: CLI 等待 120 秒视为超时。
        # subprocess.run 超时会自动 kill 子进程并 wait 回收（不留僵尸），
        # 然后抛 TimeoutExpired 交给调用方做 API fallback。
        result = subprocess.run(
            argv, shell=False, input=stdin_payload,
            capture_output=True, text=True,
            timeout=LLM_CLI_TIMEOUT_SEC,
            env=cli_env,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[:300]
            stdout = (result.stdout or "").strip()[:500]
            # Claude Code / Codex may exit non-zero due to hook failures
            # but still produce valid JSON output. Check before raising.
            recovered = False
            if stdout.startswith("{"):
                try:
                    outer = json.loads(stdout)
                    if isinstance(outer, dict):
                        # Claude Code: hook failure but result is valid
                        if outer.get("type") == "result" and not outer.get("is_error", True):
                            recovered = True
                            logger.info(f"[CLI:{provider}] Non-zero exit but valid result recovered")
                        # Codex: check for completion in JSONL
                except (json.JSONDecodeError, ValueError):
                    pass
            if not recovered:
                detail = stderr or stdout
                raise RuntimeError(f"CLI exited {result.returncode}: {detail}")

        # 1. Try reading from output file (Codex -o mode)
        raw_output = ""
        try:
            file_content = Path(out_path).read_text(encoding="utf-8").strip()
            if file_content:
                raw_output = file_content
        except Exception:
            pass

        # 2. Fall back to stdout
        if not raw_output:
            raw_output = (result.stdout or "").strip()

        # 3. If stdout is JSONL (Codex --json), extract last assistant message
        text = raw_output
        usage = {}
        if raw_output.startswith("{") and "\n" in raw_output:
            # Possibly JSONL — scan for last assistant message
            for line in reversed(raw_output.splitlines()):
                try:
                    evt = json.loads(line)
                    if isinstance(evt, dict):
                        # Codex JSONL event with message
                        if evt.get("type") == "message" and evt.get("role") == "assistant":
                            text = str(evt.get("content") or "").strip()
                            break
                        # Usage event
                        if evt.get("type") == "usage":
                            usage = evt
                except (json.JSONDecodeError, ValueError):
                    continue

        # 4. Try single JSON object (Claude Code / Gemini -o json / generic)
        if not usage:
            try:
                outer = json.loads(text if text == raw_output else raw_output)
                if isinstance(outer, dict):
                    # Claude Code: {"type":"result","result":"..."}
                    if outer.get("type") == "result" and "result" in outer:
                        text = str(outer["result"]).strip()
                        usage = outer.get("usage") or {}
                    # Gemini CLI -o json: {"session_id":..., "response":"...", "stats":{...}}
                    elif "response" in outer and "stats" in outer:
                        resp = str(outer["response"]).strip()
                        # Strip markdown code fences if present
                        if resp.startswith("```"):
                            lines = resp.split("\n")
                            lines = [l for l in lines if not l.startswith("```")]
                            resp = "\n".join(lines).strip()
                        text = resp
                        # Extract token usage from stats.models.{model}.tokens
                        stats_models = outer.get("stats", {}).get("models", {})
                        for _mname, mdata in stats_models.items():
                            toks = mdata.get("tokens", {})
                            if toks:
                                usage = {
                                    "input_tokens": toks.get("input", 0) + toks.get("prompt", 0),
                                    "output_tokens": toks.get("candidates", 0),
                                }
                                break
                    elif "text" in outer:
                        text = str(outer["text"]).strip()
                        usage = outer.get("usage") or {}
                    else:
                        usage = outer.get("usage") or {}
            except (json.JSONDecodeError, ValueError):
                pass

        parsed = _extract_json_candidate(text)
        # Estimate tokens when CLI doesn't report usage
        est_input = len(system_prompt + user_prompt) // 4
        est_output = len(text) // 4
        token_usage = {
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens") or est_input,
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens") or est_output,
            "total_tokens": (usage.get("input_tokens") or usage.get("prompt_tokens") or est_input)
                         + (usage.get("output_tokens") or usage.get("completion_tokens") or est_output),
        }
        return parsed, text, token_usage
    finally:
        for f in (sys_file, usr_file, out_path):
            try:
                os.unlink(f)
            except OSError:
                pass


def _mask_secret(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return "*" * len(raw)
    return f"{raw[:4]}***{raw[-4:]}"


def _provider_key(provider: str) -> Optional[str]:
    stored = str(_provider_saved_config(provider).get("api_key") or "").strip()
    if stored:
        return stored
    return _provider_env_key(provider)


def _provider_base_url(provider: str) -> str:
    saved = str(_provider_saved_config(provider).get("base_url") or "").strip()
    if saved:
        return saved.rstrip("/")
    config = LLM_API_CONFIG.get((provider or "").lower(), {})
    env_name = config.get("base_url_env")
    if env_name and os.getenv(env_name):
        return os.getenv(env_name).strip().rstrip("/")
    return str(config.get("default_base_url") or "").rstrip("/")


def list_llm_key_settings() -> list[dict]:
    items = []
    saved_configs = _read_llm_provider_configs()
    for owner, defaults in LLM_PROVIDER_DEFAULTS.items():
        provider = _normalize_provider(defaults.get("provider") or owner)
        config = LLM_API_CONFIG.get(provider, {})
        saved = dict(saved_configs.get(provider) or {})
        env_name = next((env_name for env_name in config.get("key_envs", []) if os.getenv(env_name)), None)
        saved_key = str(saved.get("api_key") or "").strip()
        source = "saved" if saved_key else ("env" if _provider_env_key(provider) else None)
        use_cli = bool(saved.get("use_cli", False))
        cli_cmd = str(saved.get("cli_command") or "").strip()
        cli_default = LLM_CLI_DEFAULTS.get(provider, {}).get("template", "")
        items.append({
            "owner": owner,
            "provider": provider,
            "model_name": saved.get("model_name") or defaults.get("model_name"),
            "configured": bool(saved_key or env_name or use_cli),
            "config_source": "cli" if use_cli else source,
            "has_saved_key": bool(saved_key),
            "masked_key": _mask_secret(saved_key or _provider_env_key(provider)),
            "key_env": env_name,
            "base_url": _provider_base_url(provider) if provider in LLM_API_CONFIG else None,
            "use_cli": use_cli,
            "cli_command": cli_cmd or cli_default,
            "cli_available": bool(cli_default),
            "updated_at": saved.get("updated_at"),
            "updated_by": saved.get("updated_by"),
        })
    return items


def save_llm_key_setting(provider: str, api_key: str, actor_user: Optional[dict] = None,
                         use_cli: Optional[bool] = None, cli_command: Optional[str] = None,
                         model_name: Optional[str] = None) -> dict:
    provider_name = _normalize_provider(provider)
    if provider_name not in LLM_API_CONFIG:
        raise ValueError("不支持的 LLM provider")
    configs = _read_llm_provider_configs()
    current = dict(configs.get(provider_name) or {})
    # API key (allow empty when CLI mode is used, or when only updating model_name)
    secret = str(api_key or "").strip()
    if secret:
        current["api_key"] = secret
    elif not current.get("api_key") and not use_cli and model_name is None:
        raise ValueError("API Key 不能为空")
    # Model name override
    if model_name is not None:
        current["model_name"] = str(model_name).strip()
    # CLI mode settings
    if use_cli is not None:
        current["use_cli"] = bool(use_cli)
    # Codex review 2026-04-22 [critical]: cli_command is display-only now.
    # Actual execution uses LLM_CLI_ARGV_SPECS (server-side fixed argv). We
    # refuse to store any client-supplied template to avoid misleading admins
    # into thinking they're configuring what runs. Any existing value in the
    # config file is cleared on write.
    current.pop("cli_command", None)
    if cli_command is not None and str(cli_command).strip():
        logger.warning(
            "[llm-keys] ignoring client-supplied cli_command for provider=%s "
            "(argv-based execution is fixed server-side)", provider_name,
        )
    current["updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    if actor_user:
        current["updated_by"] = (
            actor_user.get("email")
            or actor_user.get("display_name")
            or f"user:{actor_user.get('id')}"
        )
    else:
        current["updated_by"] = "system"
    configs[provider_name] = current
    _write_llm_provider_configs(configs)
    return next(item for item in list_llm_key_settings() if item["provider"] == provider_name)


def clear_llm_key_setting(provider: str) -> dict:
    provider_name = _normalize_provider(provider)
    if provider_name not in LLM_API_CONFIG:
        raise ValueError("不支持的 LLM provider")
    configs = _read_llm_provider_configs()
    configs.pop(provider_name, None)
    _write_llm_provider_configs(configs)
    return next(item for item in list_llm_key_settings() if item["provider"] == provider_name)


# ── Prompt Templates (全局唯一) ─────────────────────────────────────────────


def _extract_json_candidate(text: str) -> Optional[dict]:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


# ── Prompt version management ──────────────────────────────────────────────
# Stores user-defined prompt versions in .cache/prompt_versions.json.
# Structure:
#   {
#     "active_version": "default",
#     "versions": {
#       "default": {
#         "system_prompt": "...",
#         "user_prompt": "...",
#         "note": "...",
#         "created_at": "ISO-8601",
#         "created_by": "email/system",
#         "updated_at": "ISO-8601",
#         "updated_by": "email/system",
#         "is_builtin": true
#       },
#       "<name>": { ... }
#     }
#   }
# - "default" version is seeded from DEFAULT_LLM_SYSTEM_PROMPT and cannot be deleted.
# - _build_llm_messages fallback chain: per-call prompt_config override → active version → hardcoded default.


def _effective_model_name(provider: str, model_name: str) -> str:
    provider_name = _normalize_provider(provider)
    raw = str(model_name or "").strip()
    # If no model_name given, use saved config or global defaults
    if not raw:
        saved = _provider_saved_config(provider_name)
        raw = str(saved.get("model_name") or "").strip()
    if not raw:
        for owner, defaults in LLM_PROVIDER_DEFAULTS.items():
            if _normalize_provider(defaults.get("provider", "")) == provider_name:
                raw = defaults.get("model_name", "")
                break
    if provider_name == "gemini":
        # Normalize all Gemini display names to valid API model IDs
        lower = raw.lower().replace(" ", "-")
        if not lower or lower in {"gemini", "gemini-2.0", "gemini-2.0-pro", "gemini-pro",
                                   "gemini-3-pro", "gemini-3", "gemini-2.5"}:
            return "gemini-2.5-flash"
        if not raw.startswith("gemini-"):
            return "gemini-2.5-flash"
    if provider_name == "claude":
        lower = raw.lower()
        if lower in {"claude-sonnet", "claude", "sonnet", "claude-4", "claude-sonnet-4"}:
            return "claude-sonnet-4-6"
        if lower in {"claude-opus", "opus", "claude-opus-4"}:
            return "claude-opus-4-6"
        if lower in {"claude-haiku", "haiku"}:
            return "claude-haiku-4-5-20251001"
    return raw


# 2026-04-27: retry wrapper for LLM HTTP API calls.
# Google Gemini API is intermittently 503 (~50% during outages); OpenAI / Claude
# also occasionally 502/503/504 under load. Without a retry, a single bad
# server tick translates straight into Stage 3 ``policy_fallback`` ❌ in the
# audit table, even though the next attempt 1-3s later usually succeeds.
#
# Retry policy:
#   * Retry on 502 / 503 / 504 (transient gateway errors)
#   * Retry on requests.ConnectionError, ReadTimeout (network)
#   * NEVER retry on 4xx (auth / bad request — re-trying won't help)
#   * Default 2 retries with 1s, 2s backoff (max ~3s extra latency)
#
# Knob: env AITRADER_LLM_API_RETRIES sets the retry count (default 2).
def _post_llm_with_retry(url: str, *, headers: dict, json: dict,
                         timeout: float, provider_label: str = "llm") -> requests.Response:
    """POST with retry on transient 5xx + connection errors.

    Returns the successful Response (already raise_for_status'd).
    Raises the last exception if all retries fail.
    """
    max_retries = int(os.getenv("AITRADER_LLM_API_RETRIES", "2"))
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=json, timeout=timeout)
            # Retryable server errors
            if resp.status_code in (502, 503, 504):
                if attempt < max_retries:
                    wait = 1.0 * (2 ** attempt)  # 1s, 2s, 4s ...
                    logger.warning(
                        "[%s] HTTP %d on attempt %d/%d, retrying in %.1fs",
                        provider_label, resp.status_code,
                        attempt + 1, max_retries + 1, wait,
                    )
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.ReadTimeout) as e:
            last_exc = e
            if attempt < max_retries:
                wait = 1.0 * (2 ** attempt)
                logger.warning(
                    "[%s] %s on attempt %d/%d, retrying in %.1fs",
                    provider_label, type(e).__name__,
                    attempt + 1, max_retries + 1, wait,
                )
                time.sleep(wait)
                continue
            raise
        except requests.HTTPError as e:
            # raise_for_status raised — must be the final 5xx with no retries left
            # OR a 4xx (no retry). Either way, propagate.
            raise
    # Exhausted retries without successful return — re-raise last exception.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[{provider_label}] retry loop exited without success or exception")


def _call_openai_compatible(provider: str, model_name: str, system_prompt: str, user_prompt: str) -> tuple[Optional[dict], str, dict]:
    key = _provider_key(provider)
    if not key:
        raise RuntimeError(f"missing_api_key:{provider}")
    base_url = _provider_base_url(provider)
    path = LLM_API_CONFIG[provider]["chat_path"]
    resp = _post_llm_with_retry(
        f"{base_url}{path}",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=LLM_REQUEST_TIMEOUT_SEC,
        provider_label=provider,
    )
    data = resp.json()
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    token_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return _extract_json_candidate(text), text, token_usage


def _call_claude(model_name: str, system_prompt: str, user_prompt: str) -> tuple[Optional[dict], str, dict]:
    provider = "claude"
    key = _provider_key(provider)
    if not key:
        raise RuntimeError("missing_api_key:claude")
    resp = _post_llm_with_retry(
        _provider_base_url(provider),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "content-type": "application/json",
        },
        json={
            "model": model_name,
            "max_tokens": 200,
            "temperature": 0.2,
            "system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=LLM_REQUEST_TIMEOUT_SEC,
        provider_label="claude",
    )
    data = resp.json()
    parts = data.get("content") or []
    text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    usage = data.get("usage") or {}
    token_usage = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
    }
    return _extract_json_candidate(text), text, token_usage


def _call_gemini(model_name: str, system_prompt: str, user_prompt: str) -> tuple[Optional[dict], str, dict]:
    provider = "gemini"
    key = _provider_key(provider)
    if not key:
        raise RuntimeError("missing_api_key:gemini")
    base = _provider_base_url(provider)
    url = f"{base}/{model_name}:generateContent?key={key}"
    resp = _post_llm_with_retry(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        },
        timeout=LLM_REQUEST_TIMEOUT_SEC,
        provider_label="gemini",
    )
    data = resp.json()
    candidates = data.get("candidates") or []
    parts = ((((candidates[0] if candidates else {}).get("content") or {}).get("parts")) or [])
    text = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    usage_meta = data.get("usageMetadata") or {}
    token_usage = {
        "input_tokens": usage_meta.get("promptTokenCount", 0),
        "output_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    }
    return _extract_json_candidate(text), text, token_usage


def _call_via_api(provider: str, model_name: str, system_prompt: str,
                  user_prompt: str) -> tuple[Optional[dict], str, dict]:
    """Dispatch to the provider's HTTP API handler. Shared by normal API mode
    and by callers' CLI→API fallback path.
    """
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return _call_openai_compatible(provider, model_name, system_prompt, user_prompt)
    if provider == "claude":
        return _call_claude(model_name, system_prompt, user_prompt)
    if provider == "gemini":
        return _call_gemini(model_name, system_prompt, user_prompt)
    raise RuntimeError(f"unsupported_provider:{provider}")


def _read_daemon_status() -> dict:
    try:
        if DAEMON_STATUS_FILE.exists():
            return json.loads(DAEMON_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "running": False,
        "started_at": None,
        "last_tick_at": None,
        "last_run_at": None,
        "last_error": None,
    }


def _write_daemon_status() -> None:
    try:
        DAEMON_STATUS_FILE.write_text(
            json.dumps(_DAEMON_STATE, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("scan daemon status write failed: %s", exc)


_STRATEGY_RANK_CACHE: dict[str, tuple[str, list[dict]]] = {}
_STRATEGY_RANK_CACHE_MAX = 2000


def _strategy_cache_key(symbol: str, candidates: list[str], min_return: float, top_n: int) -> str:
    raw = f"{symbol}|{','.join(sorted(candidates))}|{min_return}|{top_n}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


#: 派进程池的门槛, 单位 = 股票数 × 候选策略数。
#: 实测(本机 12 核): 起 8 个 worker 并各 import 一次 backtest 约 1.7s,
#: 单个"股票×策略"单元实测 31.6~77.9ms (中位 38.3, 均值 41.9)。
#: 盈亏平衡点约 42~45 单元 —— 原先写 80 是按 23ms 估的, 偏保守一倍,
#: 45~80 之间本该上池子的场景被判成串行(6只×13策略实测 3.27s vs 1.96s)。
RANK_POOL_MIN_UNITS = int(os.getenv("AITRADER_RANK_POOL_MIN_UNITS", "45"))


def _rank_cache_put(key, value) -> None:
    """写排名缓存并维持上限。

    原先淘汰只写在 _rank_strategies_for_stock 里, 而进程池模式下那段在**子进程**
    执行 —— 父进程这边靠 _rank_one_for_pool 捎回来的结果直接塞, 完全绕过了
    _STRATEGY_RANK_CACHE_MAX。实测: 串行路径 5000 次写后缓存稳定在 2000 条,
    池回填路径 5000 条全留着。key 里含 last_bar_date, 每天换一批, 旧的永不回收
    —— 长跑进程 RSS 单调上涨。
    """
    if len(_STRATEGY_RANK_CACHE) >= _STRATEGY_RANK_CACHE_MAX:
        for k in list(_STRATEGY_RANK_CACHE.keys())[:_STRATEGY_RANK_CACHE_MAX // 4]:
            _STRATEGY_RANK_CACHE.pop(k, None)
    _STRATEGY_RANK_CACHE[key] = value


def _rank_one_for_pool(args):
    """进程池版入口: 顺带把 worker 里刚算好的排名缓存捎回父进程。

    _rank_strategies_for_stock 会把结果写进本进程的 _STRATEGY_RANK_CACHE,
    但那是**子进程**的内存, 父进程什么也拿不到 —— 下次再问同一只股票还得重算。
    这里把刚写进去的那条一并返回, 由父进程回填。
    """
    out = _rank_strategies_for_stock(args)
    try:
        key = _strategy_cache_key(args[0], args[2], args[3], args[4])
        return out, (key, _STRATEGY_RANK_CACHE.get(key))
    except Exception:
        return out, None


def _rank_serial(tasks):
    """串行跑, 单只股票失败不影响其它。

    旧代码(ThreadPool + as_completed)是 `try: ... except: continue`, 单只
    抛异常只跳过它自己。换成列表推导之后这个容错丢了 ——
    _rank_strategies_for_stock 里 `df.index[-1].date()` 这类是裸露的,
    一个 index 不是 DatetimeIndex 的 parquet 就会 AttributeError 一路抛到
    调用方, 把整批排名打断。
    """
    out = []
    for t in tasks:
        try:
            out.append(_rank_strategies_for_stock(t))
        except Exception as e:
            logger.warning(f"[rank] {t[0]} 失败, 跳过: {e}")
            # 以前直接 continue —— 这只股票连结果列表都进不去, 调用方
            # 只知道"它没排上", 不知道是排名过程炸了。
            out.append((t[0], t[1], [], f"rank_error:{type(e).__name__}"))
    return out


def rank_strategies_parallel(tasks, max_workers: int = 8):
    """把 _rank_strategies_for_stock 批量派出去, 返回 [(sym, market, selected), ...]。

    为什么不再用 ThreadPoolExecutor (两处调用方原先都是):
    这个函数是 CPU 密集的 —— 一次取数之后要跑 N 个策略的回测。线程池被 GIL
    串死, 实测 6 只 × 7 策略, 线程池 1.01s vs 纯串行 0.94s, **比串行还慢**
    (线程调度是净开销)。所以改成: 量小直接串行, 量大才上进程池。

    进程池里不能再起进程池 —— 嵌套会指数级放大 worker 数 (本项目出过 336 个
    worker 抢 12 核的事故), 所以先看 proc_role。
    """
    tasks = list(tasks)
    if not tasks:
        return []
    n_cands = len(tasks[0][2]) if len(tasks[0]) > 2 and tasks[0][2] else 1
    units = len(tasks) * max(1, n_cands)

    nested = False
    try:
        import proc_role
        nested = proc_role.is_pool_worker()
    except Exception:
        pass

    if nested or units < RANK_POOL_MIN_UNITS:
        return _rank_serial(tasks)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    try:
        import proc_role
        _init = proc_role.mark_pool_worker
    except Exception:
        _init = None
    n_w = max(1, min(max_workers, len(tasks)))
    out = []
    n_failed = 0
    try:
        with ProcessPoolExecutor(max_workers=n_w, initializer=_init) as ex:
            for fut in as_completed([ex.submit(_rank_one_for_pool, t) for t in tasks]):
                try:
                    res, cache_entry = fut.result()
                except Exception as e:
                    # 别静默吞: worker 被 OOM killer 干掉时同批 future 全抛
                    # BrokenProcessPool, 吞掉的话调用方拿到一份**残缺**的排名
                    # 还当成正常结果拿去给实盘池调仓, 日志里一行都没有。
                    n_failed += 1
                    if n_failed <= 3:
                        logger.warning(f"[rank] 一个分片失败: {type(e).__name__}: {e}")
                    continue   # 拿不到 future 的入参, 只能丢; 下面 n_failed 会报
                out.append(res)
                if cache_entry and cache_entry[1] is not None:
                    _rank_cache_put(cache_entry[0], cache_entry[1])
    except Exception as e:
        # 池子起不来(资源紧张/被限制)不该让整个排名失败, 退回串行。
        # 已经拿到结果的不重跑 —— 95 只 × 13 策略重来一遍是 50 秒白工。
        done_syms = {r[0] for r in out if r}
        rest = [x for x in tasks if x[0] not in done_syms]
        logger.warning(f"[rank] 进程池不可用({e}), 剩余 {len(rest)}/{len(tasks)} 退回串行")
        return out + _rank_serial(rest)
    if n_failed:
        # 让调用方能看出这批是残缺的 —— rebalance_pool 只判断"全空才退避",
        # 残缺会被当成正常结果拿去调仓。
        logger.warning(f"[rank] {n_failed}/{len(tasks)} 个分片失败, 结果不完整")
    return out


def _rank_strategies_for_stock(args):
    """Per-stock strategy ranking: run candidates on 2y data, filter by min_return, return top_n.

    返回 (symbol, market, selected, reason)。

    reason 是**为什么没排上**, 只在 selected 为空时有值。以前一律返回空数组,
    四种完全不同的情况在界面上全都显示成一个「未排」:
      · 根本取不到数据 (加密货币挂在 US 池里 / 退市 / 符号带点号)
      · 数据太短(<30 根)
      · 取数抛异常(被 except 静默吞掉)
      · 所有候选策略的年化都没到 min_annual_return_pct —— 这是**正常筛选**
    实测 MYPOOL 95 只里 13 只未排: 3 只是取不到数据, 10 只是没策略达标。
    用户没法区分"门槛卡掉的"和"数据坏了的"。
    """
    symbol, market, candidates, min_return, top_n = args
    from backtest import get_ohlcv, run_strategy_backtest
    main = _main()
    try:
        df = get_ohlcv(symbol, market, interval="1d", period="2y", local_only=False)
        if df is None or len(df) == 0:
            return symbol, market, [], "no_data"
        if len(df) < 30:
            return symbol, market, [], f"short_data:{len(df)}"
    except Exception as e:
        return symbol, market, [], f"fetch_error:{type(e).__name__}"

    last_bar_date = str(df.index[-1].date())
    cache_key = _strategy_cache_key(symbol, candidates, min_return, top_n)
    cached = _STRATEGY_RANK_CACHE.get(cache_key)
    if cached and cached[0] == last_bar_date:
        return symbol, market, cached[1], ("" if cached[1] else f"below_min_return:{min_return}")

    actual_days = (df.index[-1] - df.index[0]).days
    actual_years = max(actual_days / 365.25, 0.1)

    scores = []
    for strategy_name in candidates:
        try:
            params = main._get_standard_strategy_params(symbol, strategy_name)
            r = run_strategy_backtest(df.copy(), strategy_name, params_override=params)
            s = r.get("summary") or {}
            total_return = float(s.get("total_return", 0))
            annualized = total_return / actual_years
            if annualized < float(min_return):
                continue
            def _py(v):
                """Convert numpy types to plain Python."""
                if v is None:
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return v
            scores.append({
                "symbol": symbol,
                "market": market,
                "strategy_name": strategy_name,
                "annual_return": round(float(annualized), 2),
                "sharpe": _py(s.get("sharpe")),
                "max_drawdown": _py(s.get("max_drawdown")),
                "trade_count": int(s.get("total_trades") or 0),
                "wf_status": "standard",
                "metrics_snapshot": {
                    "total_return": round(float(total_return), 2),
                    "actual_years": round(float(actual_years), 2),
                    "annualized_return": round(float(annualized), 2),
                    "sharpe": _py(s.get("sharpe")),
                    "max_drawdown": _py(s.get("max_drawdown")),
                    "win_rate": _py(s.get("win_rate")),
                    "total_trades": int(s.get("total_trades") or 0),
                },
            })
        except Exception:
            continue

    scores.sort(key=lambda x: x["annual_return"], reverse=True)
    selected = scores[:top_n]
    for idx, item in enumerate(selected, start=1):
        item["selection_rank"] = idx
    if len(_STRATEGY_RANK_CACHE) >= _STRATEGY_RANK_CACHE_MAX:
        # Evict oldest ~25% entries
        keys = list(_STRATEGY_RANK_CACHE.keys())
        for k in keys[:len(keys) // 4]:
            _STRATEGY_RANK_CACHE.pop(k, None)
    _STRATEGY_RANK_CACHE[cache_key] = (last_bar_date, selected)
    # scores 空 = 候选策略跑完了, 但没有一个年化到得了 min_return。
    # 这是正常的业务筛选, 不是错误 —— 但必须跟"数据坏了"区分开。
    return symbol, market, selected, ("" if selected else f"below_min_return:{min_return}")


# 设置里写的 owner 别名 → 策略 owner 的规范名 (小写)。
# per_market_scan_settings.json 里一直写的是 "gpt", 而策略 owner 叫 "OpenAI"。
_OWNER_ALIASES = {"gpt": "openai", "chatgpt": "openai"}


def build_market_scan_groups(market: str, cfg: dict) -> list[dict]:
    """按市场扫描设置构建扫描组 —— 每个 owner 一组, 用该 owner 名下的全部活跃策略。

    cfg 字段: pools (股票池名列表) / owners (空 = 所有有活跃策略的 owner) /
    interval / refresh_sec / top_n / signal_side。
    返回的 group 结构直接喂给 scan_center.unified_scan。
    """
    main = _main()
    pools = [str(p).strip() for p in (cfg.get("pools") or []) if str(p).strip()]
    if not pools:
        return []
    try:
        known = main._load_pools() or {}
        pools = [p for p in pools if p in known]
    except Exception:  # noqa: BLE001
        pass
    if not pools:
        return []
    interval = str(cfg.get("interval") or "1d") or "1d"
    try:
        refresh_sec = max(30, int(cfg.get("refresh_sec") or 300))
    except (TypeError, ValueError):
        refresh_sec = 300
    try:
        top_n = max(0, int(cfg.get("top_n") or 0))
    except (TypeError, ValueError):
        top_n = 0
    signal_side = str(cfg.get("signal_side") or "both").strip().lower() or "both"
    owners_filter = {
        _OWNER_ALIASES.get(str(o).strip().lower(), str(o).strip().lower())
        for o in (cfg.get("owners") or []) if str(o).strip()
    }

    llm_registry = main._load_strategy_llm_registry()
    wh = main._load_warehouse()
    by_owner: dict[str, list[str]] = {}
    for name in (wh.get("active") or []):
        name = str(name).strip()
        if not name:
            continue
        owner = main._get_strategy_llm_owner(name, llm_registry) or ""
        by_owner.setdefault(owner, []).append(name)

    groups = []
    for owner, strategies in sorted(by_owner.items()):
        if not owner or not strategies:
            continue
        if owners_filter and owner.strip().lower() not in owners_filter:
            continue
        groups.append({
            "signature": _group_signature(interval, refresh_sec, signal_side, pools, strategies),
            "interval": interval,
            "refresh_sec": refresh_sec,
            "signal_side": signal_side,
            "pool_names": pools,
            "strategies": strategies,
            "top_n": top_n,
            "owner": owner,
            "strategy_mode": "exclusive",
            "subscribers": [],
        })
    return groups


def _per_market_run_one(market: str, cfg: dict, items: list, now_ts: float) -> None:
    """跑一个市场的一次扫描 (只到 Stage 1 出信号)。

    扫描组由该市场的设置构建 (build_market_scan_groups), 交给
    scan_center.unified_scan。in_progress 标志供 UI 进度徽章使用。
    从不抛异常 —— 外层线程池另有超时 (_PER_MARKET_SCAN_TIMEOUT_SEC)。
    """
    # Progress tracking — set in_progress on entry, clear on every exit.
    with _DAEMON_LOCK:
        _DAEMON_STATE.setdefault("per_market_in_progress", {})[market] = now_ts
    try:
        import scan_audit as _sa
        import scan_center as _sc
    except Exception as exc:  # noqa: BLE001
        logger.warning("per-market worker (%s): import failed: %r", market, exc)
        with _DAEMON_LOCK:
            _DAEMON_STATE.get("per_market_in_progress", {}).pop(market, None)
        return

    interval_override = str(cfg.get("interval") or "1d") or "1d"
    try:
        top_n_override = max(0, int(cfg.get("top_n") or 0))
    except (TypeError, ValueError):
        top_n_override = 0

    cycle_id = _sc.new_cycle_id(f"tick_{market.lower()}")

    # 扫描组直接由本市场的设置构建 (pools / owners / interval / top_n)。
    try:
        market_groups = build_market_scan_groups(market, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("per-market daemon (%s): build_market_scan_groups failed: %r", market, exc)
        market_groups = []

    # 本市场没配股票池 / 没有匹配的 owner → 也记一个空 cycle,
    # UI 上能看到这次 tick 发生过 ("0 groups")。
    if not market_groups:
        try:
            _sa.log_stage(cycle_id, "universe", {
                "symbols": [],
                "by_interval": {interval_override: []},
                "by_market": {market: []},
                "buy_pool": {market: []},
                "sell_pool": {},
                "per_market_daemon": True,
                "market": market,
            })
            _sa.log_stage(cycle_id, "scan_summary", {
                "bucket": f"{market}/{interval_override}/both",
                "interval": interval_override,
                "side": "both",
                "market": market,
                "path": "no_market_groups",
                "groups": 0,
                "universe_size": 0,
                "strategies_count": 0,
                "alerts_emitted": 0,
                "error": None,
            })
        except Exception:  # noqa: BLE001
            pass
        with _DAEMON_LOCK:
            _DAEMON_STATE.setdefault("per_market_last_run", {})[market] = now_ts
            _DAEMON_STATE.get("per_market_in_progress", {}).pop(market, None)
        return

    # 只跑扫描: unified_scan 产出 Stage 0 universe / Stage 1 raw_signal /
    # scan_summary 审计记录。不做 LLM 决策和下单。
    try:
        scan_center._scan_caches  # ensure module is initialized
    except AttributeError:
        pass
    try:
        scan_center.unified_scan(
            market_groups, scan_center._scan_caches, cycle_id=cycle_id,
        )
        with _DAEMON_LOCK:
            _DAEMON_STATE.setdefault("per_market_last_run", {})[market] = now_ts
            _DAEMON_STATE.setdefault("per_market_last_cycle", {})[market] = cycle_id
        logger.info(
            "per-market daemon: %s cycle=%s interval=%s top_n=%s groups=%d "
            "(scan-only)",
            market, cycle_id, interval_override, top_n_override, len(market_groups),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("per-market daemon: pipeline failed for %s: %r", market, exc)
        try:
            _sa.log_stage(cycle_id, "scan_summary", {
                "bucket": f"{market}/{interval_override}/both",
                "interval": interval_override,
                "side": "both",
                "market": market,
                "path": "per_market_daemon_exception",
                "groups": len(market_groups),
                "universe_size": 0,
                "strategies_count": 0,
                "alerts_emitted": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception:  # noqa: BLE001
            pass
        with _DAEMON_LOCK:
            _DAEMON_STATE.setdefault("per_market_last_run", {})[market] = now_ts
    finally:
        with _DAEMON_LOCK:
            _DAEMON_STATE.get("per_market_in_progress", {}).pop(market, None)


# Per-market SCAN-ONLY timeout. Now that LLM is decoupled (per-market
# daemon emits only Stage 0/1, no _execute_auto_trade_for_signal),
# scans should finish in <60s for any market. We give 90s as the hard
# cap — anything longer is a yfinance/akshare network hang and we want
# to release the slot quickly so the next refresh_sec tick can fire.
_PER_MARKET_SCAN_TIMEOUT_SEC = 90.0
# Process-wide lock to prevent overlapping per-market ticks. If the
# previous tick is still running when the next one arrives (15s later),
# we just skip — the ongoing tick will eventually update last_run.
_PER_MARKET_TICK_LOCK = threading.Lock()


def _per_market_daemon_tick(now_ts: float) -> None:
    """2026-04-25 (revised 2026-04-26): per-market scheduling layer.

    Reads ``per_market_scan_settings.json`` and fires a market-scoped
    realtime scan for each enabled market whose ``refresh_sec`` has
    elapsed since its last per-market run. **Per-market scans run
    concurrently in a thread pool with hard timeouts** so a slow market
    (e.g., HK akshare fallback) can't wedge the entire daemon.

    Critical bug history:
      * 2026-04-25 22:01: HK scan blocked indefinitely → serial loop
        wedged → CRYPTO/UK never ran → daemon silent for 8.7 hours.
        Fix: thread pool + per-future timeout (this revision).

    State (per-market last-run timestamps) lives in
    ``_DAEMON_STATE['per_market_last_run']``. Cycle IDs use
    ``tick_<market>`` (underscore) to keep the canonical 3-part shape.

    NEVER raises; the outer ``_daemon_loop`` try/except is belt-and-braces.
    """
    # Skip-if-busy: prevents stacking ticks if the prior one is still
    # running (~5min refresh_sec but we tick every 15s; we don't want
    # 20 concurrent ticks competing for the thread pool).
    if not _PER_MARKET_TICK_LOCK.acquire(blocking=False):
        logger.debug("per-market daemon tick: previous still running, skipping")
        return
    try:
        try:
            from main import (
                load_per_market_settings,
                _resolve_scan_universe as _resolve,
            )
            import scan_audit as _sa  # noqa: F401 — used by worker
            import scan_center as _sc  # noqa: F401 — used by worker
        except Exception as exc:  # noqa: BLE001
            logger.debug("per-market daemon: main not ready (%s)", exc)
            return

        settings = load_per_market_settings()
        if not settings:
            return
        try:
            universe_all = _resolve(pool_names=[], all_pools=True, scope="pool") or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("per-market daemon: universe resolve failed: %r", exc)
            return
        universe_by_market: "dict[str, list]" = {}
        for it in universe_all:
            m = str(it.get("market") or "").upper()
            if m:
                universe_by_market.setdefault(m, []).append(it)

        with _DAEMON_LOCK:
            per_mkt_last = dict(_DAEMON_STATE.get("per_market_last_run") or {})

        # Decide which markets are due.
        due: "list[tuple[str, dict, list]]" = []
        for market, cfg in settings.items():
            if not cfg.get("enabled", True):
                continue
            try:
                refresh_sec = max(30, int(cfg.get("refresh_sec") or 300))
            except (TypeError, ValueError):
                refresh_sec = 300
            last_ts = float(per_mkt_last.get(market, 0) or 0)
            if (now_ts - last_ts) < refresh_sec:
                continue
            items = universe_by_market.get(market, [])
            due.append((market, cfg, items))

        if not due:
            return

        # Launch all due markets in parallel. Hard per-future timeout
        # bounds tail latency. ThreadPoolExecutor not ProcessPool because
        # _run_realtime_scan already manages its own ProcessPool internally
        # and we just want IO-parallelism here.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        n_workers = max(1, min(len(due), 5))
        executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="per-market")
        future_to_market: dict = {}
        try:
            for market, cfg, items in due:
                fut = executor.submit(_per_market_run_one, market, cfg, items, now_ts)
                future_to_market[fut] = market
            # Per-future wait with timeout. We don't use as_completed because
            # we want each market to be bounded individually — a 5min HK
            # hang shouldn't push CRYPTO past its budget.
            for fut, market in future_to_market.items():
                try:
                    fut.result(timeout=_PER_MARKET_SCAN_TIMEOUT_SEC)
                except FuturesTimeout:
                    logger.warning(
                        "per-market daemon: %s scan exceeded %ds — bumping last_run "
                        "to release the slot (next tick will retry)",
                        market, int(_PER_MARKET_SCAN_TIMEOUT_SEC),
                    )
                    with _DAEMON_LOCK:
                        _DAEMON_STATE.setdefault("per_market_last_run", {})[market] = now_ts
                    # The future may still be running; we can't kill it, but
                    # ThreadPoolExecutor.shutdown(wait=False) below stops new
                    # submissions and the orphaned thread will eventually
                    # finish or crash on its own. We move on.
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "per-market daemon: %s worker raised: %r", market, exc,
                    )
                    with _DAEMON_LOCK:
                        _DAEMON_STATE.setdefault("per_market_last_run", {})[market] = now_ts
        finally:
            # Don't wait for orphans — shutdown(wait=False) lets background
            # threads finish on their own without blocking the daemon.
            executor.shutdown(wait=False)
    finally:
        _PER_MARKET_TICK_LOCK.release()

def _daemon_loop() -> None:
    """扫描守护线程: 每 15s 检查一次, 到期的市场跑一次扫描 (见 _per_market_daemon_tick)。"""
    with _DAEMON_LOCK:
        _DAEMON_STATE["running"] = True
        _DAEMON_STATE["started_at"] = _DAEMON_STATE.get("started_at") or _utc_iso()
        _DAEMON_STATE["last_error"] = None
        _write_daemon_status()
    while True:
        try:
            with _DAEMON_LOCK:
                _DAEMON_STATE["last_tick_at"] = _utc_iso()
            _per_market_daemon_tick(time.time())
            with _DAEMON_LOCK:
                _DAEMON_STATE["last_error"] = None
                _write_daemon_status()
        except Exception as exc:
            logger.warning("scan daemon tick failed: %s", exc)
            with _DAEMON_LOCK:
                _DAEMON_STATE["last_error"] = str(exc)
                _write_daemon_status()
        time.sleep(15)


def start_scan_daemon() -> bool:
    global _DAEMON_THREAD
    with _DAEMON_LOCK:
        if _DAEMON_THREAD and _DAEMON_THREAD.is_alive():
            return False
        existing = _read_daemon_status()
        _DAEMON_STATE.update(existing)
        _DAEMON_THREAD = threading.Thread(
            target=_daemon_loop,
            name="scan-daemon",
            daemon=True,
        )
        _DAEMON_THREAD.start()
        return True

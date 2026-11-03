#!/bin/bash
# 9000: 全量 dev stack (main:app) — 策略开发 / 优化 / LLM 评估
cd "$(dirname "$0")"
set -a; source .env; set +a
exec venv/bin/uvicorn main:app --host 127.0.0.1 --port 9000 --workers 1 --log-level info

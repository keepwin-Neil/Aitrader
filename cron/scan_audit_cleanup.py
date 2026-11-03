#!/usr/bin/env python3
"""Nightly retention: delete scan_audit per-date dirs older than 7 days.

Scan Center v0.5 §7.5 / P12 — partner to ``scan_audit._audit_flusher_loop``
which produces ``.cache/scan_audit/<YYYY-MM-DD>/<cycle_id>.jsonl``
segment files.

Environment:
  * ``SCAN_AUDIT_DIR``            — audit root (default ``.cache/scan_audit``)
  * ``SCAN_AUDIT_RETENTION_DAYS`` — days to keep (default 7)

Suggested crontab (nightly 04:15):
    15 4 * * * /usr/local/bin/python3 \
      /path/to/aitrader/cron/scan_audit_cleanup.py \
      >> /var/log/scan_audit_cleanup.log 2>&1
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

_RETENTION_DAYS = int(os.environ.get("SCAN_AUDIT_RETENTION_DAYS", "7"))
_AUDIT_DIR = Path(os.environ.get("SCAN_AUDIT_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "scan_audit")))


def main() -> int:
    """Delete all ``_AUDIT_DIR/<YYYY-MM-DD>/`` subdirs older than
    ``_RETENTION_DAYS``. Returns the count of directories deleted.
    """
    if not _AUDIT_DIR.exists():
        return 0
    cutoff = datetime.now() - timedelta(days=_RETENTION_DAYS)
    deleted = 0
    try:
        entries = list(_AUDIT_DIR.iterdir())
    except Exception as exc:  # noqa: BLE001
        logging.warning("scan_audit_cleanup: iterdir failed on %s: %s",
                        _AUDIT_DIR, exc)
        return 0
    for day_dir in entries:
        if not day_dir.is_dir():
            continue
        try:
            day = datetime.strptime(day_dir.name, "%Y-%m-%d")
        except ValueError:
            # Not a date-named dir — skip.
            continue
        if day < cutoff:
            try:
                shutil.rmtree(day_dir)
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                logging.warning("scan_audit_cleanup: rmtree %s failed: %s",
                                day_dir, exc)
    return deleted


if __name__ == "__main__":
    n = main()
    print(
        f"scan_audit_cleanup: deleted {n} day-dir(s) older than "
        f"{_RETENTION_DAYS}d from {_AUDIT_DIR}"
    )
    sys.exit(0)

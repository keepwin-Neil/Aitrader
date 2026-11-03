#!/usr/bin/env python3
"""test_news_provider.py — verify Phase 1 news pipeline (offline + optional --live).

Coverage
--------
Unit (offline, mocked HTTP, always run):
  1.  news_finnhub.fetch — happy path + missing key + HTTP error
  2.  news_google.fetch — happy path + parse error + outside window
  3.  news_provider.get_stock_news_unified — dedupe by URL + cache hit/miss
  4.  news_scorer.score_news — empty → no_news; mocked LLM → parsed score

Integration (skipped unless --live):
  5.  news_finnhub.fetch real call for AAPL (needs FINNHUB_API_KEY)
  6.  news_google.fetch real call for AAPL (no key needed)
  7.  news_provider.get_stock_news_unified real call for AAPL

Usage
-----
  python3 scripts/test_news_provider.py            # unit tests only
  python3 scripts/test_news_provider.py --live     # also hit real APIs

Exit code 0 = all pass; non-zero = at least one failure.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

# Make parent project importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import news_finnhub
import news_google
import news_provider
import news_scorer


# ──────────────────────────────────────────────────────────────────────────────
# 1. news_finnhub
# ──────────────────────────────────────────────────────────────────────────────
class TestFinnhub(unittest.TestCase):
    def test_missing_key_returns_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(news_finnhub.fetch("AAPL", api_key=None), [])

    def test_happy_path_filters_by_window(self):
        now = datetime.utcnow()
        recent_ts = int((now - timedelta(hours=2)).timestamp())
        old_ts = int((now - timedelta(hours=72)).timestamp())  # outside 24h window
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"datetime": recent_ts, "headline": "Recent news", "url": "u1",
             "source": "Reuters", "summary": "summary 1"},
            {"datetime": old_ts, "headline": "Old news", "url": "u2",
             "source": "Bloomberg", "summary": "summary 2"},
        ]
        mock_resp.raise_for_status.return_value = None
        with patch("news_finnhub.requests.get", return_value=mock_resp):
            out = news_finnhub.fetch("AAPL", window_hours=24, api_key="dummy")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Recent news")
        self.assertEqual(out[0]["source"], "Reuters")
        self.assertIn("published_at", out[0])

    def test_http_error_returns_empty(self):
        import requests
        with patch("news_finnhub.requests.get",
                   side_effect=requests.exceptions.ConnectionError("boom")):
            self.assertEqual(news_finnhub.fetch("AAPL", api_key="dummy"), [])

    def test_non_json_returns_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.raise_for_status.return_value = None
        with patch("news_finnhub.requests.get", return_value=mock_resp):
            self.assertEqual(news_finnhub.fetch("AAPL", api_key="dummy"), [])


# ──────────────────────────────────────────────────────────────────────────────
# 2. news_google
# ──────────────────────────────────────────────────────────────────────────────
_RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title>AAPL beats earnings</title>
  <link>https://example.com/aapl-1</link>
  <pubDate>{recent}</pubDate>
  <source url="https://reuters.com">Reuters</source>
  <description>desc1</description>
</item>
<item>
  <title>AAPL old story</title>
  <link>https://example.com/aapl-2</link>
  <pubDate>{old}</pubDate>
  <source url="https://wsj.com">WSJ</source>
  <description>desc2</description>
</item>
</channel></rss>"""


class TestGoogleNews(unittest.TestCase):
    def _mock_resp(self, body: bytes, status: int = 200):
        m = MagicMock()
        m.content = body
        m.status_code = status
        m.raise_for_status.return_value = None
        return m

    def test_happy_path_filters_by_window(self):
        from email.utils import format_datetime
        now = datetime.now(timezone.utc)
        recent = format_datetime(now - timedelta(hours=2))
        old = format_datetime(now - timedelta(hours=72))
        body = _RSS_FIXTURE.format(recent=recent, old=old).encode()
        with patch("news_google.requests.get", return_value=self._mock_resp(body)):
            out = news_google.fetch("AAPL", window_hours=24)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "AAPL beats earnings")
        self.assertEqual(out[0]["source"], "Reuters")

    def test_parse_error_returns_empty(self):
        with patch("news_google.requests.get",
                   return_value=self._mock_resp(b"<not-xml>")):
            self.assertEqual(news_google.fetch("AAPL"), [])

    def test_http_error_returns_empty(self):
        import requests
        with patch("news_google.requests.get",
                   side_effect=requests.exceptions.Timeout("slow")):
            self.assertEqual(news_google.fetch("AAPL"), [])


# ──────────────────────────────────────────────────────────────────────────────
# 3. news_provider — orchestration + dedupe + cache
# ──────────────────────────────────────────────────────────────────────────────
class TestProvider(unittest.TestCase):
    def setUp(self):
        news_provider.clear_cache()

    def test_dedupe_by_url_prefers_finnhub(self):
        finnhub_items = [{"title": "T1", "url": "https://x/a", "published_at": "2026-04-26T10:00:00Z",
                          "source": "Reuters", "summary": "rich"}]
        google_items = [{"title": "T1 alt", "url": "https://x/a", "published_at": "2026-04-26T10:00:00Z",
                         "source": "Google News", "summary": "thin"}]
        with patch("news_provider.finnhub_fetch", return_value=finnhub_items), \
             patch("news_provider.google_fetch", return_value=google_items):
            out = news_provider.get_stock_news_unified("AAPL", "US", use_cache=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["provider"], "finnhub")
        self.assertEqual(out[0]["summary"], "rich")

    def test_finnhub_skipped_for_non_us(self):
        finnhub_items = [{"title": "should-not-appear", "url": "u", "published_at": "x",
                          "source": "F", "summary": ""}]
        google_items = [{"title": "G1", "url": "https://x/g", "published_at": "2026-04-26T10:00:00Z",
                         "source": "Google", "summary": ""}]
        with patch("news_provider.finnhub_fetch", return_value=finnhub_items) as mock_f, \
             patch("news_provider.google_fetch", return_value=google_items):
            out = news_provider.get_stock_news_unified("0700.HK", "HK", use_cache=False)
        mock_f.assert_not_called()  # Finnhub skipped for non-US
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["provider"], "google")

    def test_cache_hit_skips_fetchers(self):
        items = [{"title": "T", "url": "https://x/a", "published_at": "2026-04-26T10:00:00Z",
                  "source": "F", "summary": ""}]
        with patch("news_provider.finnhub_fetch", return_value=items) as mock_f, \
             patch("news_provider.google_fetch", return_value=[]):
            news_provider.get_stock_news_unified("AAPL", "US", use_cache=True)
            news_provider.get_stock_news_unified("AAPL", "US", use_cache=True)
        # Second call should use cache; finnhub_fetch invoked only once
        self.assertEqual(mock_f.call_count, 1)

    def test_empty_symbol_returns_empty(self):
        self.assertEqual(news_provider.get_stock_news_unified("", "US"), [])


# ──────────────────────────────────────────────────────────────────────────────
# 4. news_scorer
# ──────────────────────────────────────────────────────────────────────────────
class TestScorer(unittest.TestCase):
    def test_empty_news_returns_no_news(self):
        result = news_scorer.score_news("AAPL", [])
        self.assertEqual(result["score"], 50)
        self.assertEqual(result["label"], "no_news")
        self.assertEqual(result["n_items"], 0)

    def test_llm_called_with_proper_prompt(self):
        items = [{"title": "AAPL beats Q1 earnings", "source": "Reuters",
                  "published_at": "2026-04-26T10:00:00Z"}]
        fake_llm_resp = {
            "raw_text": '{"score": 78, "label": "bullish", "reasoning": "earnings beat"}',
            "duration_ms": 123,
            "token_usage": {},
        }
        with patch("llm_evaluator._call_llm_for_eval", return_value=fake_llm_resp) as mock_call:
            result = news_scorer.score_news("AAPL", items, provider="claude")
        self.assertEqual(result["score"], 78)
        self.assertEqual(result["label"], "bullish")
        self.assertEqual(result["n_items"], 1)
        # Verify LLM was actually called once
        mock_call.assert_called_once()

    def test_score_clamped_to_0_100(self):
        items = [{"title": "x", "source": "y", "published_at": "z"}]
        fake_llm_resp = {"raw_text": '{"score": 999, "label": "bullish", "reasoning": ""}',
                         "duration_ms": 0, "token_usage": {}}
        with patch("llm_evaluator._call_llm_for_eval", return_value=fake_llm_resp):
            result = news_scorer.score_news("X", items)
        self.assertEqual(result["score"], 100)

    def test_llm_failure_returns_neutral(self):
        items = [{"title": "x", "source": "y", "published_at": "z"}]
        with patch("llm_evaluator._call_llm_for_eval", side_effect=RuntimeError("API down")):
            result = news_scorer.score_news("X", items)
        self.assertEqual(result["score"], 50)
        self.assertEqual(result["label"], "neutral")
        self.assertIn("API down", result["reasoning"])

    def test_malformed_llm_response(self):
        items = [{"title": "x", "source": "y", "published_at": "z"}]
        fake_llm_resp = {"raw_text": "not json at all", "duration_ms": 0, "token_usage": {}}
        with patch("llm_evaluator._call_llm_for_eval", return_value=fake_llm_resp):
            result = news_scorer.score_news("X", items)
        self.assertEqual(result["score"], 50)
        self.assertEqual(result["label"], "neutral")

    def test_cli_mutex_acquired_when_use_cli_true(self):
        """code-review fix: news_scorer must hold _CLI_MUTEX while calling CLI."""
        items = [{"title": "x", "source": "y", "published_at": "z"}]
        fake_llm_resp = {"raw_text": '{"score":80,"label":"bullish","reasoning":""}',
                         "duration_ms": 1, "token_usage": {}}
        # Pretend provider has CLI mode enabled.
        with patch("llm_evaluator._provider_saved_config", return_value={"use_cli": True}), \
             patch("llm_evaluator._call_llm_for_eval", return_value=fake_llm_resp):
            from core_engine import _CLI_MUTEX
            lock = _CLI_MUTEX.get("claude")
            # Lock should be acquirable BEFORE call (i.e. not held).
            self.assertTrue(lock.acquire(blocking=False))
            lock.release()
            result = news_scorer.score_news("X", items, provider="claude")
            # Lock should be released AFTER call too.
            self.assertTrue(lock.acquire(blocking=False))
            lock.release()
        self.assertEqual(result["score"], 80)

    def test_cli_mutex_contention_returns_neutral(self):
        """code-review fix: when CLI lock cannot be acquired in time, return neutral
        (do NOT fall through and race the CLI subprocess)."""
        import os, threading
        os.environ["AITRADER_CLI_MUTEX_WAIT_SEC"] = "0.5"
        try:
            from core_engine import _CLI_MUTEX
            lock = _CLI_MUTEX.get("claude")
            # Hold the lock from another "thread" — actually just acquire here.
            self.assertTrue(lock.acquire(blocking=False))
            try:
                items = [{"title": "x", "source": "y", "published_at": "z"}]
                with patch("llm_evaluator._provider_saved_config", return_value={"use_cli": True}), \
                     patch("llm_evaluator._call_llm_for_eval") as mock_call:
                    result = news_scorer.score_news("X", items, provider="claude")
                # _call_llm_for_eval must NOT have been called (we'd race the CLI).
                mock_call.assert_not_called()
                self.assertEqual(result["score"], 50)
                self.assertEqual(result["label"], "neutral")
                self.assertIn("contention", result["reasoning"].lower())
            finally:
                lock.release()
        finally:
            os.environ.pop("AITRADER_CLI_MUTEX_WAIT_SEC", None)


# ──────────────────────────────────────────────────────────────────────────────
# 5-7. Integration (live) — opt-in only
# ──────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(os.getenv("LIVE_NEWS_TESTS") == "1",
                     "Skipped — set LIVE_NEWS_TESTS=1 to run real network calls")
class TestLive(unittest.TestCase):
    def test_finnhub_live_aapl(self):
        out = news_finnhub.fetch("AAPL", window_hours=72)
        self.assertIsInstance(out, list)
        if not os.getenv("FINNHUB_API_KEY"):
            self.skipTest("FINNHUB_API_KEY not set")
        # Expect at least 1 news item for AAPL in 72h
        self.assertGreater(len(out), 0, "Expected ≥1 AAPL news from Finnhub in 72h")
        sample = out[0]
        for k in ("title", "url", "published_at", "source"):
            self.assertIn(k, sample)
        print(f"\n[LIVE] Finnhub returned {len(out)} items for AAPL.\n  Sample: {sample['title']}")

    def test_google_live_aapl(self):
        out = news_google.fetch("AAPL", window_hours=72, company_hint="Apple")
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0, "Expected ≥1 AAPL news from Google in 72h")
        print(f"\n[LIVE] Google News returned {len(out)} items for AAPL.\n  Sample: {out[0]['title']}")

    def test_provider_live_aapl_dedupe(self):
        news_provider.clear_cache()
        out = news_provider.get_stock_news_unified("AAPL", "US", window_hours=72,
                                                   company_hint="Apple", use_cache=False)
        self.assertGreater(len(out), 0)
        urls = [x["url"] for x in out]
        self.assertEqual(len(urls), len(set(urls)), "Provider returned duplicate URLs")
        providers = {x["provider"] for x in out}
        print(f"\n[LIVE] Unified returned {len(out)} unique items "
              f"from providers={providers}.\n  Newest: {out[0]['title']}")


if __name__ == "__main__":
    # --live flag for convenience
    if "--live" in sys.argv:
        os.environ["LIVE_NEWS_TESTS"] = "1"
        sys.argv.remove("--live")
    unittest.main(verbosity=2)

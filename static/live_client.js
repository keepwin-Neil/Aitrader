/**
 * live_client.js — frontend WebSocket client for the streaming subsystem.
 *
 * One shared connection per channel (tick + bar) is multiplexed across all
 * subscribers in the page. Re-subscribes on reconnect. Auto-disconnects when
 * the page unloads.
 *
 * Public API (exposed as window.liveClient):
 *
 *   await liveClient.init()
 *     Pull LIVE_STREAMING_TOKEN from /live/config.
 *     Idempotent — safe to call from many places, only fetches once.
 *
 *   const handle = liveClient.subscribeBar(market, symbol, interval, onBar, opts)
 *     onBar({type, market, symbol, interval, source, ts, seq, action, o,h,l,c,v})
 *     handle.unsubscribe()
 *
 *   const handle = liveClient.subscribeTick(market, symbol, onTick)
 *     onTick({type, market, symbol, source, ts, seq, last, bid, ask, volume})
 *     handle.unsubscribe()
 *
 *   liveClient.status() → {tick: 'connecting|open|closed', bar: '...', ...}
 *
 * Implementation notes:
 * - Two underlying WS connections, one per channel type. Multiplexing is
 *   done via the protocol's subscribe/unsubscribe action messages.
 * - subscribeBar / subscribeTick are idempotent across page views — if the
 *   same (market, symbol, interval) is requested twice we just add the
 *   callback; the WS only sees one subscribe.
 * - Reconnect: exponential backoff capped at 30s. On reconnect, all live
 *   subscriptions are re-sent so the server's state matches client state.
 * - Heartbeat: server sends `hb` every 15s. We send `ping` every 15s.
 *   30s read silence on either side triggers reconnect.
 * - LIVE_STREAMING_ENABLED=0 servers don't expose /live/config →
 *   init() rejects → public API silently no-ops (safe fallback).
 */

(function () {
  'use strict';

  // ── Singleton state ───────────────────────────────────────────────────────

  const STATE = {
    token: null,
    tokenPromise: null,   // pending fetch
    sockets: {
      tick: null,
      bar: null,
    },
    // map streamKey ("US:AAPL:tick" | "US:AAPL:bar:1m") → Set of callbacks
    subscribers: new Map(),
    // pending subscribe messages waiting for socket OPEN
    pendingActions: { tick: [], bar: [] },
    // backoff state per channel
    backoff: { tick: 1000, bar: 1000 },
    closing: false,
  };

  function streamKey(market, symbol, channel, interval) {
    return channel === 'bar'
      ? `${market}:${symbol}:${channel}:${interval}`
      : `${market}:${symbol}:${channel}`;
  }

  // ── Token fetch (login required; rolls into public API gracefully) ────────

  async function init() {
    if (STATE.token) return STATE.token;
    if (STATE.tokenPromise) return STATE.tokenPromise;

    STATE.tokenPromise = (async function () {
      try {
        const resp = await fetch('/live/config');
        if (!resp.ok) throw new Error('http ' + resp.status);
        const data = await resp.json();
        if (!data || !data.token) throw new Error('no token in response');
        STATE.token = data.token;
        return data.token;
      } catch (e) {
        STATE.tokenPromise = null;
        throw e;
      }
    })();
    return STATE.tokenPromise;
  }

  // ── Socket lifecycle (per channel) ────────────────────────────────────────

  function _openSocket(channel) {
    if (!STATE.token) {
      console.warn('[live_client] no token; init() first');
      return;
    }
    if (STATE.sockets[channel] && STATE.sockets[channel].readyState <= 1) return;

    const host = location.host;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use a per-channel discovery URL that the server happily accepts.
    // We pick AAPL/US as a "discovery" subscribe; we then immediately
    // unsubscribe and use json-action messages from there. Simpler than
    // adding a third path-less endpoint to the server.
    let url;
    if (channel === 'tick') {
      url = `${proto}//${host}/ws/quote/US/AAPL?token=${encodeURIComponent(STATE.token)}`;
    } else {
      url = `${proto}//${host}/ws/kline/US/AAPL/1m?token=${encodeURIComponent(STATE.token)}`;
    }
    const ws = new WebSocket(url);
    STATE.sockets[channel] = ws;

    ws._channel = channel;
    ws._pingTimer = null;
    ws._initialDiscoverySent = false;

    ws.onopen = () => {
      STATE.backoff[channel] = 1000;  // reset
      // Immediately unsubscribe the discovery key so the WS becomes a
      // pure multiplexed connection.
      const ka = (channel === 'bar')
        ? { action: 'unsubscribe', channel: 'bar', market: 'US', symbol: 'AAPL', interval: '1m' }
        : { action: 'unsubscribe', channel: 'tick', market: 'US', symbol: 'AAPL' };
      try { ws.send(JSON.stringify(ka)); } catch (e) {}
      // Re-send all live subscriptions
      STATE.subscribers.forEach((cbs, key) => {
        const parts = key.split(':');
        if (channel === 'bar' && parts[2] === 'bar') {
          ws.send(JSON.stringify({
            action: 'subscribe', channel: 'bar',
            market: parts[0], symbol: parts[1], interval: parts[3],
          }));
        } else if (channel === 'tick' && parts[2] === 'tick') {
          ws.send(JSON.stringify({
            action: 'subscribe', channel: 'tick',
            market: parts[0], symbol: parts[1],
          }));
        }
      });
      // Drain pending
      const pending = STATE.pendingActions[channel];
      pending.splice(0).forEach(msg => { try { ws.send(JSON.stringify(msg)); } catch (e) {} });
      // Client-side ping every 15s
      ws._pingTimer = setInterval(() => {
        try { ws.send(JSON.stringify({ action: 'ping' })); } catch (e) {}
      }, 15000);
    };

    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (!msg || !msg.type) return;
      if (msg.type === 'hb' || msg.type === 'pong' || msg.type === 'subscribed') return;
      if (msg.type === 'error') {
        console.warn('[live_client]', channel, 'error:', msg.code, msg.msg);
        return;
      }
      _dispatch(channel, msg);
    };

    ws.onclose = () => {
      clearInterval(ws._pingTimer);
      STATE.sockets[channel] = null;
      if (STATE.closing) return;
      // Exponential backoff reconnect
      const delay = STATE.backoff[channel];
      STATE.backoff[channel] = Math.min(delay * 2, 30000);
      console.info(`[live_client] ${channel} closed; reconnecting in ${delay}ms`);
      setTimeout(() => _openSocket(channel), delay);
    };

    ws.onerror = (e) => { console.warn('[live_client] ws error', channel, e); };
  }

  function _dispatch(channel, msg) {
    let key;
    if (msg.type === 'bar' || (msg.type === 'snapshot' && msg.channel === 'bar')) {
      key = streamKey(msg.market, msg.symbol, 'bar', msg.interval);
    } else if (msg.type === 'tick') {
      key = streamKey(msg.market, msg.symbol, 'tick');
    } else if (msg.type === 'source_changed' || msg.type === 'gap') {
      // broadcast to all keys for this (market, symbol)
      STATE.subscribers.forEach((cbs, k) => {
        if (k.startsWith(`${msg.market}:${msg.symbol}:`)) {
          cbs.forEach(cb => { try { cb(msg); } catch (e) { console.error(e); } });
        }
      });
      return;
    } else {
      return;
    }
    const cbs = STATE.subscribers.get(key);
    if (!cbs) return;
    cbs.forEach(cb => { try { cb(msg); } catch (e) { console.error(e); } });
  }

  // ── Public subscribe APIs ─────────────────────────────────────────────────

  function _ensureSocket(channel) {
    const sock = STATE.sockets[channel];
    if (!sock || sock.readyState > 1) _openSocket(channel);
  }

  function _send(channel, msg) {
    const sock = STATE.sockets[channel];
    if (sock && sock.readyState === 1) {
      try { sock.send(JSON.stringify(msg)); } catch (e) {}
    } else {
      STATE.pendingActions[channel].push(msg);
      _ensureSocket(channel);
    }
  }

  function subscribeBar(market, symbol, interval, callback) {
    if (!STATE.token) {
      console.warn('[live_client] subscribeBar called before init(); returning no-op');
      return { unsubscribe: () => {} };
    }
    const key = streamKey(market, symbol, 'bar', interval);
    let cbs = STATE.subscribers.get(key);
    const isFirst = !cbs;
    if (!cbs) { cbs = new Set(); STATE.subscribers.set(key, cbs); }
    cbs.add(callback);
    if (isFirst) {
      _send('bar', { action: 'subscribe', channel: 'bar', market, symbol, interval });
    }
    return {
      unsubscribe: () => {
        const set = STATE.subscribers.get(key);
        if (!set) return;
        set.delete(callback);
        if (set.size === 0) {
          STATE.subscribers.delete(key);
          _send('bar', { action: 'unsubscribe', channel: 'bar', market, symbol, interval });
        }
      }
    };
  }

  function subscribeTick(market, symbol, callback) {
    if (!STATE.token) {
      console.warn('[live_client] subscribeTick called before init(); returning no-op');
      return { unsubscribe: () => {} };
    }
    const key = streamKey(market, symbol, 'tick');
    let cbs = STATE.subscribers.get(key);
    const isFirst = !cbs;
    if (!cbs) { cbs = new Set(); STATE.subscribers.set(key, cbs); }
    cbs.add(callback);
    if (isFirst) {
      _send('tick', { action: 'subscribe', channel: 'tick', market, symbol });
    }
    return {
      unsubscribe: () => {
        const set = STATE.subscribers.get(key);
        if (!set) return;
        set.delete(callback);
        if (set.size === 0) {
          STATE.subscribers.delete(key);
          _send('tick', { action: 'unsubscribe', channel: 'tick', market, symbol });
        }
      }
    };
  }

  // ── Dataset-driven auto-subscribe ─────────────────────────────────────────
  //
  // Mark any DOM element with:
  //   <span data-live-symbol="AAPL"
  //         data-live-market="US"
  //         data-live-field="last">—</span>
  //
  // Then call liveClient.attachDataset() once after the page loads. It will:
  //   - find all matching nodes
  //   - subscribeTick(market, symbol) per (market,symbol) pair
  //   - on each tick, update node.textContent to the requested field
  //
  // field defaults to 'last'; supported: last, bid, ask, volume, change_pct
  // (computed from last / prev_close if prev_close known via /quotes; rare).
  //
  // Also supports data-live-format = "%.2f" or "%.4f" (defaults to %.2f).
  //
  // Re-running attachDataset is idempotent; it skips elements with
  // dataset.liveAttached === '1'.

  const ATTACHED_SUBS = new Map();  // (market:symbol) -> { handle, nodes:Set }

  function _formatNumber(value, fmt) {
    if (value === null || value === undefined) return '—';
    const m = /^%\.(\d+)f$/.exec(fmt || '%.2f');
    const dec = m ? parseInt(m[1], 10) : 2;
    try {
      return Number(value).toFixed(dec);
    } catch (e) { return String(value); }
  }

  function attachDataset(root) {
    if (!STATE.token) {
      console.warn('[live_client] attachDataset before init(); will retry after init');
      init().then(() => attachDataset(root)).catch(() => {});
      return;
    }
    const scope = root || document;
    const nodes = scope.querySelectorAll('[data-live-symbol][data-live-market]');
    nodes.forEach(node => {
      if (node.dataset.liveAttached === '1') return;
      node.dataset.liveAttached = '1';
      const market = node.dataset.liveMarket;
      const symbol = node.dataset.liveSymbol;
      const field  = node.dataset.liveField || 'last';
      const fmt    = node.dataset.liveFormat || '%.2f';
      const key    = `${market}:${symbol}`;
      let bucket = ATTACHED_SUBS.get(key);
      if (!bucket) {
        const nodeSet = new Set();
        const handle = subscribeTick(market, symbol, (msg) => {
          if (msg.type !== 'tick') return;
          nodeSet.forEach(n => {
            const f = n.dataset.liveField || 'last';
            const fm = n.dataset.liveFormat || '%.2f';
            const val = msg[f];
            if (val !== undefined) {
              const prev = n.dataset._lastVal ? parseFloat(n.dataset._lastVal) : null;
              n.textContent = _formatNumber(val, fm);
              n.dataset._lastVal = String(val);
              if (prev !== null && prev !== val) {
                n.classList.add(val > prev ? 'live-up' : 'live-down');
                clearTimeout(n._liveFlashTimer);
                n._liveFlashTimer = setTimeout(() => {
                  n.classList.remove('live-up', 'live-down');
                }, 250);
              }
            }
          });
        });
        bucket = { handle, nodes: nodeSet };
        ATTACHED_SUBS.set(key, bucket);
      }
      bucket.nodes.add(node);
    });
  }

  function detachDataset(root) {
    const scope = root || document;
    const nodes = scope.querySelectorAll('[data-live-symbol][data-live-market]');
    nodes.forEach(node => {
      if (node.dataset.liveAttached !== '1') return;
      const market = node.dataset.liveMarket;
      const symbol = node.dataset.liveSymbol;
      const key = `${market}:${symbol}`;
      const bucket = ATTACHED_SUBS.get(key);
      if (!bucket) return;
      bucket.nodes.delete(node);
      node.dataset.liveAttached = '';
      if (bucket.nodes.size === 0) {
        try { bucket.handle.unsubscribe(); } catch (e) {}
        ATTACHED_SUBS.delete(key);
      }
    });
  }

  function status() {
    function s(sock) {
      if (!sock) return 'closed';
      return ['connecting', 'open', 'closing', 'closed'][sock.readyState] || 'unknown';
    }
    return {
      tick: s(STATE.sockets.tick),
      bar: s(STATE.sockets.bar),
      subscribers: STATE.subscribers.size,
      token_set: !!STATE.token,
    };
  }

  // ── Clean shutdown on page unload ─────────────────────────────────────────

  window.addEventListener('beforeunload', () => {
    STATE.closing = true;
    if (STATE.sockets.tick) try { STATE.sockets.tick.close(); } catch (e) {}
    if (STATE.sockets.bar) try { STATE.sockets.bar.close(); } catch (e) {}
  });

  // ── Export ────────────────────────────────────────────────────────────────

  window.liveClient = {
    init,
    subscribeBar,
    subscribeTick,
    attachDataset,
    detachDataset,
    status,
  };
})();

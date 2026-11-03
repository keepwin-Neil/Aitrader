/* ============================================================
 * AITrader i18n engine — 中文/English UI switcher
 * ------------------------------------------------------------
 * 字典在文件头部以 window.__I18N_DICT__ 内联（构建时生成）。
 * 用法：页面引入 /static/i18n.js；右上角出现语言切换按钮。
 *       语言选择存 localStorage('aitrader_lang')，切换后刷新页面。
 * 策略：英文模式下对 DOM 文本节点做「精确匹配 → 中文段最长前缀匹配」，
 *       MutationObserver 兜底翻译 JS 动态插入的内容；
 *       alert/confirm/prompt 消息同样翻译。
 * ============================================================ */
(function () {
  'use strict';

  window.__I18N_DICT__ = {}; /* __I18N_DICT_PLACEHOLDER__ */

  var DICT = window.__I18N_DICT__ || {};
  var LANG_KEY = 'aitrader_lang';
  var lang = 'zh';
  try { lang = localStorage.getItem(LANG_KEY) || 'zh'; } catch (e) {}

  // ── 索引：首字符 → 该字符开头的 key 列表（长度降序）──
  var exact = Object.create(null);      // 规范化 key → 英文
  var byFirst = Object.create(null);    // 首字符 → [key...]（长在前）
  var CJK = /[\u4e00-\u9fff]/;

  function norm(s) { return String(s).trim().replace(/\s+/g, ' '); }

  Object.keys(DICT).forEach(function (k) {
    if (!k || !CJK.test(k)) return;
    var v = DICT[k];
    if (!v || typeof v !== 'string') return;
    exact[norm(k)] = v;
    var c = k.charAt(0);
    (byFirst[c] || (byFirst[c] = [])).push(k);
  });
  Object.keys(byFirst).forEach(function (c) {
    byFirst[c].sort(function (a, b) { return b.length - a.length; });
  });

  // 整串精确翻译（找不到返回 null）
  function translateExact(s) {
    if (!s) return null;
    var v = exact[norm(s)];
    return v === undefined ? null : v;
  }

  // 段级翻译：贪心最长匹配，只替换命中的中文段；无命中返回 null
  function translateRuns(s) {
    if (!s || !CJK.test(s)) return null;
    var out = '', i = 0, len = s.length, hit = false;
    while (i < len) {
      var ch = s.charAt(i);
      var cands = byFirst[ch];
      var matched = null;
      if (cands) {
        for (var j = 0; j < cands.length; j++) {
          var k = cands[j];
          if (k.length > len - i) continue;
          if (s.startsWith(k, i)) { matched = k; break; }
        }
      }
      if (matched) {
        var val = DICT[matched];
        out += val; hit = true; i += matched.length;
      } else {
        out += ch; i++;
      }
    }
    return hit ? out : null;
  }

  function translate(s) {
    return translateExact(s) || translateRuns(s);
  }

  // ── DOM 应用 ─────────────────────────────────────────────
  // 注意: PRE/CODE 不跳过 —— 本应用的测试日志/报告面板用 <pre> 展示用户可见文案
  var SKIP_TAGS = { SCRIPT: 1, STYLE: 1, NOSCRIPT: 1 };
  var TRANS_ATTRS = ['title', 'placeholder', 'aria-label', 'alt'];

  function applyTextNode(node) {
    var t = translate(node.nodeValue);
    if (t !== null && t !== node.nodeValue) node.nodeValue = t;
  }

  function applyElement(el) {
    if (!el || el.nodeType !== 1) return;
    if (SKIP_TAGS[el.tagName]) {
      // textarea 除外其 placeholder
      if (el.tagName === 'TEXTAREA') applyAttrs(el);
      return;
    }
    applyAttrs(el);
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        var p = n.parentNode;
        if (p && SKIP_TAGS[p.tagName]) return NodeFilter.FILTER_REJECT;
        return n.nodeValue && CJK.test(n.nodeValue)
          ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
      }
    });
    var n;
    while ((n = walker.nextNode())) applyTextNode(n);
  }

  function applyAttrs(el) {
    TRANS_ATTRS.forEach(function (a) {
      if (!el.hasAttribute || !el.hasAttribute(a)) return;
      var cur = el.getAttribute(a);
      var t = translate(cur);
      if (t !== null && t !== cur) el.setAttribute(a, t);
    });
  }

  function applyAll() {
    if (document.title) {
      var t = translate(document.title);
      if (t) document.title = t;
    }
    applyElement(document.body);
  }

  // ── alert / confirm / prompt 包装 ───────────────────────
  function wrapDialog(fn) {
    return function (msg) {
      var t = translate(String(msg == null ? '' : msg));
      return fn.call(window, t === null ? msg : t,
        arguments[1], arguments[2]);
    };
  }

  // ── 语言切换按钮 ─────────────────────────────────────────
  function mountToggle() {
    if (document.getElementById('i18n-toggle')) return;
    var b = document.createElement('button');
    b.id = 'i18n-toggle';
    b.type = 'button';
    b.textContent = lang === 'en' ? '中' : 'EN';
    b.title = lang === 'en' ? '切换到中文' : 'Switch to English';
    b.addEventListener('click', function () {
      try { localStorage.setItem(LANG_KEY, lang === 'en' ? 'zh' : 'en'); } catch (e) {}
      location.reload();
    });
    var s = b.style;
    s.position = 'fixed'; s.top = '10px'; s.right = '14px'; s.zIndex = '2147483000';
    s.padding = '4px 12px'; s.border = 'none'; s.borderRadius = '12px';
    s.background = '#1f6feb'; s.color = '#fff';
    s.font = '600 12px/1.6 system-ui, sans-serif';
    s.cursor = 'pointer'; s.boxShadow = '0 1px 4px rgba(0,0,0,.3)';
    s.opacity = '.85';
    b.addEventListener('mouseenter', function () { s.opacity = '1'; });
    b.addEventListener('mouseleave', function () { s.opacity = '.85'; });
    (document.body || document.documentElement).appendChild(b);
  }

  // ── 启动 ─────────────────────────────────────────────────
  function start() {
    mountToggle();
    if (lang !== 'en') return; // 中文模式：保持原样，仅提供切换入口
    window.alert = wrapDialog(window.alert);
    window.confirm = wrapDialog(window.confirm);
    window.prompt = wrapDialog(window.prompt);
    applyAll();
    var mo = new MutationObserver(function (muts) {
      for (var i = 0; i < muts.length; i++) {
        var m = muts[i];
        if (m.type === 'childList') {
          for (var j = 0; j < m.addedNodes.length; j++) {
            var n = m.addedNodes[j];
            if (n.nodeType === 1) applyElement(n);
            else if (n.nodeType === 3) applyTextNode(n);
          }
        } else if (m.type === 'characterData' && m.target.nodeValue) {
          applyTextNode(m.target);
        }
      }
    });
    mo.observe(document.body, { childList: true, subtree: true, characterData: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }

  // 调试接口
  window.i18n = { translate: translate, applyAll: applyAll, getLang: function () { return lang; } };
})();

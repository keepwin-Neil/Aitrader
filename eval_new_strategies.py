"""快速评测新策略 - 在100只NASDAQ股票上跑回测，计算综合评分"""
import sys, json, time, logging
import numpy as np
sys.path.insert(0, '.')
from backtest import get_ohlcv, run_strategy_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()

POOL = ["NVDA","AAPL","MSFT","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    "NFLX","TMUS","AMD","ADBE","PEP","LIN","CSCO","TXN","QCOM","ISRG",
    "INTU","AMGN","AMAT","BKNG","CMCSA","HON","PANW","ADP","VRTX","SBUX",
    "LRCX","MU","GILD","ADI","MDLZ","MELI","KLAC","PYPL","CTAS","REGN",
    "CDNS","SNPS","MAR","CRWD","ABNB","MRVL","ORLY","FTNT","DASH","CSX",
    "MNST","AEP","WDAY","PCAR","BKR","ADSK","ROST","CEG","CHTR","ROP",
    "AZN","TTD","FAST","VRSK","KHC","DXCM","EXC","GEHC","CTSH","EA",
    "XEL","KDP","IDXX","ODFL","BIIB","TEAM","ON","ANSS","CSGP","ZS",
    "CDW","WBD","ILMN","DLTR","GFS","MDB","MRNA","DDOG","FANG","TTWO",
    "SMCI","ARM","COIN","MCHP","SPLK","CPRT","PAYX","LULU","WBA","SIRI"]

def composite_score(summary):
    ret = summary.get('total_return', 0) / 100
    sharpe = summary.get('sharpe', 0)
    dd = summary.get('max_drawdown', 0)
    wr = summary.get('win_rate', 0) / 100
    pf = min(summary.get('profit_factor', 0), 5) / 5
    ret_norm = max(min(ret / 2, 1), -1)
    sharpe_norm = max(min(sharpe / 3, 1), -1)
    dd_norm = max(1 - dd, 0)
    return 0.40 * ret_norm + 0.25 * sharpe_norm + 0.20 * dd_norm + 0.10 * wr + 0.05 * pf

def trimmed_mean(scores, trim=5):
    s = sorted(scores)
    if len(s) <= trim * 2:
        return np.mean(s)
    return np.mean(s[trim:-trim])

active_strats = ['ensemble_voting', 'mean_reversion', 'breakout_trailing',
    'momentum_reversion_hybrid', 'adaptive_regime', 'dual_ma_cross',
    'divergence_combo', 'trend_hold', 'momentum_breakout', 'turtle_trading',
    'supertrend', 'adaptive_ma', 'multi_timeframe_trend']
new_strats = ['obv_momentum', 'keltner_breakout', 'zscore_reversion',
    'donchian_breakout', 'williams_r_adx', 'parabolic_sar', 'heikin_ashi']
all_strats = active_strats + new_strats

log.info(f"评测 {len(all_strats)} 个策略 x {len(POOL)} 只股票")

results = {s: [] for s in all_strats}
t0 = time.time()

for i, sym in enumerate(POOL):
    try:
        df = get_ohlcv(sym, 'US', interval='1d', period='5y')
    except Exception as e:
        log.warning(f"[{i+1}] {sym} data fail: {e}")
        continue
    if len(df) < 50:
        log.warning(f"[{i+1}] {sym} too short ({len(df)} bars)")
        continue
    for sname in all_strats:
        try:
            result = run_strategy_backtest(df.copy(), sname)
            summary = result.get('summary', {})
            score = composite_score(summary)
            results[sname].append(score)
        except Exception as e:
            results[sname].append(0)
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        log.info(f"[{i+1}/{len(POOL)}] elapsed={elapsed:.0f}s")

log.info("=" * 70)
log.info("评测结果（截尾平均综合评分）")
log.info("=" * 70)

final = {}
for sname in all_strats:
    scores = results[sname]
    if not scores: continue
    tm = trimmed_mean(scores)
    avg_score = np.mean(scores)
    pos_count = sum(1 for s in scores if s > 0.4)
    final[sname] = {'trimmed_mean': round(tm, 4), 'avg': round(avg_score, 4), 'pos_count': pos_count, 'total': len(scores)}

ranked = sorted(final.items(), key=lambda x: x[1]['trimmed_mean'], reverse=True)
for i, (name, v) in enumerate(ranked, 1):
    tag = " NEW" if name in new_strats else "    "
    print(f"  {i:2d}.{tag} {name:<30s}  截尾={v['trimmed_mean']:.4f}  平均={v['avg']:.4f}  正收益={v['pos_count']}/{v['total']}")

elapsed = time.time() - t0
log.info(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")

with open('auto_optimize_results/new_strategy_eval.json', 'w') as f:
    json.dump({'ranked': [(n, v) for n, v in ranked], 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
log.info("结果已保存")

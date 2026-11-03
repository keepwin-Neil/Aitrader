"""
Batch optimize all stocks in stock pools.
Calls local API server (must be running first).
"""
import json
import sys
import time
import requests
from pathlib import Path

# Fix Windows console encoding (skip if not supported)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, Exception):
    pass

API = "http://127.0.0.1:9000"
POOLS_FILE = Path(__file__).parent / "stock_pools.json"
OPTIMIZED_DIR = Path(__file__).parent / "optimized_params"

def get_all_symbols():
    pools = json.loads(POOLS_FILE.read_text(encoding="utf-8"))
    seen = set()
    result = []
    for name, pool in pools.items():
        market = pool.get("market", "US")
        for sym in pool.get("symbols", []):
            key = f"{market}:{sym}"
            if key not in seen:
                seen.add(key)
                result.append((sym, market))
    return result

def is_optimized(symbol):
    fp = OPTIMIZED_DIR / f"{symbol.upper()}.json"
    return fp.exists()

def optimize_one(symbol, market, period="2y", timeout=300):
    try:
        resp = requests.post(
            f"{API}/optimize/{market}/{symbol}",
            json={"period": period, "dynamic_position": False},
            timeout=timeout
        )
        if resp.ok:
            data = resp.json()
            n_strategies = len(data.get("strategies", {}))
            combos = data.get("total_combos_tested", 0)
            return True, f"{n_strategies} strategies, {combos} combos"
        else:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.Timeout:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def main():
    all_symbols = get_all_symbols()
    total = len(all_symbols)
    already_done = [(s, m) for s, m in all_symbols if is_optimized(s)]
    todo = [(s, m) for s, m in all_symbols if not is_optimized(s)]

    print("=" * 60)
    print("Batch Stock Parameter Optimization")
    print("=" * 60)
    print(f"Total stocks: {total}")
    print(f"Already done: {len(already_done)}")
    print(f"To optimize:  {len(todo)}")
    print("=" * 60)

    if not todo:
        print("[OK] All stocks already optimized!")
        return

    try:
        requests.get(f"{API}/pools", timeout=5)
    except:
        print("[ERROR] Server not running! Start: uvicorn main:app --host 127.0.0.1 --port 9000")
        return

    success = 0
    failed = 0
    failed_list = []
    start_time = time.time()

    for i, (sym, market) in enumerate(todo):
        elapsed = time.time() - start_time
        if i > 0:
            avg = elapsed / i
            eta = avg * (len(todo) - i)
            eta_str = f"ETA: {int(eta//60)}m{int(eta%60)}s"
        else:
            eta_str = ""

        print(f"\n[{i+1}/{len(todo)}] Optimizing {market}:{sym} ... {eta_str}")
        t0 = time.time()
        ok, msg = optimize_one(sym, market)
        dt = time.time() - t0

        if ok:
            success += 1
            print(f"  [OK] {sym} done in {dt:.1f}s -- {msg}")
        else:
            failed += 1
            failed_list.append((sym, market, msg))
            print(f"  [FAIL] {sym} ({dt:.1f}s) -- {msg}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! Success: {success}  Failed: {failed}  Time: {int(total_time//60)}m{int(total_time%60)}s")

    if failed_list:
        print(f"\nFailed list:")
        for sym, market, msg in failed_list:
            print(f"  {market}:{sym} -- {msg}")

if __name__ == "__main__":
    main()

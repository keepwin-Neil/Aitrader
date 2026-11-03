"""
进程角色标记 (主服务进程 vs ProcessPoolExecutor worker).

之前靠 multiprocessing.current_process().name 前缀猜 ("SpawnPoolWorker-" 等),
但 concurrent.futures.ProcessPoolExecutor 的 worker 名字是 "SpawnProcess-N",
和 uvicorn --reload 起的应用子进程同名, 根本分不开.

用法:
  ProcessPoolExecutor(max_workers=n, initializer=proc_role.mark_pool_worker)
  ...
  if proc_role.is_pool_worker(): ...   # 在 worker 里为 True

零依赖, worker 里 import 无成本.
"""
from __future__ import annotations

import os
import threading
import time

_ENV_FLAG = "AITRADER_POOL_WORKER"

#: 多久查一次父进程还在不在。5 秒足够 —— 这是清理孤儿, 不是实时控制。
_ORPHAN_CHECK_SEC = float(os.getenv("AITRADER_ORPHAN_CHECK_SEC", "5"))


def _watch_parent(orig_ppid: int) -> None:
    """父进程没了就自杀。

    为什么需要: uvicorn --reload 换代、或者直接 kill 掉服务时, 进程池的
    worker 不会跟着走 —— 它们被 init 收养(PPID 变成 1)继续活着, 各自还占着
    几百 MB 和一份 pandas/backtest。本机实测一天之内攒到 117 个, 生产上
    七天攒到 350 个, 最后是靠人工 pkill 收场。
    POSIX 下没有跨平台的"父死跟着死": Linux 有 prctl(PR_SET_PDEATHSIG),
    macOS 没有。所以用最朴素也最可靠的办法 —— 轮询 getppid()。
    """
    while True:
        time.sleep(_ORPHAN_CHECK_SEC)
        try:
            if os.getppid() != orig_ppid:
                # 用 _exit 而不是 sys.exit: 这是守护线程, 抛 SystemExit
                # 只会结束它自己, 进程照样活着。
                os._exit(0)
        except Exception:
            return


def mark_pool_worker() -> None:
    """ProcessPoolExecutor initializer: 在 worker 进程里打标, 并盯住父进程."""
    os.environ[_ENV_FLAG] = "1"
    try:
        t = threading.Thread(target=_watch_parent, args=(os.getppid(),),
                             daemon=True, name="orphan-watch")
        t.start()
    except Exception:
        pass        # 盯不住也不该让 worker 起不来


def is_pool_worker() -> bool:
    """当前进程是否是 ProcessPoolExecutor / multiprocessing.Pool 的 worker."""
    if os.environ.get(_ENV_FLAG) == "1":
        return True
    # 兼容 multiprocessing.Pool (它的 worker 名字是固定前缀, 可靠)
    try:
        import multiprocessing as _mp
        name = _mp.current_process().name
        return name.startswith(("SpawnPoolWorker-", "ForkPoolWorker-"))
    except Exception:
        return False

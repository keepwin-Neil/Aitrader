"""自带锁的 dict —— 后台任务进度表的统一容器。

为什么需要它
------------
后台 worker 对 job 的写散在 89 处 (新策略验证 / 批量优化 / 批量WF 三个 worker),
逐处手写 `with lock:` 既容易漏, 以后新增代码也一定会忘。而读侧(进度轮询接口)
只要撞上写到一半, 遍历就抛
    RuntimeError: dictionary changed size during iteration
→ HTTP 500 → 前端判定任务失败并停止跟踪, 而后台其实还在跑。

两侧都锁, 不是只锁写侧
----------------------
只锁写侧等于没锁: 读侧照样要遍历, 照样会撞。所以这里把**所有会遍历自身的
操作**都放进锁里 —— 写操作、以及 __iter__/keys/values/items/copy/repr。

其中 __iter__ 是关键: CPython 的 dict(x) 对 dict 子类本来走 C 层快路径
(不经过 keys()), 但只要子类覆写了 __iter__, tp_iter 槽就不再是 dict_iter,
构造器改走慢路径 → 经过下面带锁的 keys()。

但**不要**因此以为 `dict(job)` 就安全了: 慢路径是"先 keys() 再逐个
__getitem__", 而 __getitem__ 没有锁 —— 两步之间那个 key 被 pop 掉就是
KeyError(只是把 RuntimeError 换成了 KeyError)。实测 100% 可复现。
读侧唯一安全的入口是 snapshot() / snapshot_of()。本仓库已经全部走它。

__getitem__/get/__len__/__contains__ 不加锁: 它们是单条 C 级操作, GIL 下原子,
不存在"遍历到一半"的窗口, 加锁只会白白拖慢热路径。

锁只包住 dict 操作本身, 绝不会包住回测/优化那些重活。
写操作是纳秒级; 但 snapshot() 是在锁里做整棵树的深拷, 95只×22策略的
批量优化 job 实测持锁 19ms —— 轮询期间后台对该 job 的写会排队。
不死锁(RLock 且内部不回调用户代码), 但别在热循环里 snapshot。
用 RLock 而非 Lock: snapshot() 会递归进嵌套的 LockedDict, 同一线程要能重入。

只依赖 threading —— 故意不依赖 main, 这样 worker 脚本 / 单测都能直接 import。
"""

import threading

__all__ = ["LockedDict", "snapshot_of"]


class LockedDict(dict):

    __slots__ = ("_lock",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._lock = threading.RLock()

    # ── 写侧 ────────────────────────────────────────────────────────────
    def __setitem__(self, k, v):
        with self._lock:
            super().__setitem__(k, v)

    def __delitem__(self, k):
        with self._lock:
            super().__delitem__(k)

    def update(self, *a, **kw):
        with self._lock:
            super().update(*a, **kw)

    def setdefault(self, k, default=None):
        with self._lock:
            return super().setdefault(k, default)

    def pop(self, *a, **kw):
        with self._lock:
            return super().pop(*a, **kw)

    def popitem(self):
        with self._lock:
            return super().popitem()

    def clear(self):
        with self._lock:
            super().clear()

    def __ior__(self, other):          # job |= {...}
        with self._lock:
            super().update(other)
        return self

    # ── 读侧 (都返回脱离锁的副本, 不是 live view) ────────────────────────
    # 返回 list 而不是 dict 的 view: view 是活的, 交出去照样会撞写侧。
    def __iter__(self):
        with self._lock:
            return iter(list(dict.keys(self)))

    def keys(self):
        with self._lock:
            return list(dict.keys(self))

    def values(self):
        with self._lock:
            return list(dict.values(self))

    def items(self):
        with self._lock:
            return list(dict.items(self))

    def copy(self):
        # dict.copy() 继承下来会返回**普通 dict**(悄悄掉锁), 必须覆写。
        with self._lock:
            return self.__class__(dict(dict.items(self)))

    def __repr__(self):
        with self._lock:
            return f"{self.__class__.__name__}({dict(dict.items(self))!r})"

    # ── pickle / copy / deepcopy ────────────────────────────────────────
    # RLock 不能 pickle, 不处理的话哪天有人把 job 传进 ProcessPoolExecutor
    # 就是一个 TypeError: cannot pickle '_thread.RLock' object, 且报错点离根因很远。
    #
    # 用 __reduce__ 的第 5 位 dictitems: 对端先构造空壳(拿到自己的新锁、并被
    # pickle memo 记住), 再逐项 obj[k]=v 填回去。这样引用共享、自引用环都由
    # pickle 自己处理, 嵌套的 LockedDict 作为 value 走它自己的 __reduce__,
    # 锁一层不丢。copy.copy / copy.deepcopy 也自动走这条路, 语义正确。
    def __reduce__(self):
        with self._lock:
            items = list(dict.items(self))
        return (self.__class__, (), None, None, iter(items))

    # ── 快照 ────────────────────────────────────────────────────────────
    def snapshot(self):
        """取一份**完全脱离**写侧的普通 dict (容器逐层深拷, 叶子值直接引用)。

        读侧首选入口。拿到之后怎么遍历、怎么序列化都不会再撞上写侧。
        """
        with self._lock:
            return {k: _detach(v) for k, v in dict.items(self)}


def _detach(v, _depth=0):
    """递归脱壳: 容器逐层复制, 叶子(数字/字符串/DataFrame…)直接引用。

    必须真递归 —— 只拷一层的话, job["results"][sym]["strategy_details"]
    仍然是 worker 手里的活对象, 读侧序列化时照样会撞。
    """
    if _depth > 12:                       # 防环: 12 层足够覆盖本项目最深的 job 结构
        return v
    if isinstance(v, LockedDict):
        with v._lock:
            return {k: _detach(x, _depth + 1) for k, x in dict.items(v)}
    if isinstance(v, dict):
        return {k: _detach(x, _depth + 1) for k, x in list(v.items())}
    if isinstance(v, (list, tuple)):
        return [_detach(x, _depth + 1) for x in list(v)]
    return v


def snapshot_of(job):
    """对任意 job 取快照。非 LockedDict 也能用, 方便过渡期。"""
    if isinstance(job, LockedDict):
        return job.snapshot()
    return _detach(job)

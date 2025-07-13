from types import NoneType
from typing import Any, Callable, Generic, TypeVar, Optional
import numpy as np

T = TypeVar("T")


class CacheObject(Generic[T]):
    def __init__(self):
        self.lastx = {}
        self.val: Optional[T] = None


def cache_fetch[T](
    cache: CacheObject,
    func: Callable[..., T],
    cache_kwargs: dict[str, Any],
    **funkwargs: Any,
) -> T:
    is_old_kwargs = all(
        [
            key in cache.lastx and np.array_equal(val, cache.lastx[key])
            for key, val in cache_kwargs.items()
        ]
    )
    if is_old_kwargs:
        val = cache.val
    else:
        fval = func(**funkwargs)
        cache.val = fval
        for key, val in cache_kwargs.items():
            cache.lastx[key] = val
    return cache.val

import numpy as np


class CacheObject:
    def __init__(self):
        self.lastx = {}
        self.val = None


def cache_fetch(cache, func, cache_kwargs, **funkwargs):
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

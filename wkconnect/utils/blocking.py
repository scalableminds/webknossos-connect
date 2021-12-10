from asyncio import get_running_loop
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, TypeVar

THREAD_POOL_MAX_WORKERS = 32
_T = TypeVar("_T")


@lru_cache()
def get_thread_pool_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKERS)


async def run_blocking(blocking_function: Callable[..., _T], *args: Any) -> _T:
    executor = get_thread_pool_executor()
    loop = get_running_loop()

    return await loop.run_in_executor(executor, blocking_function, *args)

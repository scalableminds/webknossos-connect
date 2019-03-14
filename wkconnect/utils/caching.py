import time
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

from async_lru import alru_cache

T = TypeVar("T", bound=Callable[..., Awaitable[Any]])


def atlru_cache(
    seconds_to_use: int, maxsize: int = 512, typed: bool = False
) -> Callable[[T], T]:
    def decorator(f: T) -> T:
        @wraps(f)
        @alru_cache(maxsize=maxsize, typed=typed, cache_exceptions=False)
        async def time_f(*args: Any, **kwargs: Any) -> Any:
            return (time.monotonic(), await f(*args, **kwargs))

        @wraps(time_f)
        async def cache_decorated_function(*args: Any, **kwargs: Any) -> Any:
            now = time.monotonic()
            inserted, result = await time_f(*args, **kwargs)
            if now - inserted > seconds_to_use:
                time_f.invalidate(*args, **kwargs)  # type: ignore
                new_inserted, result = await time_f(*args, **kwargs)
                assert new_inserted >= now
            return result

        # cast necessary, see
        # https://github.com/python/mypy/issues/1927
        return cast(T, cache_decorated_function)

    return decorator

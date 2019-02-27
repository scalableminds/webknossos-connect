import asyncio
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar, cast

T = TypeVar("T", bound=Callable[..., Awaitable[Any]])


def repeat_every_seconds(
    interval_seconds: float, initial_call: bool = True
) -> Callable[[T], T]:
    def decorator(f: T) -> T:
        @wraps(f)
        async def decorator(*args: Any, **kwargs: Any) -> Any:
            if initial_call:
                await f(*args, **kwargs)
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                await f(*args, **kwargs)

        # cast necessary, see
        # https://github.com/python/mypy/issues/1927
        return cast(T, decorator)

    return decorator

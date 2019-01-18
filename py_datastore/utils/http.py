from __future__ import annotations

import aiohttp

from typing import Any, Callable


class HttpError(RuntimeError):
    def __init__(self, status: int, *args: Any) -> None:
        super().__init__((status, *args))
        self.status = status


class HttpClient:
    async def __aenter__(self) -> HttpClient:
        self.http_session = await aiohttp.ClientSession().__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.http_session.__aexit__(exc_type, exc, tb)

    async def __perform_request(
        self, request_fn: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        if "response_fn" in kwargs:
            response_fn = kwargs["response_fn"]
            del kwargs["response_fn"]
        else:
            response_fn = None

        async with request_fn(*args, **kwargs) as response:
            if response.status is not 200:
                text = await response.text()
                raise HttpError(response.status, (text, *args))
            else:
                return await response_fn(response) if response_fn else None

    async def patch(self, *args: Any, **kwargs: Any) -> Any:
        return await self.__perform_request(self.http_session.patch, *args, **kwargs)

    async def get(self, *args: Any, **kwargs: Any) -> Any:
        return await self.__perform_request(self.http_session.get, *args, **kwargs)

    async def put(self, *args: Any, **kwargs: Any) -> Any:
        return await self.__perform_request(self.http_session.put, *args, **kwargs)

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        return await self.__perform_request(self.http_session.post, *args, **kwargs)

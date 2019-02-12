from __future__ import annotations

from asyncio import gather
from functools import wraps
from typing import Any, Awaitable, Callable, NamedTuple, Optional, TypeVar, cast

from sanic.request import Request
from sanic.response import HTTPResponse, text

from .models import DataSourceId


class AccessRequest(NamedTuple):
    resourceId: DataSourceId
    resourceType: str
    mode: str

    @classmethod
    def read_dataset(
        cls, organization_name: str, dataset_name: str, **_: Any
    ) -> AccessRequest:
        return cls(
            resourceId=DataSourceId(organization_name, dataset_name),
            resourceType="datasource",
            mode="read",
        )


class AccessAnswer(NamedTuple):
    granted: bool
    msg: Optional[str] = None


T = TypeVar("T", bound=Callable[..., Awaitable[HTTPResponse]])


def authorized(fn_access_request: Callable[..., AccessRequest]) -> Callable[[T], T]:
    def decorator(f: T) -> T:
        @wraps(f)
        async def decorated_function(
            request: Request, *args: Any, **kwargs: Any
        ) -> HTTPResponse:
            tokens = request.args.getlist("token", default=[])
            parameters = request.app.router.get(request)[2]
            access_request = fn_access_request(**parameters)

            access_answers = await gather(
                *(
                    request.app.webknossos.request_access(
                        token=token, access_request=access_request
                    )
                    for token in tokens
                )
            )

            if any(i.granted for i in access_answers):
                return await f(request, *args, **kwargs)
            else:
                messages = [i.msg for i in access_answers if i.msg is not None]
                return text(
                    f"Forbidden: {'. '.join(messages)}"
                    if len(messages) > 0
                    else "Token authentication failed",
                    status=403,
                )

        return cast(T, decorated_function)

    return decorator

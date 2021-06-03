from __future__ import annotations

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

    @classmethod
    def administrate_datasets(cls, **_: Any) -> AccessRequest:
        return cls(
            resourceId=DataSourceId("", ""),
            resourceType="datasource",
            mode="administrate",
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
            token = request.args.get("token", default="")
            parameters = request.app.router.get(
                request.path, request.method, request.host
            )[2]
            access_request = fn_access_request(**parameters)

            access_answer = await request.app.ctx.webknossos.request_access(
                token=token, access_request=access_request
            )

            if access_answer.granted:
                return await f(request, *args, **kwargs)
            else:
                return text(
                    f"Forbidden: {access_answer.msg}"
                    if access_answer.msg is not None
                    else "Token authentication failed",
                    status=403,
                )

        return cast(T, decorated_function)

    return decorator

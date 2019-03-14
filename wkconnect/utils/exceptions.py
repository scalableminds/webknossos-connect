import traceback

from aiohttp import ClientResponseError


def exception_traceback(exception: Exception) -> str:
    return "".join(
        traceback.format_exception(
            etype=type(exception), value=exception, tb=exception.__traceback__
        )
    )


def format_exception(exception: Exception) -> str:
    if isinstance(exception, ClientResponseError):
        description = f"HTTP {exception.message} ({exception.status})"
    else:
        description = type(exception).__name__
    return f"{description} in webknossos-connect."

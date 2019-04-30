import traceback

from aiohttp import ClientResponseError


class RuntimeErrorWithUserMessage(RuntimeError):
    def __init__(self, user_message: str):
        self.user_message = user_message
        super().__init__(user_message)


def exception_traceback(exception: Exception) -> str:
    return "".join(
        traceback.format_exception(
            etype=type(exception), value=exception, tb=exception.__traceback__
        )
    )


def format_exception(exception: Exception) -> str:
    if isinstance(exception, ClientResponseError):
        description = f"HTTP {exception.message} ({exception.status})"
    if isinstance(exception, RuntimeErrorWithUserMessage):
        return exception.user_message
    else:
        description = type(exception).__name__
    return f"{description} in webknossos-connect"

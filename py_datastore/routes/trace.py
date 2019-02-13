import asyncio
import os

from sanic import Blueprint, response
from sanic.request import Request

trace = Blueprint(__name__, url_prefix="/triggers")

trace.static("/trace", "data/trace.html", name="trace_get")
trace.static("/trace/flame.svg", "data/flame.svg", name="trace_get")


@trace.route("/trace", methods=["POST"])
async def trace_post(request: Request) -> response.HTTPResponse:
    # py-spy is only in dev-packages
    p = await asyncio.create_subprocess_exec(
        "py-spy",
        "--flame",
        "data/flame.svg",
        "--duration",
        "20",
        "--pid",
        str(os.getpid()),
    )
    await p.wait()
    return (
        response.text("Ok") if p.returncode == 0 else response.text("Error", status=500)
    )

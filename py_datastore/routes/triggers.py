from sanic import Blueprint, response
from sanic.request import Request

triggers = Blueprint(__name__, url_prefix="/triggers")


@triggers.route("/checkInboxBlocking")
async def check_inbox_blocking(request: Request) -> response.HTTPResponse:
    await request.app.load_persisted_datasets()
    return response.text("Ok")

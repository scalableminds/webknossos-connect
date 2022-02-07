from typing import TYPE_CHECKING, cast

from sanic import Blueprint, response
from sanic.request import Request

from ..webknossos.access import AccessRequest, authorized

if TYPE_CHECKING:
    from ..__main__ import Server

triggers = Blueprint(__name__, url_prefix="/triggers")


@triggers.route("/checkInboxBlocking", methods=["POST"])
@authorized(AccessRequest.administrate_datasets)
async def check_inbox_blocking(request: Request) -> response.HTTPResponse:
    app = cast("Server", request.app)
    await app.load_persisted_datasets()
    return response.text("Ok")


@triggers.route("/newOrganizationFolder", methods=["POST"])
@authorized(AccessRequest.administrate_datasets)
async def new_organization_folder(request: Request) -> response.HTTPResponse:
    return response.text("Ok")


@triggers.route("/reload/<organization_name>/<dataset_name>", methods=["POST"])
@authorized(AccessRequest.administrate_datasets)
async def reload_dataset(
    request: Request, organization_name: str, dataset_name: str
) -> response.HTTPResponse:
    app = cast("Server", request.app)
    await app.load_persisted_datasets()
    (backend_name, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = app.ctx.backends[backend_name]
    backend.clear_dataset_cache(dataset)
    return response.text("Ok")

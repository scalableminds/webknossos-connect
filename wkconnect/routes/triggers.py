from sanic import Blueprint, response
from sanic.request import Request

from ..webknossos.access import AccessRequest, authorized

triggers = Blueprint(__name__, url_prefix="/triggers")


@triggers.route("/checkInboxBlocking")
@authorized(AccessRequest.administrate_datasets)
async def check_inbox_blocking(request: Request) -> response.HTTPResponse:
    await request.app.load_persisted_datasets()
    return response.text("Ok")


@triggers.route("/newOrganizationFolder")
@authorized(AccessRequest.administrate_datasets)
async def new_organization_folder(request: Request) -> response.HTTPResponse:
    return response.text("Ok")


@triggers.route("/checkNewOrganizationFolder")
async def check_new_organization_folder(request: Request) -> response.HTTPResponse:
    return response.text("Ok")


@triggers.route("/reload/<organization_name>/<dataset_name>")
@authorized(AccessRequest.administrate_datasets)
async def reload_dataset(
    request: Request, organization_name: str, dataset_name: str
) -> response.HTTPResponse:
    await request.app.load_persisted_datasets()
    (backend_name, dataset) = request.app.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.backends[backend_name]
    backend.clear_dataset_cache(dataset)
    return response.text("Ok")

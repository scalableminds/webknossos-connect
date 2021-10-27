from sanic import Blueprint, response
from sanic.request import Request

from ...webknossos.access import AccessRequest, authorized

meshes = Blueprint(__name__)


@meshes.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/meshes", methods=["GET"]
)
#@authorized(AccessRequest.read_dataset)
async def histogram_post(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    del organization_name
    del dataset_name
    del layer_name
    response.json(to_json([]))

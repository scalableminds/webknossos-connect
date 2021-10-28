from sanic import Blueprint, response
from sanic.request import Request

meshes = Blueprint(__name__)


@meshes.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/meshes", methods=["GET"]
)
async def get_meshes(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    del organization_name
    del dataset_name
    del layer_name
    return response.json([])

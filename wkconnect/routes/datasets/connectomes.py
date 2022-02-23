from sanic import Blueprint, response
from sanic.request import Request

connectomes = Blueprint(__name__)


@connectomes.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/connectomes",
    methods=["GET"],
)
async def get_connectomes(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    del organization_name
    del dataset_name
    del layer_name
    return response.json([])

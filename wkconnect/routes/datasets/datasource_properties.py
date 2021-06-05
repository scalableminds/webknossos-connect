from sanic import Blueprint, response
from sanic.request import Request

from ...utils.json import to_json
from ...webknossos.access import AccessRequest, authorized

datasource_properties = Blueprint(__name__)


@datasource_properties.route("/<organization_name>/<dataset_name>", methods=["GET"])
@authorized(AccessRequest.read_dataset)
async def datasource_properties_get(
    request: Request, organization_name: str, dataset_name: str
) -> response.HTTPResponse:
    (_, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    return response.json(
        {"dataSource": to_json(dataset.to_webknossos()), "messages": []}
    )


@datasource_properties.route("/<organization_name>/<dataset_name>", methods=["POST"])
async def datasource_properties_post(
    request: Request, organization_name: str, dataset_name: str
) -> response.HTTPResponse:
    return response.text(
        "Not Implemented: webknossos-connect does not support editing the datasource-properties. Try refreshing the dataset instead.",
        status=501,
    )

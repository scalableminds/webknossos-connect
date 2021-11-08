from sanic import Blueprint, response
from sanic.request import Request

from wkconnect.utils.types import Vec3D

meshes = Blueprint(__name__)


@meshes.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/meshes", methods=["GET"]
)
async def get_meshes(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    (backend_name, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.ctx.backends[backend_name]

    try:
        return response.json(await backend.get_meshes(dataset, layer_name))
    except NotImplementedError:
        return response.empty(status=501)


@meshes.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/meshes/chunks",
    methods=["POST"],
)
async def get_mesh_chunks(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    request_body = request.json

    (backend_name, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.ctx.backends[backend_name]

    segment_id = request_body["segmentId"]
    try:
        chunks = await backend.get_chunks_for_mesh(
            dataset, layer_name, "mesh", segment_id
        )
        if chunks is None:
            return response.empty(status=404)
        return response.json(chunks)
    except NotImplementedError:
        return response.empty(status=501)


@meshes.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/meshes/chunks/data",
    methods=["POST"],
)
async def get_mesh_chunk_data(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    request_body = request.json

    (backend_name, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.ctx.backends[backend_name]
    segment_id = request_body["segmentId"]
    position = Vec3D(*request_body["position"])

    try:
        chunk_data = await backend.get_chunk_data_for_mesh(
            dataset, layer_name, "mesh", segment_id, position
        )
        if chunk_data is None:
            return response.empty(status=404)

        return response.raw(chunk_data)
    except NotImplementedError:
        return response.empty(status=501)

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

    if backend_name == "neuroglancer":
        if layer_name in dataset.layers and dataset.layers[layer_name].mesh:
            return response.json(["mesh"])

    return response.json([])


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

    if backend_name == "neuroglancer":
        segment_id = request_body["segmentId"]
        chunks = await backend.get_chunks_for_mesh(
            dataset, layer_name, "mesh", segment_id
        )
        return response.json(chunks)

    return response.json([])


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

    if backend_name == "neuroglancer":
        segment_id = request_body["segmentId"]
        position = Vec3D(*request_body["position"])
        chunk_data = await backend.get_chunk_data_for_mesh(
            dataset, layer_name, "mesh", segment_id, position
        )
        return response.raw(chunk_data)

    return response.text("", 404)

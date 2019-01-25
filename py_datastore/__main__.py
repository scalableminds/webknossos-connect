import asyncio
import base64
import json
import numpy as np
import os

from aiohttp import ClientSession
from copy import deepcopy
from io import BytesIO
from PIL import Image
from sanic import Sanic, response
from sanic.request import Request
from sanic_cors import cross_origin
from typing import Dict, List, Tuple, Type
from uvloop import Loop

from .backends.backend import Backend
from .backends.neuroglancer.backend import NeuroglancerBackend as Neuroglancer
from .repository import Repository
from .utils.json import from_json, to_json
from .utils.types import JSON
from .webknossos.access import AccessRequest, authorized
from .webknossos.client import WebKnossosClient as WebKnossos
from .webknossos.models import DataRequest as WKDataRequest


class Server(Sanic):
    def __init__(self) -> None:
        super().__init__()
        self.http_client: ClientSession
        self.repository: Repository
        self.webknossos: WebKnossos
        self.backends: Dict[str, Backend]
        self.available_backends: List[Type[Backend]] = [Neuroglancer]

    async def add_dataset(
        self,
        dataset_config: JSON,
        backend_name: str,
        organization_name: str,
        dataset_name: str,
    ) -> None:
        backend = self.backends[backend_name]
        dataset = await backend.handle_new_dataset(
            organization_name, dataset_name, deepcopy(dataset_config)
        )
        self.repository.add_dataset(backend_name, dataset)
        await self.webknossos.report_dataset(dataset.to_webknossos())

    async def load_persisted_datasets(self) -> None:
        try:
            with open(self.config["datasets_path"]) as datasets_file:
                datasets = json.load(datasets_file)
        except FileNotFoundError:
            datasets = {}
        await asyncio.gather(
            *(
                app.add_dataset(
                    dataset_config=dataset_details,
                    backend_name=backend,
                    organization_name=organization,
                    dataset_name=dataset,
                )
                for backend, backend_details in datasets.items()
                for organization, organization_details in backend_details.items()
                for dataset, dataset_details in organization_details.items()
            )
        )


app = Server()

with open("data/config.json") as config_file:
    app.config.update(json.load(config_file))


## TASKS ##


@app.listener("before_server_start")
async def setup(app: Server, loop: Loop) -> None:
    def instanciate_backend(backend_class: Type[Backend]) -> Tuple[str, Backend]:
        backend_name = backend_class.name()
        config = app.config["backends"][backend_name]
        return (backend_name, backend_class(config, app.http_client))

    app.http_client = await ClientSession(raise_for_status=True).__aenter__()
    app.repository = Repository()
    app.webknossos = WebKnossos(app.config, app.http_client)
    app.backends = dict(map(instanciate_backend, app.available_backends))
    await app.load_persisted_datasets()


@app.listener("after_server_stop")
async def close_http_client(app: Server, loop: Loop) -> None:
    await app.http_client.__aexit__(None, None, None)


async def ping_webknossos(app: Server) -> None:
    ping_interval_seconds = app.config["webknossos"]["ping_interval_minutes"] * 60
    while True:
        await app.webknossos.report_status()
        await asyncio.sleep(ping_interval_seconds)


## ROUTES ##


@app.route("/data/health")
async def health(request: Request) -> response.HTTPResponse:
    return response.text("Ok")


@app.route("/api/buildinfo")
async def build_info(request: Request) -> response.HTTPResponse:
    return response.json(
        {
            "py-datastore": {
                "name": "py-datastore",
                "version": "0.1",
                "datastoreApiVersion": "1.0",
            }
        }
    )


@app.route("/api/<backend_name>/<organization_name>/<dataset_name>", methods=["POST"])
async def add_dataset(
    request: Request, backend_name: str, organization_name: str, dataset_name: str
) -> response.HTTPResponse:
    dataset_config = request.json
    await app.add_dataset(dataset_config, backend_name, organization_name, dataset_name)

    ds_path = app.config["datasets_path"]
    os.makedirs(os.path.dirname(ds_path), exist_ok=True)
    if not os.path.isfile(ds_path):
        with open(ds_path, "w") as datasets_file:
            json.dump({}, datasets_file)
    with open(ds_path, "r+") as datasets_file:
        datasets = json.load(datasets_file)
        datasets.setdefault(backend_name, {}).setdefault(organization_name, {})[
            dataset_name
        ] = dataset_config
        datasets_file.seek(0)
        json.dump(datasets, datasets_file)

    return response.text("Ok")


@app.route(
    "/data/datasets/<organization_name>/<dataset_name>/layers/<layer_name>/data",
    methods=["POST", "OPTIONS"],
)
@authorized(AccessRequest.readDataset)
@cross_origin(app, automatic_options=True)
async def get_data_post(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    (backend_name, dataset) = app.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = app.backends[backend_name]

    bucket_requests = from_json(request.json, List[WKDataRequest])
    assert all(not request.fourBit for request in bucket_requests)

    buckets = await asyncio.gather(
        *(
            backend.read_data(
                dataset,
                layer_name,
                r.zoomStep,
                r.position,
                (r.cubeSize, r.cubeSize, r.cubeSize),
            )
            for r in bucket_requests
        )
    )
    missing_buckets = [index for index, data in enumerate(buckets) if data is None]
    existing_buckets = [data.flatten(order="F") for data in buckets if data is not None]
    data = np.concatenate(existing_buckets) if len(existing_buckets) > 0 else b""

    headers = {
        "Access-Control-Expose-Headers": "MISSING-BUCKETS",
        "MISSING-BUCKETS": json.dumps(missing_buckets),
    }
    return response.raw(data, headers=headers)


@app.route(
    "/data/datasets/<organization_name>/<dataset_name>/layers/<layer_name>/thumbnail.json"
)
@authorized(AccessRequest.readDataset)
async def get_thumbnail(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    width = int(request.args.get("width"))
    height = int(request.args.get("height"))

    (backend_name, dataset) = app.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = app.backends[backend_name]
    data = await backend.read_data(
        dataset, layer_name, 0, (6400, 6400, 640), (width, height, 1)
    )
    thumbnail = Image.fromarray(data[:, :, 0])
    with BytesIO() as output:
        thumbnail.save(output, "JPEG")
        return response.json(
            {"mimeType": "image/jpeg", "value": base64.b64encode(output.getvalue())}
        )


## MAIN ##


if __name__ == "__main__":
    app.add_task(ping_webknossos(app))
    app.run(
        host=app.config["server"]["host"],
        port=app.config["server"]["port"],
        access_log=False,
    )

import asyncio
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Type

from aiohttp import ClientSession
from sanic import Sanic, response
from sanic.request import Request
from sanic_cors import CORS
from uvloop import Loop

from .backends.backend import Backend
from .backends.neuroglancer.backend import NeuroglancerBackend as Neuroglancer
from .repository import Repository
from .routes import routes
from .utils.types import JSON
from .webknossos.client import WebKnossosClient as WebKnossos


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
CORS(app, automatic_options=True)

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


app.blueprint(routes)


@app.route("/data/health")
async def health(request: Request) -> response.HTTPResponse:
    return response.text("Ok")


@app.route("/api/buildinfo")
async def build_info(request: Request) -> response.HTTPResponse:
    return response.json(
        {
            "webknossos-connect": {
                "name": "webknossos-connect",
                "version": "0.1",
                "datastoreApiVersion": "1.0",
            }
        }
    )


## MAIN ##


if __name__ == "__main__":
    app.add_task(ping_webknossos(app))
    app.run(
        host=app.config["server"]["host"],
        port=app.config["server"]["port"],
        access_log=False,
    )

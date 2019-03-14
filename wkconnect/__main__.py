import asyncio
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Type

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectorError
from sanic import Sanic, response
from sanic.exceptions import SanicException
from sanic.handlers import ErrorHandler
from sanic.request import Request
from sanic_cors import CORS
from uvloop import Loop

from .backends.backend import Backend
from .backends.boss.backend import Boss
from .backends.neuroglancer.backend import Neuroglancer
from .repository import Repository
from .routes import routes
from .utils.scheduler import repeat_every_seconds
from .utils.types import JSON
from .webknossos.client import WebKnossosClient as WebKnossos
from .webknossos.models import DataSourceId, UnusableDataSource

logger = logging.getLogger()


class Server(Sanic):
    def __init__(self) -> None:
        super().__init__()
        self.http_client: ClientSession
        self.repository: Repository
        self.webknossos: WebKnossos
        self.backends: Dict[str, Backend]
        self.available_backends: List[Type[Backend]] = [Boss, Neuroglancer]

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
        def expandvars_hook(dict: Dict[str, Any]) -> Dict[str, Any]:
            for key, val in dict.items():
                if isinstance(val, str):
                    dict[key] = os.path.expandvars(val)
            return dict

        try:
            with open(self.config["datasets_path"]) as datasets_file:
                datasets = json.load(datasets_file, object_hook=expandvars_hook)
        except FileNotFoundError:
            datasets = {}
        dataset_tuples = [
            (backend, organization, dataset, dataset_details)
            for backend, backend_details in datasets.items()
            for organization, organization_details in backend_details.items()
            for dataset, dataset_details in organization_details.items()
        ]
        results = await asyncio.gather(
            *(
                self.add_dataset(
                    dataset_config=dataset_details,
                    backend_name=backend,
                    organization_name=organization,
                    dataset_name=dataset,
                )
                for backend, organization, dataset, dataset_details in dataset_tuples
            ),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error(exception_traceback(result))
        await asyncio.gather(
            *(
                self.webknossos.report_dataset(
                    UnusableDataSource(
                        DataSourceId(organization, dataset),
                        status=format_exception(result),
                    )
                )
                for result, (_, organization, dataset, _) in zip(
                    results, dataset_tuples
                )
                if isinstance(result, Exception)
            )
        )


app = Server()
CORS(app, automatic_options=True)

with open("data/config.json") as config_file:
    app.config.update(json.load(config_file))


## ERROR HANDLING


class CustomErrorHandler(ErrorHandler):
    def default(self, request: Request, exception: Exception) -> response.HTTPResponse:
        """handles errors that have no other error handlers assigned"""
        if isinstance(exception, SanicException):
            return super().default(request, exception)
        else:
            message = format_exception(exception)
            stack = exception_traceback(exception)
            message_json = {"messages": [{"error": message, "chain": [stack]}]}
            logger.error(stack)
            return response.json(message_json, status=500)


app.error_handler = CustomErrorHandler()


## TASKS ##


@app.listener("before_server_start")
async def setup(app: Server, loop: Loop) -> None:
    def instanciate_backend(backend_class: Type[Backend]) -> Tuple[str, Backend]:
        backend_name = backend_class.name()
        config = app.config["backends"].get(backend_name, {})
        return (backend_name, backend_class(config, app.http_client))

    app.http_client = await ClientSession(raise_for_status=True).__aenter__()
    app.repository = Repository()
    app.webknossos = WebKnossos(app.config, app.http_client)
    app.backends = dict(map(instanciate_backend, app.available_backends))


@app.listener("before_server_stop")
async def stop_tasks(app: Server, loop: Loop) -> None:
    for task in asyncio.Task.all_tasks():
        if not task == asyncio.Task.current_task():
            task.cancel()


@app.listener("after_server_stop")
async def shutdown(app: Server, loop: Loop) -> None:
    await asyncio.gather(*(backend.on_shutdown() for backend in app.backends.values()))
    await app.http_client.__aexit__(None, None, None)


## ROUTES ##


app.blueprint(routes)


@app.route("/data/health")
async def health(request: Request) -> response.HTTPResponse:
    return response.text("Ok")


@app.route("/api/buildinfo")
async def build_info(request: Request) -> response.HTTPResponse:
    return response.json(
        {
            "webknossosDatastore": {
                "webknossos-connect": {
                    "name": "webknossos-connect",
                    "version": "0.1",
                    "datastoreApiVersion": "1.0",
                }
            }
        }
    )


## MAIN ##


if __name__ == "__main__":

    @repeat_every_seconds(10 * 60)
    async def ping_webknossos(app: Server) -> None:
        try:
            await app.webknossos.report_status()
        except ClientConnectorError:
            logger.warning("Could not ping webknossos, retrying in 10 min.")

    app.add_task(ping_webknossos)

    @repeat_every_seconds(10 * 60)  # , initial_call = False)
    async def scan_inbox(app: Server) -> None:
        try:
            await app.load_persisted_datasets()
        except ClientConnectorError:
            logger.warning(
                "Could not report datasets to webknossos, retrying in 10 min."
            )

    app.add_task(scan_inbox)

    app.run(
        host=app.config["server"]["host"],
        port=app.config["server"]["port"],
        access_log=False,
    )

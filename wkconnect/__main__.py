import asyncio
import json
import logging
import os
from copy import deepcopy
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Type

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientConnectorError, ClientResponseError
from sanic import Sanic, response
from sanic.exceptions import SanicException
from sanic.handlers import ErrorHandler
from sanic.request import Request
from sanic_cors import CORS
from uvloop import Loop

from .backends.backend import Backend
from .backends.boss.backend import Boss
from .backends.neuroglancer.backend import Neuroglancer
from .backends.tiff.backend import Tiff
from .backends.wkw.backend import Wkw
from .repository import Repository
from .routes import routes
from .utils.exceptions import exception_traceback, format_exception
from .utils.scheduler import repeat_every_seconds
from .utils.types import JSON
from .webknossos.client import WebKnossosClient as WebKnossos
from .webknossos.models import DataSource, DataSourceId, UnusableDataSource

logger = logging.getLogger()


class Server(Sanic):
    def __init__(self) -> None:
        super().__init__()
        self.http_client: ClientSession
        self.repository: Repository
        self.webknossos: WebKnossos
        self.backends: Dict[str, Backend]
        self.available_backends: List[Type[Backend]] = [Boss, Neuroglancer, Tiff, Wkw]

    async def add_dataset(
        self,
        dataset_config: JSON,
        backend_name: str,
        organization_name: str,
        dataset_name: str,
        report_to_wk: bool = True,
    ) -> DataSource:
        backend = self.backends[backend_name]
        dataset = await backend.handle_new_dataset(
            organization_name, dataset_name, deepcopy(dataset_config)
        )
        self.repository.add_dataset(backend_name, dataset)
        wk_dataset = dataset.to_webknossos()
        if report_to_wk:
            await self.webknossos.report_dataset(wk_dataset)
        return wk_dataset

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
                    report_to_wk=False,
                )
                for backend, organization, dataset, dataset_details in dataset_tuples
            ),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.error(exception_traceback(result))

        usable = [result for result in results if not isinstance(result, Exception)]
        unusable = [
            UnusableDataSource(
                DataSourceId(organization, dataset), status=format_exception(result)
            )
            for result, (_, organization, dataset, _) in zip(results, dataset_tuples)
            if isinstance(result, Exception)
        ]
        await self.webknossos.report_all_datasets(usable + unusable)


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
            message = format_exception(exception) + "."
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

    def add_regular_interaction(
        action_name: str, fn: Callable[[Server], Awaitable[None]]
    ) -> None:
        @repeat_every_seconds(60)
        async def do_interaction(app: Server) -> None:
            try:
                await fn(app)
            except ClientConnectorError:
                logger.warning(f"Could not {action_name}, retrying in 1 min.")
            except ClientResponseError as exception:
                logger.warning(
                    f"Could not {action_name}, retrying in 1 min. Got {format_exception(exception)}."
                )

        app.add_task(repeat_every_seconds(10 * 60)(do_interaction))

    add_regular_interaction(
        "ping webknossos", lambda app: app.webknossos.report_status()
    )
    add_regular_interaction(
        "report datasets to webknossos", lambda app: app.load_persisted_datasets()
    )

    app.run(
        host=app.config["server"]["host"],
        port=app.config["server"]["port"],
        access_log=False,
    )

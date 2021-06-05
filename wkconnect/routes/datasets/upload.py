import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, Iterable, Tuple, cast

from sanic import Blueprint, response
from sanic.request import Request

from ...utils.types import JSON
from ...webknossos.access import AccessRequest, authorized

if TYPE_CHECKING:
    from ...__main__ import Server

upload = Blueprint(__name__)


def iterate_datasets(datasets: JSON) -> Iterable[Tuple[Any, str, str, str]]:
    for backend_name, backend_vals in datasets.items():
        for organization_name, organization_vals in backend_vals.items():
            for dataset_name, dataset_config in organization_vals.items():
                yield dataset_config, backend_name, organization_name, dataset_name


@upload.route("", methods=["POST"])
@authorized(AccessRequest.administrate_datasets)
async def add_dataset(request: Request) -> response.HTTPResponse:
    app = cast("Server", request.app)
    await asyncio.gather(
        *(
            app.add_dataset(*dataset_args)
            for dataset_args in iterate_datasets(request.json)
        )
    )

    ds_path = app.config["datasets_path"]
    os.makedirs(os.path.dirname(ds_path), exist_ok=True)
    if not os.path.isfile(ds_path):
        with open(ds_path, "w") as datasets_file:
            json.dump({}, datasets_file)
    with open(ds_path, "r+") as datasets_file:
        datasets = json.load(datasets_file)
        datasets_file.seek(0)
        for (
            dataset_config,
            backend_name,
            organization_name,
            dataset_name,
        ) in iterate_datasets(request.json):
            datasets.setdefault(backend_name, {}).setdefault(organization_name, {})[
                dataset_name
            ] = dataset_config
        json.dump(datasets, datasets_file, indent=4, sort_keys=True)
        # remove rest of file, as it might be shorter
        datasets_file.truncate()

    return response.text("")

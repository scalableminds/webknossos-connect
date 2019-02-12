import asyncio
import json
from typing import List

import numpy as np
from sanic import Blueprint, response
from sanic.request import Request

from ...utils.json import from_json
from ...utils.types import Vec3D
from ...webknossos.access import AccessRequest, authorized
from ...webknossos.models import DataRequest as WKDataRequest

read_data = Blueprint(__name__)


@read_data.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/data",
    methods=["POST", "OPTIONS"],
)
@authorized(AccessRequest.read_dataset)
async def get_data_post(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    (backend_name, dataset) = request.app.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.backends[backend_name]

    bucket_requests = from_json(request.json, List[WKDataRequest])
    assert all(not request.fourBit for request in bucket_requests)

    buckets = await asyncio.gather(
        *(
            backend.read_data(
                dataset,
                layer_name,
                r.zoomStep,
                Vec3D(*r.position),
                Vec3D(r.cubeSize, r.cubeSize, r.cubeSize),
            )
            for r in bucket_requests
        )
    )
    missing_buckets = [index for index, data in enumerate(buckets) if data is None]
    existing_buckets = [data.flatten(order="F") for data in buckets if data is not None]
    data = (
        np.concatenate(existing_buckets).tobytes() if len(existing_buckets) > 0 else b""
    )

    headers = {
        "Access-Control-Expose-Headers": "MISSING-BUCKETS",
        "MISSING-BUCKETS": json.dumps(missing_buckets),
    }
    return response.raw(data, headers=headers)

import asyncio
from typing import List
from collections import OrderedDict

import numpy as np
from sanic import Blueprint, response
from sanic.request import Request

from ...utils.json import to_json
from ...utils.types import Vec3D
from ...webknossos.access import AccessRequest, authorized
from ...webknossos.models import (
    DataRequest as WKDataRequest,
    BoundingBox as WkBoundingBox,
    Histogram,
)

histogram = Blueprint(__name__)


@histogram.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/histogram", methods=["GET"]
)
@authorized(AccessRequest.read_dataset)
async def histogram_post(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    (backend_name, dataset) = request.app.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.backends[backend_name]

    layer = [i for i in dataset.to_webknossos().dataLayers if i.name == layer_name][0]

    import time

    before = time.time()
    sample_positions = generate_sample_positions(2, layer.boundingBox, 32)
    sample_positions_distinct = list(OrderedDict.fromkeys(sample_positions))

    bucket_requests = [
        WKDataRequest(position, 1, 32, False) for position in sample_positions_distinct
    ]

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
    existing_buckets = [
        data for r, data in zip(bucket_requests, buckets) if data is not None
    ]
    data = np.concatenate(existing_buckets) if len(existing_buckets) > 0 else b""
    data = data[np.nonzero(data)]

    after = time.time()
    print(
        f"fetched histogram data for {len(sample_positions_distinct)} positions, shape {data.shape} dtype {data.dtype}. took {(after - before)*1000:.2f} ms "
    )

    if np.issubdtype(data.dtype, np.integer):
        minimum, maximum = np.iinfo(data.dtype).min, np.iinfo(data.dtype).max
        counts, _ = np.histogram(data, bins=np.arange(maximum))
        histogram = Histogram(
            [int(c) for c in list(counts)], len(data), int(minimum), int(maximum)
        )
    else:
        raise Exception("float histograms not supported yet")

    return response.json(to_json([histogram]))


def generate_sample_positions(
    iterations: int, bounding_box: WkBoundingBox, resolution_limit: int
) -> List[Vec3D]:
    positions = []
    for exponent in range(1, iterations + 1):
        power = 2 ** exponent
        distance = Vec3D(
            int(bounding_box.width / power),
            int(bounding_box.height / power),
            int(bounding_box.depth / power),
        )
        if (
            distance.x < resolution_limit
            and distance.y < resolution_limit
            and distance.z < resolution_limit
        ):
            return positions
        top_left = bounding_box.topLeft
        positions_for_exponent = []
        for x in range(1, power):
            for y in range(1, power):
                for z in range(1, power):
                    positions_for_exponent.append(top_left + Vec3D(x, y, z) * distance)
        positions.extend(positions_for_exponent)
    return positions

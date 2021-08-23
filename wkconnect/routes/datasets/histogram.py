import asyncio
from typing import List, Tuple

import numpy as np
from sanic import Blueprint, response
from sanic.request import Request
from wkcuber.mag import Mag

from ...utils.json import to_json
from ...utils.types import Vec3D
from ...webknossos.access import AccessRequest, authorized
from ...webknossos.models import BoundingBox as WkBoundingBox
from ...webknossos.models import DataLayer as WKDataLayer
from ...webknossos.models import DataRequest as WKDataRequest
from ...webknossos.models import Histogram

histogram = Blueprint(__name__)

BUCKET_SIZE = 32


@histogram.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/histogram", methods=["GET"]
)
@authorized(AccessRequest.read_dataset)
async def histogram_post(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    (backend_name, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.ctx.backends[backend_name]

    layer = [i for i in dataset.to_webknossos().dataLayers if i.name == layer_name][0]

    sample_positions = generate_sample_positions(2, layer.boundingBox, BUCKET_SIZE)
    sample_positions, zoom_step = align_positions_with_mag(sample_positions, layer)
    sample_positions_distinct = sorted(set(sample_positions))

    bucket_requests = [
        WKDataRequest(position, zoomStep=zoom_step, cubeSize=BUCKET_SIZE, fourBit=False)
        for position in sample_positions_distinct
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

    if np.issubdtype(data.dtype, np.integer):
        minimum, maximum = np.iinfo(data.dtype).min, np.iinfo(data.dtype).max
        counts, _ = np.histogram(data, bins=np.arange(minimum, maximum))
        histogram = Histogram(
            [int(c) for c in counts], len(data), int(minimum), int(maximum)
        )
    elif np.issubdtype(data.dtype, np.floating):
        minimum, maximum = np.min(data), np.max(data)
        bucket_size = (maximum - minimum) / 255
        bucket_size = 1.0 if np.isclose(bucket_size, 0.0) else bucket_size
        counts, _ = np.histogram(data, bins=np.arange(minimum, maximum, bucket_size))
        histogram = Histogram(
            [int(c) for c in counts], len(data), float(minimum), float(maximum)
        )
    else:
        raise Exception("Histogram for data type {data.dtype} is not supported.")

    return response.json(to_json([histogram]))


def align_positions_with_mag(
    sample_positions: List[Vec3D], layer: WKDataLayer
) -> Tuple[List[Vec3D], int]:
    # For the WKW backend, the bucket requests need to be bucket-aligned in the target mag
    available_mags = sorted([Mag(mag["resolution"]) for mag in layer.wkwResolutions])
    mag = available_mags[0]

    # This is equivalent to `int(log2(mag.as_np().max()))`, but avoids intermediate floats
    zoom_step = int(mag.as_np().max()).bit_length() - 1

    align = Vec3D(*(mag.as_np() * BUCKET_SIZE))
    sample_positions = [
        Vec3D(*((position // align) * align)) for position in sample_positions
    ]
    return sample_positions, zoom_step


def generate_sample_positions(
    iterations: int, bounding_box: WkBoundingBox, resolution_limit: int
) -> List[Vec3D]:
    positions: List[Vec3D] = []
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

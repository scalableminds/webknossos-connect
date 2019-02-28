import asyncio

import numpy as np
from aiohttp import ClientSession

from wkconnect.backends.neuroglancer.backend import Neuroglancer
from wkconnect.utils.types import Vec3D

neuroglancer: Neuroglancer
data: np.ndarray


async def setup() -> None:
    global neuroglancer
    global data
    http_client = await ClientSession().__aenter__()
    neuroglancer = Neuroglancer({}, http_client)

    data_url = "https://storage.googleapis.com/neuroglancer-public-data/kasthuri2011/image/24_24_30/896-960_1152-1216_1472-1536"

    async with await http_client.get(data_url) as r:
        data = await r.read()


asyncio.run(setup())


def timeit() -> None:
    neuroglancer.decoders["jpeg"](data, "uint8", Vec3D(64, 64, 64), None)

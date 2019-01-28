import asyncio

from aiohttp import ClientSession
from py_datastore.backends.neuroglancer.backend import NeuroglancerBackend


async def setup():
    global neuroglancer
    global data
    http_client = await ClientSession().__aenter__()
    neuroglancer = NeuroglancerBackend({}, http_client)

    data_url = "https://storage.googleapis.com/neuroglancer-public-data/kasthuri2011/image/24_24_30/896-960_1152-1216_1472-1536"

    async with await http_client.get(data_url) as r:
        data = await r.read()


asyncio.run(setup())


def timeit():
    neuroglancer.decoders["jpeg"](data, "uint8", (64, 64, 64))

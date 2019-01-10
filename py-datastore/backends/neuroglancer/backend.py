import asyncio
import math
import numpy as np

from io import BytesIO
from PIL import Image

from ...utils.json import from_json
from ..backend import Backend
from .models import Dataset, Layer


class NeuroglancerBackend(Backend):
    @classmethod
    def name(cls):
        return "neuroglancer"

    def __init__(self, config, http_client):
        self.http_client = http_client
        self.decoders = {"raw": self.__decode_raw, "jpeg": self.__decode_jpeg}

    async def __handle_layer(self, layer_name, layer):
        assert layer["source"][:14] == "precomputed://"

        layer["source"] = layer["source"][14:].replace(
            "gs://", "https://storage.googleapis.com/"
        )
        info_url = layer["source"] + "/info"

        response = await self.http_client.get(info_url, response_fn=lambda r: r.json())
        response["scales"] = response["scales"][:1]  # TODO remove later
        layer.update(response)
        return (layer_name, from_json(layer, Layer))

    async def handle_new_dataset(self, organization_name, dataset_name, dataset_info):
        layers = dict(
            await asyncio.gather(
                *(
                    self.__handle_layer(*layer)
                    for layer in dataset_info["layers"].items()
                )
            )
        )
        dataset = Dataset(organization_name, dataset_name, layers)
        return dataset

    def __decode_raw(self, buffer, data_type, chunk_size):
        return np.asarray(buffer).astype(data_type).reshape(chunk_size)

    def __decode_jpeg(self, buffer, data_type, chunk_size):
        with BytesIO(buffer) as input:
            image_data = Image.open(input).getdata()
            return (
                np.asarray(list(image_data))
                .astype(data_type)
                .reshape(chunk_size, order="F")
            )

    def __chunks(self, offset, shape, scale, chunk_size):
        # clip data outside available data
        min_x = max(offset[0], scale.voxel_offset[0])
        min_y = max(offset[1], scale.voxel_offset[1])
        min_z = max(offset[2], scale.voxel_offset[2])

        max_x = min(offset[0] + shape[0], scale.voxel_offset[0] + scale.size[0])
        max_y = min(offset[1] + shape[1], scale.voxel_offset[1] + scale.size[1])
        max_z = min(offset[2] + shape[2], scale.voxel_offset[2] + scale.size[2])

        # align request to chunks
        min_x = (
            math.floor((min_x - scale.voxel_offset[0]) / chunk_size[0]) * chunk_size[0]
            + scale.voxel_offset[0]
        )
        min_y = (
            math.floor((min_y - scale.voxel_offset[1]) / chunk_size[1]) * chunk_size[1]
            + scale.voxel_offset[1]
        )
        min_z = (
            math.floor((min_z - scale.voxel_offset[2]) / chunk_size[2]) * chunk_size[2]
            + scale.voxel_offset[2]
        )

        max_x = (
            math.ceil((max_x - scale.voxel_offset[0]) / chunk_size[0]) * chunk_size[0]
            + scale.voxel_offset[0]
        )
        max_y = (
            math.ceil((max_y - scale.voxel_offset[1]) / chunk_size[1]) * chunk_size[1]
            + scale.voxel_offset[1]
        )
        max_z = (
            math.ceil((max_z - scale.voxel_offset[2]) / chunk_size[2]) * chunk_size[2]
            + scale.voxel_offset[2]
        )

        for x in range(min_x, max_x, chunk_size[0]):
            for y in range(min_y, max_y, chunk_size[1]):
                for z in range(min_z, max_z, chunk_size[2]):
                    chunk_offset = [x, y, z]
                    yield (chunk_offset, chunk_size)

    async def __read_chunk(self, layer, scale, chunk_offset, chunk_size, decoder_fn):
        url_coords = "_".join(
            [
                f"{offset}-{offset + size}"
                for offset, size in zip(chunk_offset, chunk_size)
            ]
        )
        data_url = f"{layer.source}/{scale.key}/{url_coords}"

        response_buffer = await self.http_client.get(
            data_url, response_fn=lambda r: r.read()
        )
        chunk_data = decoder_fn(response_buffer, layer.data_type, chunk_size)
        return (chunk_offset, chunk_size, chunk_data)

    def __cutout(self, chunks, data_type, offset, shape):
        result = np.zeros(shape, dtype=data_type, order="F")

        for chunk_offset, chunk_size, chunk_data in chunks:
            min_x = max(offset[0], chunk_offset[0])
            min_y = max(offset[1], chunk_offset[1])
            min_z = max(offset[2], chunk_offset[2])

            max_x = min(offset[0] + shape[0], chunk_offset[0] + chunk_size[0])
            max_y = min(offset[1] + shape[1], chunk_offset[1] + chunk_size[1])
            max_z = min(offset[2] + shape[2], chunk_offset[2] + chunk_size[2])

            result[
                min_x - offset[0] : max_x - offset[0],
                min_y - offset[1] : max_y - offset[1],
                min_z - offset[2] : max_z - offset[2],
            ] = chunk_data[
                min_x - chunk_offset[0] : max_x - chunk_offset[0],
                min_y - chunk_offset[1] : max_y - chunk_offset[1],
                min_z - chunk_offset[2] : max_z - chunk_offset[2],
            ]

        return result

    async def read_data(self, dataset, layer_name, resolution, offset, shape):
        layer = dataset.layers[layer_name]
        # scale = next(scale for scale in layer.scales if scale.resolution == resolution)
        scale = layer.scales[0]
        decoder = self.decoders[scale.encoding]
        chunk_size = scale.chunk_sizes[0]  # TODO

        chunk_coords = self.__chunks(offset, shape, scale, chunk_size)
        chunks = await asyncio.gather(
            *(
                self.__read_chunk(layer, scale, chunk_offset, chunk_size, decoder)
                for chunk_offset, chunk_sizes in chunk_coords
            )
        )

        return self.__cutout(chunks, layer.data_type, offset, shape)

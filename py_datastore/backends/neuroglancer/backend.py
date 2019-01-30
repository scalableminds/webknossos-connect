import asyncio
import jpeg4py as jpeg
import math
import numpy as np

from aiohttp import ClientSession, ClientResponseError
from async_lru import alru_cache
from io import BytesIO
from typing import cast, Any, Callable, Dict, Iterable, Tuple

from .models import Dataset, Layer, Scale
from ..backend import Backend, DatasetInfo
from ...utils.json import from_json
from ...utils.types import JSON, Vec3D


DecoderFn = Callable[[bytes, str, Vec3D], np.ndarray]
Chunk = Tuple[Vec3D, Vec3D, np.ndarray]


class NeuroglancerBackend(Backend):
    @classmethod
    def name(cls) -> str:
        return "neuroglancer"

    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        self.http_client = http_client
        self.decoders: Dict[str, DecoderFn] = {
            "raw": self.__decode_raw,
            "jpeg": self.__decode_jpeg,
            "compressed_segmentation": self.__decode_compressed_segmentation,
        }

    async def __handle_layer(
        self, layer_name: str, layer: Dict[str, Any]
    ) -> Tuple[str, Layer]:
        assert layer["source"][:14] == "precomputed://"

        layer["source"] = layer["source"][14:].replace(
            "gs://", "https://storage.googleapis.com/"
        )
        info_url = layer["source"] + "/info"

        async with await self.http_client.get(info_url) as r:
            response = await r.json()
        layer.update(response)
        return (layer_name, from_json(layer, Layer))

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
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

    def __decode_raw(
        self, buffer: bytes, data_type: str, chunk_size: Vec3D
    ) -> np.ndarray:
        return np.asarray(buffer).astype(data_type).reshape(chunk_size)

    def __decode_jpeg(
        self, buffer: bytes, data_type: str, chunk_size: Vec3D
    ) -> np.ndarray:
        np_bytes = np.fromstring(buffer, dtype="uint8")
        return (
            jpeg.JPEG(np_bytes)
            .decode()[:, :, 0]
            .astype(data_type)
            .T.reshape(chunk_size, order="F")
        )

    def __decode_compressed_segmentation(
        self, buffer: bytes, data_type: str, chunk_size: Vec3D
    ) -> np.ndarray:
        array = np.zeros(chunk_size, dtype=data_type)
        array.fill(2)
        array[:, :, : chunk_size[2] // 2] = 3
        return array

    def __chunks(
        self, offset: Vec3D, shape: Vec3D, scale: Scale, chunk_size: Vec3D
    ) -> Iterable[Tuple[Vec3D, Vec3D]]:
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
                    chunk_offset = (x, y, z)
                    yield (chunk_offset, chunk_size)

    @alru_cache(maxsize=2 ** 12)
    async def __read_chunk(
        self,
        layer: Layer,
        scale_key: str,
        chunk_offset: Vec3D,
        chunk_size: Vec3D,
        decoder_fn: DecoderFn,
    ) -> Chunk:
        url_coords = "_".join(
            [
                f"{offset}-{offset + size}"
                for offset, size in zip(chunk_offset, chunk_size)
            ]
        )
        data_url = f"{layer.source}/{scale_key}/{url_coords}"

        try:
            async with await self.http_client.get(data_url) as r:
                response_buffer = await r.read()
        except ClientResponseError:
            chunk_data = np.zeros(chunk_size, dtype=layer.wk_data_type())
        else:
            chunk_data = decoder_fn(
                response_buffer, layer.data_type, chunk_size
            ).astype(layer.wk_data_type())
        return (chunk_offset, chunk_size, chunk_data)

    def __cutout(
        self, chunks: Iterable[Chunk], data_type: str, offset: Vec3D, shape: Vec3D
    ) -> np.ndarray:
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

    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        zoomStep: int,
        wk_offset: Vec3D,
        wk_shape: Vec3D,
    ) -> np.ndarray:
        dataset = cast(Dataset, dataset)
        layer = dataset.layers[layer_name]
        # TODO add lookup table for scale for efficiency
        def fits_resolution(scale: Scale) -> bool:
            wk_resolution = tuple(
                res_dim // scale_dim
                for res_dim, scale_dim in zip(
                    scale.resolution, cast(Dataset, dataset).scale
                )
            )
            return max(wk_resolution) == 2 ** zoomStep

        scale = next(scale for scale in layer.scales if fits_resolution(scale))
        decoder = self.decoders[scale.encoding]
        chunk_size = scale.chunk_sizes[0]  # TODO

        def wk_to_neuroglancer(vec: Vec3D) -> Vec3D:
            return cast(
                Vec3D,
                tuple(
                    vec_dim * scale_dim // res_dim
                    for vec_dim, scale_dim, res_dim in zip(
                        vec, cast(Dataset, dataset).scale, scale.resolution
                    )
                ),
            )

        offset = wk_to_neuroglancer(wk_offset)
        shape = wk_to_neuroglancer(wk_shape)

        chunk_coords = self.__chunks(offset, shape, scale, chunk_size)
        chunks = await asyncio.gather(
            *(
                self.__read_chunk(layer, scale.key, chunk_offset, chunk_size, decoder)
                for chunk_offset, chunk_sizes in chunk_coords
            )
        )

        return self.__cutout(chunks, layer.wk_data_type(), offset, wk_shape)

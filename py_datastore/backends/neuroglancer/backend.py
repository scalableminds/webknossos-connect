import asyncio
import compressed_segmentation
import jpeg4py as jpeg
import math
import numpy as np

from aiohttp import ClientSession, ClientResponseError
from async_lru import alru_cache
from io import BytesIO
from typing import cast, Any, Callable, Dict, Iterable, Optional, Tuple

from .models import Dataset, Layer, Scale
from ..backend import Backend, DatasetInfo
from ...utils.json import from_json
from ...utils.types import JSON, Vec3D


DecoderFn = Callable[[bytes, str, Vec3D, Optional[Vec3D]], np.ndarray]
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
        self, buffer: bytes, data_type: str, chunk_size: Vec3D, _: Optional[Vec3D]
    ) -> np.ndarray:
        return np.asarray(buffer).astype(data_type).reshape(chunk_size)

    def __decode_jpeg(
        self, buffer: bytes, data_type: str, chunk_size: Vec3D, _: Optional[Vec3D]
    ) -> np.ndarray:
        np_bytes = np.fromstring(buffer, dtype="uint8")
        return (
            jpeg.JPEG(np_bytes)
            .decode()[:, :, 0]
            .astype(data_type)
            .T.reshape(chunk_size, order="F")
        )

    def __decode_compressed_segmentation(
        self,
        buffer: bytes,
        data_type: str,
        chunk_size: Vec3D,
        block_size: Optional[Vec3D],
    ) -> np.ndarray:
        assert block_size is not None
        return compressed_segmentation.decompress(
            buffer, chunk_size, data_type, block_size, order="F"
        )

    def __chunks(
        self, offset: Vec3D, shape: Vec3D, scale: Scale, chunk_size: Vec3D
    ) -> Iterable[Tuple[Vec3D, Vec3D]]:
        # clip data outside available data
        min_inside = offset.pairmax(scale.voxel_offset)
        max_inside = (offset + shape).pairmin(scale.voxel_offset + scale.size)

        # align request to chunks
        min_aligned = (
            min_inside - scale.voxel_offset
        ) // chunk_size * chunk_size + scale.voxel_offset
        max_aligned = (max_inside - scale.voxel_offset).ceildiv(
            chunk_size
        ) * chunk_size + scale.voxel_offset

        for x in range(min_aligned.x, max_aligned.x, chunk_size.x):
            for y in range(min_aligned.y, max_aligned.y, chunk_size.y):
                for z in range(min_aligned.z, max_aligned.z, chunk_size.z):
                    chunk_offset = Vec3D(x, y, z)
                    # The size is at most chunk_size but capped to fit the dataset:
                    capped_chunk_size = chunk_size.pairmin(scale.size - chunk_offset)
                    yield (chunk_offset, capped_chunk_size)

    @alru_cache(maxsize=2 ** 12)
    async def __read_chunk(
        self,
        layer: Layer,
        scale_key: str,
        chunk_offset: Vec3D,
        chunk_size: Vec3D,
        decoder_fn: DecoderFn,
        compressed_segmentation_block_size: Optional[Vec3D],
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
                response_buffer,
                layer.data_type,
                chunk_size,
                compressed_segmentation_block_size,
            ).astype(layer.wk_data_type())
        return (chunk_offset, chunk_size, chunk_data)

    def __cutout(
        self, chunks: Iterable[Chunk], data_type: str, offset: Vec3D, shape: Vec3D
    ) -> np.ndarray:
        result = np.zeros(shape, dtype=data_type, order="F")

        for chunk_offset, chunk_size, chunk_data in chunks:
            inner_min = offset.pairmax(chunk_offset)
            inner_max = (offset + shape).pairmin(chunk_offset + chunk_size)

            rel_min = inner_min - offset
            rel_max = inner_max - offset
            chunk_min = inner_min - chunk_offset
            chunk_max = inner_max - chunk_offset

            result[
                rel_min.x : rel_max.x, rel_min.y : rel_max.y, rel_min.z : rel_max.z
            ] = chunk_data[
                chunk_min.x : chunk_max.x,
                chunk_min.y : chunk_max.y,
                chunk_min.z : chunk_max.z,
            ]

        return result

    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        zoomStep: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> np.ndarray:
        neuroglancer_dataset = cast(Dataset, dataset)
        layer = neuroglancer_dataset.layers[layer_name]
        # TODO add lookup table for scale for efficiency
        def fits_resolution(scale: Scale) -> bool:
            wk_resolution = scale.resolution // neuroglancer_dataset.scale
            return max(wk_resolution) == 2 ** zoomStep

        scale = next(scale for scale in layer.scales if fits_resolution(scale))
        decoder = self.decoders[scale.encoding]
        chunk_size = scale.chunk_sizes[0]  # TODO

        offset = wk_offset * neuroglancer_dataset.scale // scale.resolution

        chunk_coords = self.__chunks(offset, shape, scale, chunk_size)
        chunks = await asyncio.gather(
            *(
                self.__read_chunk(
                    layer,
                    scale.key,
                    chunk_offset,
                    chunk_size,
                    decoder,
                    scale.compressed_segmentation_block_size,
                )
                for chunk_offset, chunk_size in chunk_coords
            )
        )

        return self.__cutout(chunks, layer.wk_data_type(), offset, shape)

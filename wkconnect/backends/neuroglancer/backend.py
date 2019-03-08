import asyncio
from typing import Any, Callable, Dict, Optional, Tuple, cast

import jpeg4py as jpeg
import numpy as np
from aiohttp import ClientResponseError, ClientSession
from async_lru import alru_cache

import compressed_segmentation

from ...utils.json import from_json
from ...utils.types import JSON, Box3D, Vec3D
from ..backend import Backend, Chunk, DatasetInfo
from .models import Dataset, Layer

DecoderFn = Callable[[bytes, str, Vec3D, Optional[Vec3D]], np.ndarray]


class Neuroglancer(Backend):
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
        layer["source"] = layer["source"].replace(
            "gs://", "https://storage.googleapis.com/"
        )
        info_url = layer["source"] + "/info"

        async with await self.http_client.get(info_url) as r:
            response = await r.json()
        layer.update(response)
        layer["scales"] = sorted(layer["scales"], key=lambda scale: scale["resolution"])
        min_resolution = Vec3D(*layer["scales"][0]["resolution"])
        for scale in layer["scales"]:
            scale["resolution"] = Vec3D(*scale["resolution"]) // min_resolution
        layer["relative_scale"] = min_resolution
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

    @alru_cache(maxsize=2 ** 12)
    async def __read_chunk(
        self,
        layer: Layer,
        scale_key: str,
        chunk_box: Box3D,
        decoder_fn: DecoderFn,
        compressed_segmentation_block_size: Optional[Vec3D],
    ) -> Chunk:
        url_coords = "_".join(
            f"{left_dim}-{right_dim}" for left_dim, right_dim in zip(*chunk_box)
        )
        data_url = f"{layer.source}/{scale_key}/{url_coords}"

        async with await self.http_client.get(data_url) as r:
            response_buffer = await r.read()
        chunk_data = decoder_fn(
            response_buffer,
            layer.data_type,
            chunk_box.size(),
            compressed_segmentation_block_size,
        ).astype(layer.wk_data_type())
        return (chunk_box, chunk_data)

    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        neuroglancer_dataset = cast(Dataset, dataset)
        layer = neuroglancer_dataset.layers[layer_name]
        scale = layer.scales[zoom_step]
        decoder = self.decoders[scale.encoding]

        # convert to coordinate system of current scale
        offset = wk_offset // scale.resolution
        box = Box3D.from_size(offset, shape)

        chunk_boxes = self._chunks(box, scale.box(), scale.chunk_sizes[0])
        try:
            chunks = await asyncio.gather(
                *(
                    self.__read_chunk(
                        layer,
                        scale.key,
                        chunk_box,
                        decoder,
                        scale.compressed_segmentation_block_size,
                    )
                    for chunk_box in chunk_boxes
                )
            )
        except ClientResponseError:
            # will be reported in MISSING-BUCKETS, frontend will retry
            return None

        return self._cutout(chunks, box)

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        self.__read_chunk.cache_clear()  # pylint: disable=no-member

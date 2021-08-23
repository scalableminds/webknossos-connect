import asyncio
import json
import os
import tempfile
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import compressed_segmentation
import jpeg4py as jpeg
import numpy as np
from aiohttp import ClientResponseError, ClientSession
from async_lru import alru_cache
from gcloud.aio.auth import Token

from ...utils.exceptions import RuntimeErrorWithUserMessage
from ...utils.json import from_json
from ...utils.types import JSON, Box3D, Vec3D, Vec3Df
from ..backend import Backend, DatasetInfo
from .models import Dataset, Layer, Scale
from .sharding import MinishardInfo, ShardingSpec

DecoderFn = Callable[[bytes, str, Vec3D, Optional[Vec3D]], np.ndarray]
Chunk = Tuple[Box3D, np.ndarray]


class NeuroglancerAuthenticationError(RuntimeError):
    pass


class NeuroglancerAuthenticationMissingError(RuntimeErrorWithUserMessage):
    pass


async def get_header(token: Optional[Token]) -> Dict[str, str]:
    if token is None:
        return {}
    else:
        try:
            token_str = await token.get()
        except Exception as e:
            raise NeuroglancerAuthenticationError(*e.args)
        if token_str is None:
            return {}
        return {"Authorization": "Bearer " + token_str}


class Neuroglancer(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        self.http_client = http_client
        self.decoders: Dict[str, DecoderFn] = {
            "raw": self.__decode_raw,
            "jpeg": self.__decode_jpeg,
            "compressed_segmentation": self.__decode_compressed_segmentation,
        }

    def create_token(self, credentials: Optional[JSON]) -> Optional[Token]:
        if credentials is None:
            return None
        tmpfile = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        json.dump(credentials, tmpfile)
        tmpfile.close()
        try:
            token = Token(
                service_file=tmpfile.name,
                session=self.http_client,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        finally:
            os.remove(tmpfile.name)
        return token

    async def __handle_layer(
        self, layer_name: str, layer: Dict[str, Any], token: Optional[Token]
    ) -> Tuple[str, Layer]:
        layer["source"] = (
            layer["source"]
            .replace("gs://", "https://storage.googleapis.com/")
            .replace("s3://", "https://s3.amazonaws.com/")
        )
        info_url = layer["source"] + "/info"

        try:
            async with await self.http_client.get(
                info_url, headers=await get_header(token)
            ) as r:
                response = await r.json(content_type=None)
        except ClientResponseError as e:
            if e.status == 403:
                raise NeuroglancerAuthenticationMissingError(
                    "Could not authenticate the neuroglancer dataset, "
                    + "please add authentication"
                )
            else:
                raise e
        layer.update(response)
        layer["scales"] = sorted(layer["scales"], key=lambda scale: scale["resolution"])
        min_resolution = Vec3Df(*layer["scales"][0]["resolution"])

        for scale in layer["scales"]:
            resolution = Vec3Df(*scale["resolution"])
            assert resolution % min_resolution == Vec3D.zeros()
            scale["resolution"] = (resolution // min_resolution).to_int()
            scale["voxel_offset"] = Vec3D(*scale["voxel_offset"]).to_int()
            if (
                "sharding" in scale
                and scale["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"
            ):
                scale["sharding"] = ShardingSpec(
                    dataset_size=Vec3D(*scale["size"]),
                    chunk_size=Vec3D(*scale["chunk_sizes"][0]),
                    preshift_bits=int(scale["sharding"]["preshift_bits"]),
                    shard_bits=int(scale["sharding"]["shard_bits"]),
                    minishard_bits=int(scale["sharding"]["minishard_bits"]),
                    minishard_index_encoding=scale["sharding"][
                        "minishard_index_encoding"
                    ],
                    hashfn=scale["sharding"]["hash"],
                )

        layer["relative_scale"] = min_resolution.to_int()
        return (layer_name, from_json(layer, Layer))

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        try:
            token = self.create_token(dataset_info.get("credentials", None))
        except Exception as e:
            NeuroglancerAuthenticationError(*e.args)
        layers = dict(
            await asyncio.gather(
                *(
                    self.__handle_layer(layer_name, layer, token)
                    for layer_name, layer in dataset_info["layers"].items()
                )
            )
        )
        dataset = Dataset(organization_name, dataset_name, layers, token)
        return dataset

    def __decode_raw(
        self, buffer: bytes, data_type: str, chunk_size: Vec3D, _: Optional[Vec3D]
    ) -> np.ndarray:
        return np.frombuffer(buffer, dtype=data_type).reshape(chunk_size, order="F")

    def __decode_jpeg(
        self, buffer: bytes, data_type: str, chunk_size: Vec3D, _: Optional[Vec3D]
    ) -> np.ndarray:
        np_bytes = np.frombuffer(buffer, dtype="uint8")
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
        self, requested: Box3D, ds_frame: Box3D, wk_chunk_size: Vec3D
    ) -> Iterable[Box3D]:
        # clip data outside available data
        inside = requested.intersect(ds_frame)

        # align request to chunks
        aligned = (inside - ds_frame.left).div(
            wk_chunk_size
        ) * wk_chunk_size + ds_frame.left

        for chunk_offset in aligned.range(offset=wk_chunk_size):
            chunk = Box3D.from_size(chunk_offset, wk_chunk_size)
            # chunk is capped to fit the dataset:
            yield chunk.intersect(ds_frame)

    async def __read_ranged_data(
        self, url: str, range: Tuple[int, int], token: Optional[Token]
    ) -> bytes:
        async with await self.http_client.get(
            url,
            headers={
                **await get_header(token),
                "Range": f"bytes={int(range[0])}-{int(range[1]-1)}",
            },
        ) as r:
            response_buffer = await r.read()
        return response_buffer

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __read_shard_index(
        self, shard_url: str, shard_spec: ShardingSpec, token: Optional[Token]
    ) -> np.ndarray:
        shard_index_range = shard_spec.get_shard_index_range()
        return shard_spec.parse_shard_index(
            await self.__read_ranged_data(shard_url, shard_index_range, token)
        )

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __read_minishard_index(
        self,
        shard_url: str,
        shard_spec: ShardingSpec,
        minishard_info: MinishardInfo,
        token: Optional[Token],
    ) -> np.ndarray:
        minishard_index_range = shard_spec.get_minishard_index_range(
            minishard_info.minishard_number,
            await self.__read_shard_index(shard_url, shard_spec, token),
        )
        return shard_spec.parse_minishard_index(
            await self.__read_ranged_data(shard_url, minishard_index_range, token)
        )

    async def __read_sharded_chunk(
        self,
        source_url: str,
        shard_spec: ShardingSpec,
        position: Vec3D,
        token: Optional[Token],
    ) -> bytes:
        chunk_id = shard_spec.get_chunk_key(position)
        minishard_info = shard_spec.get_minishard_info(chunk_id)
        shard_url = (
            f"{source_url}/{shard_spec.format_shard_for_url(minishard_info)}.shard"
        )
        print(minishard_info)
        minishard_index = await self.__read_minishard_index(
            shard_url, shard_spec, minishard_info, token
        )
        print(minishard_index)
        chunk_entry = minishard_index[minishard_index[:, 0] == chunk_id][0]
        print(chunk_entry)
        buf = await self.__read_ranged_data(
            shard_url, (chunk_entry[1], chunk_entry[2] + chunk_entry[1]), token
        )
        return buf

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __read_chunk(
        self,
        layer: Layer,
        scale: Scale,
        chunk_box: Box3D,
        decoder_fn: DecoderFn,
        token: Optional[Token],
    ) -> Chunk:
        response_buffer = None
        if scale.sharding is not None:
            response_buffer = await self.__read_sharded_chunk(
                layer.source, scale.sharding, chunk_box.left, token
            )
        else:
            url_coords = "_".join(
                f"{left_dim}-{right_dim}" for left_dim, right_dim in zip(*chunk_box)
            )
            data_url = f"{layer.source}/{scale.key}/{url_coords}"

            async with await self.http_client.get(
                data_url, headers=await get_header(token)
            ) as r:
                response_buffer = await r.read()

        chunk_data = decoder_fn(
            response_buffer,
            layer.data_type,
            chunk_box.size(),
            scale.compressed_segmentation_block_size,
        ).astype(layer.wk_data_type())
        return (chunk_box, chunk_data)

    def __cutout(self, chunks: List[Chunk], box: Box3D) -> np.ndarray:
        result = np.zeros(box.size(), dtype=chunks[0][1].dtype, order="F")
        for chunk_box, chunk_data in chunks:
            inner = chunk_box.intersect(box)
            result[(inner - box.left).np_slice()] = chunk_data[
                (inner - chunk_box.left).np_slice()
            ]
        return result

    async def read_data(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        dataset = cast(Dataset, abstract_dataset)
        layer = dataset.layers[layer_name]
        # we can use zoom_step as the index, as the scales are sorted
        # see assertion in the Layer initialization
        scale = layer.scales[zoom_step]
        decoder = self.decoders[scale.encoding]

        # convert to coordinate system of current scale
        offset = wk_offset // scale.resolution
        box = Box3D.from_size(offset, shape)

        chunk_boxes = self.__chunks(box, scale.box(), scale.chunk_sizes[0])
        try:
            chunks = await asyncio.gather(
                *(
                    self.__read_chunk(layer, scale, chunk_box, decoder, dataset.token)
                    for chunk_box in chunk_boxes
                )
            )
        except ClientResponseError:
            # will be reported in MISSING-BUCKETS, frontend will retry
            return None

        return self.__cutout(chunks, box)

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        self.__read_chunk.cache_clear()  # pylint: disable=no-member
        self.__read_shard_index.cache_clear()  # pylint: disable=no-member
        self.__read_minishard_index.cache_clear()  # pylint: disable=no-member

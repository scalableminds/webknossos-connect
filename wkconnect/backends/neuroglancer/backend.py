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
from ...utils.types import JSON, Box3D, Vec3D, Vec3Df
from ..backend import Backend, DatasetInfo, MeshfileInfo
from ..neuroglancer.meshes import Meshfile, MeshfileLod, MeshInfo
from .models import Dataset, Layer, Scale
from .sharding import MinishardInfo, ShardingInfo

DecoderFn = Callable[[bytes, str, Vec3D, Optional[Vec3D]], np.ndarray]
Chunk = Tuple[Box3D, np.ndarray]

MESH_LOD = 2
MESH_NAME = "mesh"


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


def select_lod(meshfile: Meshfile) -> MeshfileLod:
    return meshfile.lods[min(len(meshfile.lods) - 1, MESH_LOD)]


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

        if "mesh" in layer:
            mesh_info_url = layer["source"] + "/" + layer["mesh"] + "/info"
            try:
                async with await self.http_client.get(
                    mesh_info_url, headers=await get_header(token)
                ) as r:
                    mesh_info = await r.json(content_type=None)
                    layer["mesh"] = MeshInfo.parse(mesh_info)
            except ClientResponseError as e:
                if e.status == 403:
                    raise NeuroglancerAuthenticationMissingError(
                        "Could not authenticate the neuroglancer dataset, "
                        + "please add authentication"
                    )
                elif e.status == 404:
                    del layer["mesh"]
                else:
                    raise e

        layer["scales"] = sorted(layer["scales"], key=lambda scale: scale["resolution"])
        min_resolution = Vec3Df(*layer["scales"][0]["resolution"])

        new_scales: List[Scale] = []
        for scale in layer["scales"]:
            resolution = Vec3Df(*scale["resolution"])
            chunk_size = Vec3D(*scale["chunk_sizes"][0])
            dataset_size = Vec3D(*scale["size"])
            assert resolution % min_resolution == Vec3Df.zeros()
            new_scales.append(
                Scale(
                    chunk_size=chunk_size,
                    encoding=scale["encoding"],
                    key=scale["key"],
                    mag=(resolution // min_resolution).to_int(),
                    size=dataset_size,
                    voxel_offset=(
                        Vec3D(*scale["voxel_offset"]).to_int()
                        if "voxel_offset" in scale
                        else Vec3D.zeros()
                    ),
                    sharding=ShardingInfo.parse(
                        scale["sharding"], dataset_size, chunk_size
                    )
                    if "sharding" in scale
                    else None,
                    compressed_segmentation_block_size=Vec3D(
                        *scale["compressed_segmentation_block_size"]
                    )
                    if "compressed_segmentation_block_size" in scale
                    else None,
                )
            )

        return (
            layer_name,
            Layer(
                source=layer["source"],
                data_type=layer["data_type"],
                num_channels=int(layer["num_channels"]),
                scales=tuple(new_scales),
                resolution=min_resolution,
                type=layer["type"],
                mesh=layer.get("mesh"),
            ),
        )

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
        layers, scale = Dataset.fix_scales(layers)
        dataset = Dataset(organization_name, dataset_name, layers, scale, token)
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
            buffer,
            chunk_size.as_tuple(),
            dtype=data_type,
            block_size=block_size,
            order="F",
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
        if range[0] == range[1]:
            return b""
        headers = {
            **await get_header(token),
            "Range": f"bytes={int(range[0])}-{int(range[1]-1)}",
        }
        async with await self.http_client.get(url, headers=headers) as r:
            r.raise_for_status()
            response_buffer = await r.read()
        return response_buffer

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __read_shard_index(
        self, shard_url: str, sharding_info: ShardingInfo, token: Optional[Token]
    ) -> np.ndarray:
        shard_index_range = sharding_info.get_shard_index_range()
        index = sharding_info.parse_shard_index(
            await self.__read_ranged_data(shard_url, shard_index_range, token)
        )
        return index

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __read_minishard_index(
        self,
        shard_url: str,
        sharding_info: ShardingInfo,
        minishard_info: MinishardInfo,
        token: Optional[Token],
    ) -> np.ndarray:
        shard_index = await self.__read_shard_index(shard_url, sharding_info, token)
        minishard_index_range = sharding_info.get_minishard_index_range(
            minishard_info.minishard_number, shard_index
        )
        buf = await self.__read_ranged_data(shard_url, minishard_index_range, token)
        index = sharding_info.parse_minishard_index(buf)
        return index

    async def __read_sharded_chunk(
        self,
        source_url: str,
        sharding_info: ShardingInfo,
        position: Vec3D,
        token: Optional[Token],
    ) -> Optional[bytes]:
        chunk_id = sharding_info.get_chunk_key(position)
        minishard_info = sharding_info.get_minishard_info(chunk_id)
        shard_url = (
            f"{source_url}/{sharding_info.format_shard_for_url(minishard_info)}.shard"
        )
        minishard_index = await self.__read_minishard_index(
            shard_url, sharding_info, minishard_info, token
        )
        chunk_range = sharding_info.get_chunk_range(chunk_id, minishard_index)
        if chunk_range is None:
            return None
        buf = sharding_info.parse_chunk(
            await self.__read_ranged_data(shard_url, chunk_range, token)
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
                f"{layer.source}/{scale.key}", scale.sharding, chunk_box.left, token
            )
            if response_buffer is None:
                return (
                    chunk_box,
                    np.zeros(chunk_box.size().as_tuple(), dtype=layer.wk_data_type()),
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
        scale = [
            scale for scale in layer.scales if 2 ** zoom_step == scale.mag.as_np().max()
        ][0]
        decoder = self.decoders[scale.encoding]

        # convert to coordinate system of current scale
        offset = wk_offset // scale.mag
        box = Box3D.from_size(offset, shape)

        chunk_boxes = self.__chunks(box, scale.box(), scale.chunk_size)
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

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __read_meshfile(
        self, layer: Layer, segment_id: int, token: Optional[Token]
    ) -> Optional[Meshfile]:
        assert layer.mesh is not None
        mesh_info = layer.mesh
        sharding_info = mesh_info.sharding
        chunk_id = np.uint64(segment_id)
        minishard_info = sharding_info.get_minishard_info(chunk_id)
        shard_url = f"{layer.source}/mesh/{sharding_info.format_shard_for_url(minishard_info)}.shard"
        minishard_index = await self.__read_minishard_index(
            shard_url, sharding_info, minishard_info, token
        )
        chunk_range = sharding_info.get_chunk_range(chunk_id, minishard_index)
        # print("chunk_range", chunk_range)
        if chunk_range is None:
            return None
        buf = sharding_info.parse_chunk(
            await self.__read_ranged_data(shard_url, chunk_range, token)
        )
        return mesh_info.parse_meshfile(buf, chunk_range[0])

    async def __read_mesh_data(
        self,
        layer: Layer,
        meshfile: Meshfile,
        segment_id: int,
        position: Vec3D,
        token: Optional[Token],
    ) -> Optional[bytes]:
        assert layer.mesh is not None
        sharding_info = layer.mesh.sharding
        fragment = next(
            fragment
            for fragment in select_lod(meshfile).fragments
            if fragment.position.to_int() == position
        )
        chunk_id = np.uint64(segment_id)
        minishard_info = sharding_info.get_minishard_info(chunk_id)
        shard_url = f"{layer.source}/mesh/{sharding_info.format_shard_for_url(minishard_info)}.shard"
        buf = await self.__read_ranged_data(
            shard_url,
            (fragment.byte_offset, fragment.byte_offset + fragment.byte_size),
            token,
        )
        return meshfile.decode_data(fragment, buf)

    async def get_meshes(
        self, abstract_dataset: DatasetInfo, layer_name: str
    ) -> Optional[List[MeshfileInfo]]:
        dataset = cast(Dataset, abstract_dataset)
        if layer_name in dataset.layers and dataset.layers[layer_name].mesh:
            return [MeshfileInfo(mesh_file_name=MESH_NAME)]
        return []

    async def get_chunks_for_mesh(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        mesh_name: str,
        segment_id: int,
    ) -> Optional[List[Vec3D]]:
        dataset = cast(Dataset, abstract_dataset)
        layer = dataset.layers[layer_name]
        assert layer.mesh is not None
        assert mesh_name == MESH_NAME
        meshfile = await self.__read_meshfile(layer, segment_id, dataset.token)
        if meshfile is None:
            return None
        return [
            fragment.position.to_int() for fragment in select_lod(meshfile).fragments
        ]

    async def get_chunk_data_for_mesh(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        mesh_name: str,
        segment_id: int,
        position: Vec3D,
    ) -> Optional[bytes]:
        dataset = cast(Dataset, abstract_dataset)
        layer = dataset.layers[layer_name]
        assert layer.mesh is not None
        assert mesh_name == MESH_NAME
        meshfile = await self.__read_meshfile(layer, segment_id, dataset.token)
        if meshfile is None:
            return None
        return await self.__read_mesh_data(
            layer, meshfile, segment_id, position, dataset.token
        )

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        self.__read_chunk.cache_clear()  # pylint: disable=no-member
        self.__read_shard_index.cache_clear()  # pylint: disable=no-member
        self.__read_minishard_index.cache_clear()  # pylint: disable=no-member
        self.__read_meshfile.cache_clear()  # pylint: disable=no-member

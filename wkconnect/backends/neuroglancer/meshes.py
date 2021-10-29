import asyncio
from aiohttp import ClientSession
from dataclasses import dataclass, replace
import numpy as np
from async_lru import alru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast


from .sharding import MinishardInfo, ShardingInfo
from ...utils.types import Vec3D, Vec3Df
from wkconnect.backends.neuroglancer import sharding


@dataclass(frozen=True)
class MeshfileFragment:
    position: Vec3Df
    byte_offset: int
    byte_size: int


@dataclass(frozen=True)
class MeshfileLod:
    scale: int
    vertex_offset: Vec3Df
    chunk_shape: Vec3Df
    fragments: List[MeshfileFragment]


@dataclass(frozen=True)
class Meshfile:
    chunk_shape: Vec3Df
    grid_origin: Vec3Df
    lods: List[MeshfileLod]

    def fragment_offset_and_shape(self, lod: int, fragment: int):
        return (
            self.grid_origin
            + self.lods[lod].fragments[fragment].position
            * self.chunk_shape
            * self.lods[lod].scale,
            self.grid_origin
            + self.lods[lod].fragments[fragment].position
            * self.chunk_shape
            * self.lods[lod].scale
            + self.chunk_shape * self.lods[lod].scale,
            self.chunk_shape * self.lods[lod].scale,
        )


@dataclass(frozen=True)
class MeshInfo:
    sharding: ShardingInfo
    transform: np.ndarray
    vertex_quantization_bits: int
    lod_scale_multiplier: int = 1

    def parse_meshfile(self, buf: bytes, byte_offset: int):
        p = 0
        chunk_shape = np.frombuffer(buf[p : p + 12], dtype="f4").copy()
        p += 12
        grid_origin = np.frombuffer(buf[p : p + 12], dtype="f4").copy()
        p += 12
        num_lods = int(np.frombuffer(buf[p : p + 4], dtype="<u4").copy())
        p += 4
        lod_scales = np.frombuffer(buf[p : (p + num_lods * 4)], dtype="f4").copy()
        p += num_lods * 4
        vertex_offsets = (
            np.frombuffer(buf[p : (p + num_lods * 3 * 4)], dtype="f4")
            .reshape((-1, 3))
            .copy()
        )
        p += num_lods * 3 * 4
        num_fragments_per_lod = np.frombuffer(
            buf[p : (p + num_lods * 4)], dtype="<u4"
        ).copy()
        p += num_lods * 4
        lods = []
        frag_byte_offset = 0
        for lod in range(num_lods):
            if num_fragments_per_lod[lod] > 0:
                fragment_positions = (
                    np.frombuffer(
                        buf[p : (p + int(num_fragments_per_lod[lod]) * 3 * 4)],
                        dtype="<u4",
                    )
                    .reshape((3, -1))
                    .T.copy()
                )
                p += int(num_fragments_per_lod[lod]) * 3 * 4
                fragment_byte_sizes = np.frombuffer(
                    buf[p : (p + int(num_fragments_per_lod[lod]) * 4)], dtype="<u4"
                ).copy()
                fragment_byte_offsets = np.zeros(
                    (num_fragments_per_lod[lod],), dtype="<u4"
                )
                fragment_byte_offsets[0] = frag_byte_offset
                fragment_byte_offsets[1:] = frag_byte_offset + np.cumsum(
                    fragment_byte_sizes[:-1]
                )
                frag_byte_offset += np.sum(fragment_byte_sizes)
                p += int(num_fragments_per_lod[lod]) * 4

                fragments = []
                for i in range(num_fragments_per_lod[lod]):
                    position = fragment_positions[i]
                    global_position = (
                        grid_origin
                        + fragment_positions[i] * chunk_shape * lod_scales[lod]
                    )
                    fragments.append(
                        MeshfileFragment(
                            position=Vec3Df(
                                *np.matmul(global_position, self.transform)[0:3]
                            ),
                            byte_offset=fragment_byte_offsets[i],
                            byte_size=int(fragment_byte_sizes[i]),
                        )
                    )

                lods.append(
                    MeshfileLod(
                        scale=int(lod_scales[lod]),
                        vertex_offset=Vec3Df(*vertex_offsets[lod]),
                        chunk_shape=Vec3Df(*chunk_shape * lod_scales[lod]),
                        fragments=fragments,
                    )
                )

        return Meshfile(
            chunk_shape=Vec3Df(*chunk_shape),
            grid_origin=Vec3Df(*grid_origin),
            lods=[
                replace(
                    lod,
                    fragments=[
                        replace(
                            frag,
                            byte_offset=int(
                                byte_offset - frag_byte_offset + frag.byte_offset
                            ),
                        )
                        for frag in lod.fragments
                    ],
                )
                for lod in lods
            ],
        )


async def __read_ranged_data(
    http_client: ClientSession, url: str, range: Tuple[int, int]
) -> bytes:
    if range[0] == range[1]:
        return b""
    headers = {"Range": f"bytes={int(range[0])}-{int(range[1]-1)}"}
    async with await http_client.get(url, headers=headers) as r:
        r.raise_for_status()
        response_buffer = await r.read()
    return response_buffer


@alru_cache(maxsize=2 ** 12, cache_exceptions=False)
async def __read_shard_index(
    http_client: ClientSession, shard_url: str, sharding_info: ShardingInfo
) -> np.ndarray:
    shard_index_range = sharding_info.get_shard_index_range()
    print("shard_index_range", shard_index_range)
    index = sharding_info.parse_shard_index(
        await __read_ranged_data(http_client, shard_url, shard_index_range)
    )
    return index


@alru_cache(maxsize=2 ** 12, cache_exceptions=False)
async def __read_minishard_index(
    http_client: ClientSession,
    shard_url: str,
    sharding_info: ShardingInfo,
    minishard_info: MinishardInfo,
) -> np.ndarray:
    shard_index = await __read_shard_index(http_client, shard_url, sharding_info)
    minishard_index_range = sharding_info.get_minishard_index_range(
        minishard_info.minishard_number, shard_index
    )
    print("minishare_index_range", minishard_index_range)
    buf = await __read_ranged_data(http_client, shard_url, minishard_index_range)
    index = sharding_info.parse_minishard_index(buf)
    return index


async def __read_sharded_mesh(
    http_client: ClientSession, source_url: str, mesh_info: MeshInfo, segment_id: int
) -> Optional[Meshfile]:
    sharding_info = mesh_info.sharding
    chunk_id = np.uint64(segment_id)
    minishard_info = sharding_info.get_minishard_info(chunk_id)
    shard_url = f"{source_url}%2F{sharding_info.format_shard_for_url(minishard_info)}.shard?alt=media"
    print("shard_url", shard_url)
    minishard_index = await __read_minishard_index(
        http_client, shard_url, sharding_info, minishard_info
    )
    chunk_range = sharding_info.get_chunk_range(chunk_id, minishard_index)
    print("chunk_range", chunk_range)
    if chunk_range is None:
        return None
    buf = sharding_info.parse_chunk(
        await __read_ranged_data(http_client, shard_url, chunk_range)
    )

    return mesh_info.parse_meshfile(buf, chunk_range[0])


async def main():
    async with ClientSession(raise_for_status=True) as http_client:
        # segment_id = 673639143
        segment_id = 977667682

        info = MeshInfo(
            sharding=ShardingInfo(
                dataset_size=Vec3D(0, 0, 0),
                chunk_size=Vec3D(0, 0, 0),
                preshift_bits=6,
                minishard_bits=8,
                minishard_index_encoding="gzip",
                shard_bits=10,
                data_encoding="gzip",
                hashfn="murmurhash3_x86_128",
            ),
            lod_scale_multiplier=1,
            transform=np.array([[16, 0, 0, 0], [0, 16, 0, 0], [0, 0, 16, 0]]),
            vertex_quantization_bits=16,
        )

        mesh = await __read_sharded_mesh(
            http_client,
            "https://www.googleapis.com/storage/v1/b/neuroglancer-janelia-flyem-hemibrain/o/v1.0%2Fsegmentation%2Fmesh",
            info,
            segment_id,
        )

        print(mesh)
        for i, lod in enumerate(mesh.lods):
            print(
                "Mean frag byte_size",
                1,
                np.mean([frag.byte_size for frag in lod.fragments]),
            )
            for j, frag in enumerate(lod.fragments):
                print(
                    i,
                    j,
                    frag.position,
                    frag.position + lod.chunk_shape,
                    frag.byte_offset,
                    frag.byte_offset + frag.byte_size,
                )


asyncio.run(main())

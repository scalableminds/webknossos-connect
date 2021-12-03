import gzip
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

from ...utils.types import Vec3D
from .mmh3 import hash64


def compressed_morton_code(pos: Vec3D, grid_size: Vec3D) -> np.uint64:
    """
    Computes the compressed morton code as per
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#compressed-morton-code
    https://github.com/google/neuroglancer/blob/162b698f703c86e0b3e92b8d8e0cacb0d3b098df/src/neuroglancer/util/zorder.ts#L72
    """
    bits = Vec3D(
        math.ceil(math.log2(grid_size.x)),
        math.ceil(math.log2(grid_size.y)),
        math.ceil(math.log2(grid_size.z)),
    )

    max_bits = bits.as_np().max()
    output_bit = np.uint64(0)
    one = np.uint64(1)

    output = np.uint64(0)
    for bit in range(max_bits):
        if bit < bits.x:
            output |= ((np.uint64(pos.x) >> np.uint64(bit)) & one) << output_bit
            output_bit += one
        if bit < bits.y:
            output |= ((np.uint64(pos.y) >> np.uint64(bit)) & one) << output_bit
            output_bit += one
        if bit < bits.z:
            output |= ((np.uint64(pos.z) >> np.uint64(bit)) & one) << output_bit
            output_bit += one

    return output


def identity(a: np.uint64) -> np.uint64:
    return a


def compute_minishard_mask(minishard_bits: int) -> np.uint64:
    assert minishard_bits >= 0, "minishard_bits must be ≥ 0"
    if minishard_bits == 0:
        return np.uint64(0)

    minishard_mask = np.uint64(1)
    for i in range(minishard_bits - 1):
        minishard_mask <<= np.uint64(1)
        minishard_mask |= np.uint64(1)
    return np.uint64(minishard_mask)


def compute_shard_mask(shard_bits: int, minishard_bits: int) -> np.uint64:
    one_mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    cursor = np.uint64(minishard_bits + shard_bits)
    shard_mask = ~((one_mask >> cursor) << cursor)
    minishard_mask = compute_minishard_mask(minishard_bits)
    return shard_mask & (~minishard_mask)  # pylint: disable=invalid-unary-operand-type


@dataclass(frozen=True)
class MinishardInfo:
    shard_number: np.uint64
    minishard_number: np.uint64


@dataclass(frozen=True)
class ShardingInfo:
    """
    Stores all necessary information for implementing the shared precomputed
    format of Neuroglancer. The actual downloading of the indices and chunk
    data needs to happen in the backend, because of the availability of the
    http client and controllable caching.
    """

    dataset_size: Vec3D
    chunk_size: Vec3D
    preshift_bits: int
    minishard_bits: int
    shard_bits: int
    hashfn: str
    minishard_index_encoding: str = "raw"
    data_encoding: str = "raw"
    _shard_mask: np.uint64 = field(init=False)
    _minishard_mask: np.uint64 = field(init=False)

    def __post_init__(self) -> None:
        # object.__setattr__ must be used to circumvent errors since the dataclass is frozen
        object.__setattr__(
            self,
            "_shard_mask",
            compute_shard_mask(self.shard_bits, self.minishard_bits),
        )
        object.__setattr__(
            self, "_minishard_mask", compute_minishard_mask(self.minishard_bits)
        )

    @staticmethod
    def parse(
        info_json: Any,
        dataset_size: Optional[Vec3D] = None,
        chunk_size: Optional[Vec3D] = None,
    ) -> "ShardingInfo":
        assert (
            info_json["@type"] == "neuroglancer_uint64_sharded_v1"
        ), "Only `neuroglancer_uint64_sharded_v1` sharding type is supported."

        return ShardingInfo(
            dataset_size=dataset_size if dataset_size is not None else Vec3D.zeros(),
            chunk_size=chunk_size if chunk_size is not None else Vec3D.zeros(),
            preshift_bits=int(info_json["preshift_bits"]),
            shard_bits=int(info_json["shard_bits"]),
            minishard_bits=int(info_json["minishard_bits"]),
            hashfn=info_json["hash"],
            minishard_index_encoding=info_json.get("minishard_index_encoding"),
            data_encoding=info_json.get("data_encoding"),
        )

    def get_chunk_key(self, pos: Vec3D) -> np.uint64:
        chunk_addr = pos // self.chunk_size
        grid_size = Vec3D(
            *np.ceil(np.array(self.dataset_size) / np.array(self.chunk_size))
        )
        key = compressed_morton_code(chunk_addr, grid_size)
        return key

    def format_shard_for_url(self, loc: MinishardInfo) -> str:
        return format(loc.shard_number, "x").zfill(int(np.ceil(self.shard_bits / 4.0)))

    def hash_chunk_id(self, chunk_id: np.uint64) -> np.uint64:
        assert self.hashfn in (
            "murmurhash3_x86_128",
            "identity",
        ), "Only `identity` or `murmurhash3_x86_128` hash functions are supported."
        if self.hashfn == "murmurhash3_x86_128":
            return np.uint64(hash64(chunk_id.tobytes())[0])
        else:
            return chunk_id

    def get_minishard_info(self, key: np.uint64) -> MinishardInfo:
        chunk_id = np.uint64(key) >> np.uint64(self.preshift_bits)
        chunk_id = self.hash_chunk_id(chunk_id)
        minishard_number = np.uint64(chunk_id & self._minishard_mask)
        shard_number = np.uint64(
            (chunk_id & self._shard_mask) >> np.uint64(self.minishard_bits)
        )
        return MinishardInfo(shard_number, minishard_number)

    def get_shard_index_range(self) -> Tuple[int, int]:
        shard_index_end = (np.uint64(1) << np.uint64(self.minishard_bits)) * 16
        return (0, shard_index_end)

    def parse_shard_index(self, buf: bytes) -> np.ndarray:
        # See: https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md#shard-index-format
        # Format: [[minishard_index_start, minishard_index_end], ...]
        # Offsets are relative to shard_index_end
        return (
            np.frombuffer(buf, dtype="<u8")
            .reshape((np.uint64(1) << np.uint64(self.minishard_bits), 2))
            .copy()
        )

    def get_minishard_index_range(
        self, minishard_number: np.uint64, shard_index: np.ndarray
    ) -> Tuple[int, int]:
        _, shard_index_end = self.get_shard_index_range()
        minishard_index_start = shard_index_end + shard_index[minishard_number, 0]
        minishard_index_end = shard_index_end + shard_index[minishard_number, 1]
        return (minishard_index_start, minishard_index_end)

    def parse_minishard_index(self, buf: bytes) -> np.ndarray:
        if self.minishard_index_encoding == "gzip":
            buf = gzip.decompress(buf)
        # See: https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/sharded.md#minishard-index-format
        # Format: [[chunk_id, chunk_offset, chunk_byte_count], ...]
        # Offsets are relative to shard_index_end
        index = np.frombuffer(buf, dtype="<u8").reshape((3, -1)).copy()
        if index.shape[1] == 0:
            return index.T
        index[0, :] = np.cumsum(index[0])
        index[1, 1:] = index[1, 0] + np.cumsum(index[1, 1:]) + np.cumsum(index[2, 0:-1])
        return index.T

    def get_chunk_range(
        self, chunk_id: np.uint64, minishard_index: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        _, shard_index_end = self.get_shard_index_range()
        chunk_entries = minishard_index[minishard_index[:, 0] == chunk_id]
        if chunk_entries.shape[0] == 0:
            return None
        chunk_entry = chunk_entries[0]
        return (
            int(shard_index_end + chunk_entry[1]),
            int(shard_index_end + chunk_entry[2] + chunk_entry[1]),
        )

    def parse_chunk(self, buf: bytes) -> bytes:
        if self.data_encoding == "gzip":
            return gzip.decompress(buf)
        return buf

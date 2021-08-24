import gzip
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, cast

import numpy as np

from ...utils.types import Vec3D


def compressed_morton_code(pos: Vec3D, grid_size: Vec3D) -> np.uint64:
    """
    Computes the compressed morton code as per
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#compressed-morton-code
    """
    pos_np = pos.as_np().astype(np.uint32)
    grid_size_np = grid_size.as_np()

    num_bits = max((math.ceil(math.log2(s)) for s in grid_size_np))
    j = np.uint64(0)
    one = np.uint64(1)

    output = np.uint64(0)
    for i in range(num_bits):
        for dim in range(3):
            if 2 ** i <= grid_size_np[dim]:
                bit = ((np.uint64(pos_np[dim]) >> np.uint64(i)) & one) << j
                output |= bit
                j += one

    return output


# flake8: noqa: C901
def mmh3_128(key: bytes, seed: int = 0x0) -> int:
    """
    Implements 128bit murmur3 hash for x86.
    pymmh3 was written by Fredrik Kihlander and enhanced by Swapnil Gusani.
    Public domain code. https://code.google.com/p/smhasher/wiki/MurmurHash3
    """

    def fmix(h: int) -> int:
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        return h

    length = len(key)
    nblocks = int(length / 16)

    h1 = seed
    h2 = seed
    h3 = seed
    h4 = seed

    c1 = 0x239B961B
    c2 = 0xAB0E9789
    c3 = 0x38B34AE5
    c4 = 0xA1E38B93

    # body
    for block_start in range(0, nblocks * 16, 16):
        k1 = (
            key[block_start + 3] << 24
            | key[block_start + 2] << 16
            | key[block_start + 1] << 8
            | key[block_start + 0]
        )

        k2 = (
            key[block_start + 7] << 24
            | key[block_start + 6] << 16
            | key[block_start + 5] << 8
            | key[block_start + 4]
        )

        k3 = (
            key[block_start + 11] << 24
            | key[block_start + 10] << 16
            | key[block_start + 9] << 8
            | key[block_start + 8]
        )

        k4 = (
            key[block_start + 15] << 24
            | key[block_start + 14] << 16
            | key[block_start + 13] << 8
            | key[block_start + 12]
        )

        k1 = (c1 * k1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        k1 = (c2 * k1) & 0xFFFFFFFF
        h1 ^= k1

        h1 = (h1 << 19 | h1 >> 13) & 0xFFFFFFFF  # inlined ROTL32
        h1 = (h1 + h2) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0x561CCD1B) & 0xFFFFFFFF

        k2 = (c2 * k2) & 0xFFFFFFFF
        k2 = (k2 << 16 | k2 >> 16) & 0xFFFFFFFF  # inlined ROTL32
        k2 = (c3 * k2) & 0xFFFFFFFF
        h2 ^= k2

        h2 = (h2 << 17 | h2 >> 15) & 0xFFFFFFFF  # inlined ROTL32
        h2 = (h2 + h3) & 0xFFFFFFFF
        h2 = (h2 * 5 + 0x0BCAA747) & 0xFFFFFFFF

        k3 = (c3 * k3) & 0xFFFFFFFF
        k3 = (k3 << 17 | k3 >> 15) & 0xFFFFFFFF  # inlined ROTL32
        k3 = (c4 * k3) & 0xFFFFFFFF
        h3 ^= k3

        h3 = (h3 << 15 | h3 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        h3 = (h3 + h4) & 0xFFFFFFFF
        h3 = (h3 * 5 + 0x96CD1C35) & 0xFFFFFFFF

        k4 = (c4 * k4) & 0xFFFFFFFF
        k4 = (k4 << 18 | k4 >> 14) & 0xFFFFFFFF  # inlined ROTL32
        k4 = (c1 * k4) & 0xFFFFFFFF
        h4 ^= k4

        h4 = (h4 << 13 | h4 >> 19) & 0xFFFFFFFF  # inlined ROTL32
        h4 = (h1 + h4) & 0xFFFFFFFF
        h4 = (h4 * 5 + 0x32AC3B17) & 0xFFFFFFFF

    # tail
    tail_index = nblocks * 16
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0
    tail_size = length & 15

    if tail_size >= 15:
        k4 ^= key[tail_index + 14] << 16
    if tail_size >= 14:
        k4 ^= key[tail_index + 13] << 8
    if tail_size >= 13:
        k4 ^= key[tail_index + 12]

    if tail_size > 12:
        k4 = (k4 * c4) & 0xFFFFFFFF
        k4 = (k4 << 18 | k4 >> 14) & 0xFFFFFFFF  # inlined ROTL32
        k4 = (k4 * c1) & 0xFFFFFFFF
        h4 ^= k4

    if tail_size >= 12:
        k3 ^= key[tail_index + 11] << 24
    if tail_size >= 11:
        k3 ^= key[tail_index + 10] << 16
    if tail_size >= 10:
        k3 ^= key[tail_index + 9] << 8
    if tail_size >= 9:
        k3 ^= key[tail_index + 8]

    if tail_size > 8:
        k3 = (k3 * c3) & 0xFFFFFFFF
        k3 = (k3 << 17 | k3 >> 15) & 0xFFFFFFFF  # inlined ROTL32
        k3 = (k3 * c4) & 0xFFFFFFFF
        h3 ^= k3

    if tail_size >= 8:
        k2 ^= key[tail_index + 7] << 24
    if tail_size >= 7:
        k2 ^= key[tail_index + 6] << 16
    if tail_size >= 6:
        k2 ^= key[tail_index + 5] << 8
    if tail_size >= 5:
        k2 ^= key[tail_index + 4]

    if tail_size > 4:
        k2 = (k2 * c2) & 0xFFFFFFFF
        k2 = (k2 << 16 | k2 >> 16) & 0xFFFFFFFF  # inlined ROTL32
        k2 = (k2 * c3) & 0xFFFFFFFF
        h2 ^= k2

    if tail_size >= 4:
        k1 ^= key[tail_index + 3] << 24
    if tail_size >= 3:
        k1 ^= key[tail_index + 2] << 16
    if tail_size >= 2:
        k1 ^= key[tail_index + 1] << 8
    if tail_size >= 1:
        k1 ^= key[tail_index + 0]

    if tail_size > 0:
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # finalization
    h1 ^= length
    h2 ^= length
    h3 ^= length
    h4 ^= length

    h1 = (h1 + h2) & 0xFFFFFFFF
    h1 = (h1 + h3) & 0xFFFFFFFF
    h1 = (h1 + h4) & 0xFFFFFFFF
    h2 = (h1 + h2) & 0xFFFFFFFF
    h3 = (h1 + h3) & 0xFFFFFFFF
    h4 = (h1 + h4) & 0xFFFFFFFF

    h1 = fmix(h1)
    h2 = fmix(h2)
    h3 = fmix(h3)
    h4 = fmix(h4)

    h1 = (h1 + h2) & 0xFFFFFFFF
    h1 = (h1 + h3) & 0xFFFFFFFF
    h1 = (h1 + h4) & 0xFFFFFFFF
    h2 = (h1 + h2) & 0xFFFFFFFF
    h3 = (h1 + h3) & 0xFFFFFFFF
    h4 = (h1 + h4) & 0xFFFFFFFF

    return h4 << 96 | h3 << 64 | h2 << 32 | h1


def mmh3_64(key: np.uint64, seed: int = 0x0) -> np.uint64:
    """ Implements 64bit murmur3 hash. """

    hash_128 = mmh3_128(key.tobytes(), seed)

    unsigned_val1 = hash_128 & 0xFFFFFFFFFFFFFFFF
    if unsigned_val1 & 0x8000000000000000 == 0:
        signed_val1 = unsigned_val1
    else:
        signed_val1 = -((unsigned_val1 ^ 0xFFFFFFFFFFFFFFFF) + 1)

    # unsigned_val2 = (hash_128 >> 64) & 0xFFFFFFFFFFFFFFFF
    # if unsigned_val2 & 0x8000000000000000 == 0:
    #     signed_val2 = unsigned_val2
    # else:
    #     signed_val2 = -((unsigned_val2 ^ 0xFFFFFFFFFFFFFFFF) + 1)

    return np.uint64(signed_val1)


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
        object.__setattr__(
            self,
            "_shard_mask",
            compute_shard_mask(self.shard_bits, self.minishard_bits),
        )
        object.__setattr__(
            self, "_minishard_mask", compute_minishard_mask(self.minishard_bits)
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

    def get_minishard_info(self, key: np.uint64) -> MinishardInfo:
        chunk_id = np.uint64(key) >> np.uint64(self.preshift_bits)
        hashfn = cast(
            Callable[[np.uint64], np.uint64],
            mmh3_64 if self.hashfn == "murmurhash3_x86_128" else identity,
        )
        chunk_id = hashfn(chunk_id)
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

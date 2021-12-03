from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any, List, Tuple

import DracoPy
import numpy as np
from stl import Mesh

from ...utils.types import Vec3Df
from .sharding import ShardingInfo


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

    def fragment_offset_and_shape(
        self, lod: int, fragment: int
    ) -> Tuple[Vec3Df, Vec3Df, Vec3Df]:
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

    def decode_data(self, fragment: MeshfileFragment, buf: bytes) -> bytes:
        mesh_obj = DracoPy.decode_buffer_to_mesh(buf)
        vertices = np.array(mesh_obj.points, dtype="f4").reshape((-1, 3))
        faces = np.array(mesh_obj.faces, dtype="<u4").reshape((-1, 3))
        stl_mesh = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = fragment.position.as_np() + vertices[f[j], :]
        out_buf = BytesIO()
        stl_mesh.save("mesh.stl", out_buf, update_normals=True)
        return out_buf.getvalue()


@dataclass(frozen=True)
class MeshInfo:
    """
    Stores the representation of the meshes of a segmentation layer.
    The format is defined at https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md.
    """

    sharding: ShardingInfo
    _transform: Tuple[float, ...]
    vertex_quantization_bits: int
    lod_scale_multiplier: int = 1

    @staticmethod
    def parse(info_json: Any) -> "MeshInfo":
        assert info_json["@type"] == "neuroglancer_multilod_draco"
        return MeshInfo(
            sharding=ShardingInfo.parse(info_json["sharding"]),
            _transform=tuple(info_json["transform"]),
            vertex_quantization_bits=info_json["vertex_quantization_bits"],
            lod_scale_multiplier=info_json["lod_scale_multiplier"],
        )

    @property
    def transform(self) -> np.ndarray:
        return np.array(self._transform, dtype="f4").reshape((3, 4))

    def parse_meshfile(self, buf: bytes, byte_offset: int) -> Meshfile:
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

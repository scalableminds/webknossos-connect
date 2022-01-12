import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, cast

import h5py
import numpy as np
from aiohttp import ClientSession
from wkcuber.api.Dataset import WKDataset

from ...fast_wkw import DatasetCache  # pylint: disable=no-name-in-module
from ...utils.blocking import run_blocking
from ...utils.types import JSON, Vec3D
from ..backend import Backend, DatasetInfo, MeshfileInfo
from .models import Dataset

logger = logging.getLogger()


class Wkw(Backend):
    wkw_cache: DatasetCache

    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        super().__init__(config, http_client)
        cache_size = config.get("fileCacheMaxEntries", 1000)
        self.wkw_cache = DatasetCache(cache_size)

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:

        path = Wkw.path(dataset_info, organization_name, dataset_name)
        return Dataset(
            organization_name, dataset_name, WKDataset(str(path)), self.wkw_cache
        )

    async def read_data(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        dataset = cast(Dataset, abstract_dataset)
        return await dataset.read_data(layer_name, zoom_step, wk_offset, shape)

    async def get_meshes(
        self, abstract_dataset: DatasetInfo, layer_name: str
    ) -> Optional[List[MeshfileInfo]]:
        dataset = cast(Dataset, abstract_dataset)
        assert (
            layer_name in dataset.dataset_handle.layers
        ), f"Layer {layer_name} does not exist"
        meshes_folder = dataset.dataset_handle.path / layer_name / "meshes"

        def read_mesh_mapping_name(mesh_path: Path) -> Optional[str]:
            with h5py.File(mesh_path, "r") as mesh_file:
                return mesh_file.attrs.get("metadata/mapping_name", None)

        if meshes_folder.exists() and meshes_folder.is_dir:
            output = []
            for mesh_path in meshes_folder.glob("*.hdf5"):
                output.append(
                    MeshfileInfo(
                        mesh_file_name=mesh_path.name[:-5],
                        mapping_name=await run_blocking(
                            read_mesh_mapping_name, mesh_path
                        ),
                    )
                )
            return output
        return []

    async def get_chunks_for_mesh(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        mesh_name: str,
        segment_id: int,
    ) -> Optional[List[Vec3D]]:
        dataset = cast(Dataset, abstract_dataset)
        assert (
            layer_name in dataset.dataset_handle.layers
        ), f"Layer {layer_name} does not exist"
        meshes_folder = dataset.dataset_handle.path / layer_name / "meshes"
        assert (
            meshes_folder.exists() and meshes_folder.is_dir()
        ), f"Mesh folder for layer {layer_name} does not exist"
        mesh_path = meshes_folder / f"{mesh_name}.hdf5"
        assert (
            meshes_folder.exists()
        ), f"Mesh {mesh_name} for layer {layer_name} does not exist"

        def read_mesh(mesh_path: Path, segment_id: int) -> List[Vec3D]:
            with h5py.File(mesh_path, "r") as mesh_file:
                segment_group = mesh_file.get(str(segment_id), None)
                if segment_group is None:
                    return []
                lod_group = segment_group["0"]
                chunk_keys = [
                    Vec3D(*[int(a) for a in chunk_key.split("_")])
                    for chunk_key in lod_group.keys()
                ]
                return chunk_keys

        return await run_blocking(read_mesh, mesh_path, segment_id)

    async def get_chunk_data_for_mesh(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        mesh_name: str,
        segment_id: int,
        position: Vec3D,
    ) -> Optional[bytes]:
        dataset = cast(Dataset, abstract_dataset)
        assert (
            layer_name in dataset.dataset_handle.layers
        ), f"Layer {layer_name} does not exist"
        meshes_folder = dataset.dataset_handle.path / layer_name / "meshes"
        assert (
            meshes_folder.exists() and meshes_folder.is_dir()
        ), f"Mesh folder for layer {layer_name} does not exist"
        mesh_path = meshes_folder / f"{mesh_name}.hdf5"
        assert (
            meshes_folder.exists()
        ), f"Mesh {mesh_name} for layer {layer_name} does not exist"

        def read_chunk(mesh_path: Path, segment_id: int, position: Vec3D) -> bytes:
            with h5py.File(mesh_path, "r") as mesh_file:
                chunk_key = f"{position[0]}_{position[1]}_{position[2]}"
                mesh_encoding = mesh_file.attrs.get("metadata/encoding", "stl")
                mesh_binary = mesh_file[f"/{segment_id}/0/{chunk_key}"][:].tobytes()
                if mesh_encoding == "stl+gzip":
                    mesh_binary = gzip.decompress(mesh_binary)
                return mesh_binary

        return await run_blocking(read_chunk, mesh_path, segment_id, position)

    def clear_dataset_cache(self, abstract_dataset: DatasetInfo) -> None:
        dataset = cast(Dataset, abstract_dataset)
        dataset.clear_cache()

    @staticmethod
    def path(dataset_info: JSON, organization_name: str, dataset_name: str) -> Path:
        if "path" in dataset_info:
            return Path(dataset_info["path"])
        else:
            return Path("data", "binary", organization_name, dataset_name)

import logging
from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np
from aiohttp import ClientSession
from wkcuber.api.Dataset import WKDataset

from ...fast_wkw import DatasetCache  # pylint: disable=no-name-in-module
from ...utils.types import JSON, Vec3D
from ..backend import Backend, DatasetInfo
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
    ) -> Optional[List[str]]:
        dataset = cast(Dataset, abstract_dataset)
        assert (
            layer_name in dataset.dataset_handle.layers
        ), f"Layer {layer_name} does not exist"
        meshes_folder = dataset.dataset_handle.path / layer_name / "meshes"
        if meshes_folder.exists() and meshes_folder.is_dir:
            mesh_paths = list(meshes_folder.glob("*.hdf5"))
            return [mesh_path.name[-6] for mesh_path in mesh_paths]
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

        raise NotImplementedError()

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

        raise NotImplementedError()

    def clear_dataset_cache(self, abstract_dataset: DatasetInfo) -> None:
        dataset = cast(Dataset, abstract_dataset)
        dataset.clear_cache()

    @staticmethod
    def path(dataset_info: JSON, organization_name: str, dataset_name: str) -> Path:
        if "path" in dataset_info:
            return Path(dataset_info["path"])
        else:
            return Path("data", "binary", organization_name, dataset_name)

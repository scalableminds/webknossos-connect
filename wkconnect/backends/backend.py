from abc import ABCMeta, abstractmethod
from typing import Dict, List, NamedTuple, Optional

import numpy as np
from aiohttp import ClientSession

from ..utils.types import JSON, Vec3D
from ..webknossos.models import DataSource as WKDataSource


class DatasetInfo(metaclass=ABCMeta):
    organization_name: str
    dataset_name: str

    @abstractmethod
    def to_webknossos(self) -> WKDataSource:
        pass


class Backend(metaclass=ABCMeta):
    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()

    @abstractmethod
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        pass

    @abstractmethod
    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        pass

    @abstractmethod
    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        """
        Read voxels from the backend.

        :param dataset:
        :param layer_name:
        :param zoomStep: 2^zoomStep is the smallest dimension of the scale
        :param wk_offset: in wk voxels
        :param shape: in scale voxels
        :returns: numpy array of shape shape
        """

    async def get_meshes(
        self, dataset: DatasetInfo, layer_name: str
    ) -> Optional[List["MeshfileInfo"]]:
        """
        List the available meshfiles of a layer from the backend.

        :param dataset:
        :param layer_name:
        :returns: list of meshfile names
        """
        raise NotImplementedError()

    async def get_chunks_for_mesh(
        self, dataset: DatasetInfo, layer_name: str, mesh_name: str, segment_id: int
    ) -> Optional[List[Vec3D]]:
        """
        List the available chunks of a mesh from the backend.

        :param dataset:
        :param layer_name:
        :param mesh_name:
        :param segment_id:
        :returns: list of tuples with the top-left position of each chunk
        """
        raise NotImplementedError()

    async def get_chunk_data_for_mesh(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        mesh_name: str,
        segment_id: int,
        position: Vec3D,
    ) -> Optional[bytes]:
        """
        Read a chunk of a mesh from the backend.

        :param dataset:
        :param layer_name:
        :param mesh_name:
        :param segment_id:
        :param position:
        :returns: bytes of a mesh chunk in binary STL format
        """
        raise NotImplementedError()

    @abstractmethod
    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass


class MeshfileInfo(NamedTuple):
    mesh_file_name: str
    mapping_name: Optional[str] = None

    def as_json(self) -> Dict[str, Optional[str]]:
        return {"meshFileName": self.mesh_file_name, "mappingName": self.mapping_name}

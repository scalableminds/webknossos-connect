from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

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

    @abstractmethod
    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        pass

    async def on_shutdown(self) -> None:
        pass

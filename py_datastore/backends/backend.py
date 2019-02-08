from typing import Dict

import numpy as np
from aiohttp import ClientSession

from ..utils.types import JSON, Vec3D
from ..webknossos.models import DataSource as WKDataSource


class DatasetInfo:
    organization_name: str
    dataset_name: str

    def to_webknossos(self) -> WKDataSource:
        pass


class Backend:
    @classmethod
    def name(cls) -> str:
        return "Subclass responsibility"

    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        pass

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        pass

    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        offset: Vec3D,
        shape: Vec3D,
    ) -> np.ndarray:
        """
        Read voxels from the backend.

        :param dataset:
        :param layer_name:
        :param zoomStep: 2^zoomStep is the smallest dimension of the scale
        :param offset: in wk voxels
        :param shape: in scale voxels
        :returns: numpy array of shape shape
        """
        pass

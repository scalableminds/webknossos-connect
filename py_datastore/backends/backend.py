import numpy as np

from typing import Any, Dict, Tuple

from ..utils.http import HttpClient
from ..utils.types import Vec3D
from ..webknossos.models import DataSource as WKDataSource


class DatasetInfo:
    organization_name: str
    dataset_name: str

    def to_webknossos(self) -> WKDataSource:
        pass


class Backend:
    @classmethod
    def name(cls) -> str:
        return "TODO"

    def __init__(self, config: Dict, http_client: HttpClient) -> None:
        pass

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: Any
    ) -> DatasetInfo:
        pass

    async def read_data(
        self,
        dataset: DatasetInfo,
        layer_name: str,
        resolution: Vec3D,
        offset: Vec3D,
        shape: Vec3D,
    ) -> np.ndarray:
        pass

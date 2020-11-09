import logging
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
from aiohttp import ClientSession
from wkcuber.api.Dataset import WKDataset

from ...utils.types import JSON, Vec3D
from ..backend import Backend, DatasetInfo
from .models import Dataset

logger = logging.getLogger()


class Wkw(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        super().__init__(config, http_client)

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:

        path = Wkw.path(dataset_info, organization_name, dataset_name)
        return Dataset(organization_name, dataset_name, WKDataset(str(path)))

    async def read_data(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        dataset = cast(Dataset, abstract_dataset)
        return dataset.read_data(layer_name, zoom_step, wk_offset, shape)

    def clear_dataset_cache(self, abstract_dataset: DatasetInfo) -> None:
        dataset = cast(Dataset, abstract_dataset)
        dataset.clear_cache()

    @staticmethod
    def path(dataset_info: JSON, organization_name: str, dataset_name: str) -> Path:
        if "path" in dataset_info:
            return Path(dataset_info["path"])
        else:
            return Path("data", "binary", organization_name, dataset_name)

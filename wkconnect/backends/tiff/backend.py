import logging
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
from aiohttp import ClientSession

from ...utils.types import JSON, Vec3D, Vec3Df
from ..backend import Backend, DatasetInfo
from .models import Dataset

logger = logging.getLogger()


class Tiff(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        super().__init__(config, http_client)
        self.config = config

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        scale = (
            Vec3Df(dataset_info["scale"][0], dataset_info["scale"][1], 1)
            if "scale" in dataset_info
            else Vec3Df(1, 1, 1)
        )
        path = Tiff.path(dataset_info, organization_name, dataset_name)
        untiled_size_maximum_mp = (
            int(self.config["tif_untiled_size_maximum_mp"])
            if "tif_untiled_size_maximum_mp" in self.config
            else 20
        )
        return Dataset(
            organization_name, dataset_name, scale, path, untiled_size_maximum_mp
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

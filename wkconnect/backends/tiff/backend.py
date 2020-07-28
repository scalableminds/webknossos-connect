import logging
from typing import Dict, Optional

import numpy as np
from aiohttp import ClientSession
from wkcuber import TiffDataset

from ..backend import Backend, DatasetInfo
from ...utils.types import JSON, Box3D, HashableDict, Vec3D, Vec3Df

logger = logging.getLogger()


class Tiff(Backend):

    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        super().__init__(config, http_client)

    async def handle_new_dataset(
            self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        pass

    async def read_data(self, dataset: DatasetInfo, layer_name: str, zoom_step: int, wk_offset: Vec3D, shape: Vec3D) -> \
            Optional[np.ndarray]:
        pass

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        pass
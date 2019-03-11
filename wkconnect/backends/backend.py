from abc import ABCMeta, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from aiohttp import ClientSession

from ..utils.types import JSON, Box3D, Vec3D
from ..webknossos.models import DataSource as WKDataSource

Chunk = Tuple[Box3D, np.ndarray]


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

    def _chunks(self, box: Box3D, frame: Box3D, chunk_size: Vec3D) -> Iterable[Box3D]:
        # clip data outside available data
        inside = box.intersect(frame)

        # align request to chunks
        aligned = (inside - frame.left).div(chunk_size) * chunk_size + frame.left

        for chunk_offset in aligned.range(offset=chunk_size):
            # The size is at most chunk_size but capped to fit the dataset:
            capped_chunk_size = chunk_size.pairmin(frame.size() - chunk_offset)
            yield Box3D.from_size(chunk_offset, capped_chunk_size)

    def _cutout(self, chunks: List[Chunk], box: Box3D) -> np.ndarray:
        result = np.zeros(box.size(), dtype=chunks[0][1].dtype, order="F")
        for chunk_box, chunk_data in chunks:
            inner = chunk_box.intersect(box)
            result[(inner - box.left).np_slice()] = chunk_data[
                (inner - chunk_box.left).np_slice()
            ]
        return result

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

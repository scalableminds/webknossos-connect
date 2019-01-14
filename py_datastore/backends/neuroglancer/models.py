from functools import reduce
from typing import Any, Dict, List, Tuple

from ..backend import DatasetInfo
from ...utils.types import Vec3D
from ...webknossos.models import (
    BoundingBox,
    DataSource as WKDataSource,
    DataSourceId as WKDataSourceId,
    DataLayer as WKDataLayer,
)


class Scale:
    supported_encodings = ["raw", "jpeg"]

    chunk_sizes: List[Vec3D]
    encoding: str
    key: str
    resolution: Vec3D
    size: Vec3D
    voxel_offset: Vec3D

    def __init__(
        self,
        chunk_sizes: List[Vec3D],
        encoding: str,
        key: str,
        resolution: Vec3D,
        size: Vec3D,
        voxel_offset: Vec3D,
        **kwargs: Any
    ) -> None:
        self.chunk_sizes = chunk_sizes
        self.encoding = encoding
        self.key = key
        self.resolution = resolution
        self.size = size
        self.voxel_offset = voxel_offset

        assert self.encoding in self.supported_encodings

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.voxel_offset, *self.size)


class Layer:
    supported_data_types = ["uint8", "uint16", "uint32", "uint64"]
    supported_types = ["image", "segmentation"]

    source: str
    data_type: str
    num_channels: int
    scales: List[Scale]
    type: str

    def __init__(
        self,
        source: str,
        data_type: str,
        num_channels: int,
        scales: List[Scale],
        type: str,
        **kwargs: Any
    ) -> None:
        self.source = source
        self.data_type = data_type
        self.num_channels = num_channels
        self.scales = scales
        self.type = type

        assert self.data_type in self.supported_data_types
        assert self.num_channels == 1
        assert self.type in self.supported_types

    def to_webknossos(self, layer_name: str) -> WKDataLayer:
        bounding_boxes = map(lambda scale: scale.bounding_box(), self.scales)
        bounding_box = reduce(lambda a, b: a.union(b), bounding_boxes)
        return WKDataLayer(
            layer_name,
            {"image": "color", "segmentation": "segmentation"}[self.type],
            bounding_box,
            list(map(lambda scale: scale.resolution, self.scales)),
            self.data_type,
        )


class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    layers: Dict[str, Layer]

    def __init__(
        self, organization_name: str, dataset_name: str, layers: Dict[str, Layer]
    ) -> None:
        self.organization_name = organization_name
        self.dataset_name = dataset_name
        self.layers = layers

    def to_webknossos(self) -> WKDataSource:
        return WKDataSource(
            WKDataSourceId(self.organization_name, self.dataset_name),
            list(
                [
                    layer.to_webknossos(layer_name)
                    for layer_name, layer in self.layers.items()
                ]
            ),
            (1.0, 1.0, 1.0),
        )

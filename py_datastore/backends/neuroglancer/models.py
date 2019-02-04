from functools import reduce
from operator import mul
from typing import Any, Dict, List, Optional, Tuple, cast

from ..backend import DatasetInfo
from ...utils.types import Vec3D
from ...webknossos.models import (
    BoundingBox,
    DataSource as WKDataSource,
    DataSourceId as WKDataSourceId,
    DataLayer as WKDataLayer,
)


class Scale:
    supported_encodings = ["raw", "jpeg", "compressed_segmentation"]

    chunk_sizes: List[Vec3D]
    encoding: str
    key: str
    resolution: Vec3D
    size: Vec3D
    voxel_offset: Vec3D
    compressed_segmentation_block_size: Optional[Vec3D]

    def __init__(
        self,
        chunk_sizes: List[Vec3D],
        encoding: str,
        key: str,
        resolution: Vec3D,
        size: Vec3D,
        voxel_offset: Vec3D,
        compressed_segmentation_block_size: Optional[Vec3D] = None,
        **kwargs: Any
    ) -> None:
        self.chunk_sizes = chunk_sizes
        self.encoding = encoding
        self.key = key
        self.resolution = resolution
        self.size = size
        self.voxel_offset = voxel_offset
        if encoding == "compressed_segmentation":
            assert compressed_segmentation_block_size is not None
            self.compressed_segmentation_block_size = cast(
                Vec3D, tuple(compressed_segmentation_block_size)
            )
        else:
            self.compressed_segmentation_block_size = None

        assert self.encoding in self.supported_encodings

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.voxel_offset, *self.size)


class Layer:
    supported_data_types = {
        "image": ["uint8", "uint16", "uint32", "uint64"],
        "segmentation": ["uint32", "uint64"],
    }

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

        assert self.type in self.supported_data_types
        assert self.data_type in self.supported_data_types[self.type]
        assert self.num_channels == 1

    def wk_data_type(self) -> str:
        if self.type == "segmentation":
            return "uint32"
        return self.data_type

    def to_webknossos(self, layer_name: str, global_scale: Vec3D) -> WKDataLayer:
        def to_wk(vec: Vec3D) -> Vec3D:
            return cast(
                Vec3D,
                tuple(
                    vec_dim // scale_dim
                    for vec_dim, scale_dim in zip(vec, global_scale)
                ),
            )

        min_scale = min(self.scales, key=lambda scale: reduce(mul, scale.resolution))
        normalized_resolutions = [to_wk(scale.resolution) for scale in self.scales]
        return WKDataLayer(
            layer_name,
            {"image": "color", "segmentation": "segmentation"}[self.type],
            min_scale.bounding_box(),
            normalized_resolutions,
            self.wk_data_type(),
        )


class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    layers: Dict[str, Layer]
    scale: Vec3D

    def __init__(
        self, organization_name: str, dataset_name: str, layers: Dict[str, Layer]
    ) -> None:
        self.organization_name = organization_name
        self.dataset_name = dataset_name
        self.layers = layers
        min_resolution = set(
            min(
                layer.scales, key=lambda scale: reduce(mul, scale.resolution)
            ).resolution
            for layer in self.layers.values()
        )
        assert len(min_resolution) == 1
        self.scale = next(iter(min_resolution))

    def to_webknossos(self) -> WKDataSource:
        return WKDataSource(
            WKDataSourceId(self.organization_name, self.dataset_name),
            list(
                [
                    layer.to_webknossos(layer_name, self.scale)
                    for layer_name, layer in self.layers.items()
                ]
            ),
            self.scale,
        )

from typing import Any, Dict, List, NamedTuple, Optional

from ...utils.types import Box3D, Vec3D
from ...webknossos.models import BoundingBox
from ...webknossos.models import DataLayer as WKDataLayer
from ...webknossos.models import DataSource as WKDataSource
from ...webknossos.models import DataSourceId as WKDataSourceId
from ..backend import DatasetInfo


class Scale(NamedTuple):
    chunk_sizes: List[Vec3D]
    encoding: str
    key: str
    resolution: Vec3D
    size: Vec3D
    voxel_offset: Vec3D
    compressed_segmentation_block_size: Optional[Vec3D] = None

    # assert len(chunk_sizes) > 0
    # assert self.encoding in ["raw", "jpeg", "compressed_segmentation"]
    # if encoding == "compressed_segmentation":
    #     assert compressed_segmentation_block_size is not None

    def box(self) -> Box3D:
        return Box3D.from_size(self.voxel_offset, self.size)

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

    def min_scale(self) -> Scale:
        return min(self.scales, key=lambda scale: sum(scale.resolution))

    def to_webknossos(self, layer_name: str, global_scale: Vec3D) -> WKDataLayer:
        normalized_resolutions = [
            scale.resolution // global_scale for scale in self.scales
        ]
        return WKDataLayer(
            layer_name,
            {"image": "color", "segmentation": "segmentation"}[self.type],
            self.min_scale().bounding_box(),
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
            layer.min_scale().resolution for layer in self.layers.values()
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

from typing import Any, Dict, Optional, Tuple

from dataclasses import InitVar, dataclass, field

from ...utils.types import Box3D, Vec3D
from ...webknossos.models import BoundingBox
from ...webknossos.models import DataLayer as WKDataLayer
from ...webknossos.models import DataSource as WKDataSource
from ...webknossos.models import DataSourceId as WKDataSourceId
from ..backend import DatasetInfo


@dataclass(frozen=True)
class Scale:
    chunk_sizes: Tuple[Vec3D, ...]
    encoding: str
    key: str
    resolution: Vec3D
    size: Vec3D
    voxel_offset: Vec3D
    compressed_segmentation_block_size: Optional[Vec3D] = None

    def __post_init__(self) -> None:
        assert len(self.chunk_sizes) > 0
        assert self.encoding in ["raw", "jpeg", "compressed_segmentation"]
        if self.encoding == "compressed_segmentation":
            assert self.compressed_segmentation_block_size is not None

    def box(self) -> Box3D:
        return Box3D.from_size(self.voxel_offset, self.size)

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.voxel_offset, *self.size)


@dataclass(frozen=True)
class Layer:
    source: str
    data_type: str
    num_channels: int
    scales: Tuple[Scale, ...]
    type: str
    # InitVar allows to consume mesh argument in init without storing it
    mesh: InitVar[Any] = None

    def __post_init__(self, mesh: Any) -> None:
        supported_data_types = {
            "image": ["uint8", "uint16", "uint32", "uint64"],
            "segmentation": ["uint32", "uint64"],
        }
        assert self.type in supported_data_types
        assert self.data_type in supported_data_types[self.type]
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


@dataclass
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    layers: Dict[str, Layer]
    scale: Vec3D = field(init=False)

    def __post_init__(self) -> None:
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

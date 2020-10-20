from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Optional, Tuple

from gcloud.aio.auth import Token

from ...utils.types import Box3D, Vec3D
from ...webknossos.models import BoundingBox
from ...webknossos.models import DataLayer as WkDataLayer
from ...webknossos.models import DataSource as WkDataSource
from ...webknossos.models import DataSourceId as WkDataSourceId
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
    relative_scale: Vec3D
    type: str
    largestSegmentId: Optional[int] = None
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
        assert all(
            max(scale.resolution) == 2 ** i for i, scale in enumerate(self.scales)
        )

    def wk_data_type(self) -> str:
        if self.type == "segmentation":
            return "uint32"
        return self.data_type

    def to_webknossos(self, layer_name: str) -> WkDataLayer:
        return WkDataLayer(
            layer_name,
            {"image": "color", "segmentation": "segmentation"}[self.type],
            self.scales[0].bounding_box(),
            [scale.resolution for scale in self.scales],
            self.wk_data_type(),
            largestSegmentId=self.largestSegmentId,
        )


@dataclass
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    layers: Dict[str, Layer]
    token: Optional[Token] = None
    scale: Vec3D = field(init=False)

    def __post_init__(self) -> None:
        relative_scale = set(layer.relative_scale for layer in self.layers.values())
        assert len(relative_scale) == 1
        self.scale = relative_scale.pop()

    def to_webknossos(self) -> WkDataSource:
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            [
                layer.to_webknossos(layer_name)
                for layer_name, layer in self.layers.items()
            ],
            self.scale.to_float(),
        )

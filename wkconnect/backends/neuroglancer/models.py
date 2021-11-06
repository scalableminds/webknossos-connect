from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from gcloud.aio.auth import Token

from ...utils.types import Box3D, Vec3D, Vec3Df
from ...webknossos.models import BoundingBox
from ...webknossos.models import DataLayer as WkDataLayer
from ...webknossos.models import DataSource as WkDataSource
from ...webknossos.models import DataSourceId as WkDataSourceId
from ..backend import DatasetInfo
from .meshes import MeshInfo
from .sharding import ShardingInfo


@dataclass(frozen=True)
class Scale:
    chunk_size: Vec3D
    encoding: str
    key: str
    mag: Vec3D
    size: Vec3D
    voxel_offset: Vec3D
    sharding: Optional[ShardingInfo]
    compressed_segmentation_block_size: Optional[Vec3D] = None

    def __post_init__(self) -> None:
        assert self.encoding in ["raw", "jpeg", "compressed_segmentation"]
        if self.encoding == "compressed_segmentation":
            assert self.compressed_segmentation_block_size is not None

    def box(self) -> Box3D:
        return Box3D.from_size(self.voxel_offset, self.size)

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.voxel_offset, *self.size)

    def mag1_bounding_box(self) -> BoundingBox:
        return BoundingBox(self.voxel_offset * self.mag, *(self.size * self.mag))


@dataclass(frozen=True)
class Layer:
    source: str
    data_type: str
    num_channels: int
    scales: Tuple[Scale, ...]
    resolution: Vec3Df
    type: str
    mesh: Optional[MeshInfo] = None
    largestSegmentId: Optional[int] = None

    def __post_init__(self) -> None:
        supported_data_types = {
            "image": ["uint8", "uint16", "uint32", "uint64"],
            "segmentation": ["uint32", "uint64"],
        }
        assert self.type in supported_data_types
        assert self.data_type in supported_data_types[self.type]
        assert self.num_channels == 1
        min_mag = sorted([scale.mag for scale in self.scales])[0]
        assert all(scale.mag % min_mag == Vec3D.zeros() for scale in self.scales)

    def wk_data_type(self) -> str:
        return self.data_type

    def to_webknossos(self, layer_name: str) -> WkDataLayer:
        return WkDataLayer(
            layer_name,
            {"image": "color", "segmentation": "segmentation"}[self.type],
            self.scales[0].mag1_bounding_box(),
            [scale.mag for scale in self.scales],
            self.wk_data_type(),
            largestSegmentId=self.largestSegmentId,
        )


@dataclass
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    layers: Dict[str, Layer]
    scale: Vec3Df
    token: Optional[Token] = None

    def to_webknossos(self) -> WkDataSource:
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            [
                layer.to_webknossos(layer_name)
                for layer_name, layer in self.layers.items()
            ],
            self.scale,
        )

    @staticmethod
    def fix_scales(layers: Dict[str, Layer]) -> Tuple[Dict[str, Layer], Vec3Df]:
        """
        This function adapts the scale specification from neuroglancer to the
        webKnossos scheme. In neuroglancer, each `Scale` has an attached
        `resolution` (in nm), whereas webKnossos uses a single `scale` for all
        layers and differentiates through `Mag`s, which are power-of-two
        factors on the dataset's scale. In this function, the dataset's `scale`
        is determined and the `mag`s in the layers are computed and fixed.
        """
        layer_resolution_map = {
            layer_name: layer.resolution for layer_name, layer in layers.items()
        }
        min_resolution = sorted(list(layer_resolution_map.values()))[0]
        new_layers = {}
        for layer_name, layer in layers.items():
            assert layer.resolution % min_resolution == Vec3Df.zeros()
            factor_vec = layer.resolution / min_resolution
            new_scales = tuple(
                replace(scale, mag=(scale.mag.to_float() * factor_vec).to_int())
                for scale in layer.scales
            )
            new_layer = replace(layer, scales=new_scales, resolution=min_resolution)
            new_layers[layer_name] = new_layer
        return new_layers, min_resolution

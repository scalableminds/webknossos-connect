from __future__ import annotations

from typing import List, NamedTuple, Tuple

from ..utils.types import Box3D, Vec3D


class DataStoreStatus(NamedTuple):
    ok: bool
    url: str


class DataRequest(NamedTuple):
    position: Vec3D
    zoomStep: int
    cubeSize: int
    fourBit: bool


class BoundingBox(NamedTuple):
    topLeft: Vec3D
    width: int
    height: int
    depth: int

    def box(self) -> Box3D:
        return Box3D.from_size(self.topLeft, Vec3D(self.width, self.height, self.depth))


class DataSourceId(NamedTuple):
    team: str
    name: str


class DataLayer(NamedTuple):
    name: str
    category: str
    boundingBox: BoundingBox
    resolutions: List[Vec3D]
    elementClass: str

    # assert self.category in ["color", "segmentation"]
    # assert self.elementClass in ["uint8", "uint16", "uint32", "uint64"]


class DataSource(NamedTuple):
    id: DataSourceId
    dataLayers: List[DataLayer]
    scale: Tuple[float, float, float]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, NamedTuple, Optional, cast

import numpy as np

from ..utils.types import JSON, Box3D, Vec3D, Vec3Df


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

    @classmethod
    def from_box(cls, box: Box3D) -> BoundingBox:
        return cls(box.left, *box.size())


class DataSourceId(NamedTuple):
    team: str
    name: str


class Histogram(NamedTuple):
    elementCounts: List[int]
    numberOfElements: int
    min: float
    max: float


@dataclass(unsafe_hash=True)
class DataLayer:
    name: str
    category: str
    boundingBox: BoundingBox
    resolutions: List[Vec3D]
    elementClass: str
    dataFormat: str = field(default="wkw", init=False)
    wkwResolutions: JSON = field(init=False)
    largestSegmentId: Optional[int] = None
    mappings: Optional[List[Any]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        assert self.category in ["color", "segmentation"]
        assert self.elementClass in ["uint8", "uint16", "uint24", "uint32", "uint64"]
        if self.category == "segmentation":
            if self.largestSegmentId is None:
                self.largestSegmentId = cast(int, np.iinfo(self.elementClass).max)
            self.mappings = []
        self.wkwResolutions = [
            {"resolution": i, "cubeLength": 1024} for i in self.resolutions
        ]


class DataSource(NamedTuple):
    id: DataSourceId
    dataLayers: List[DataLayer]
    scale: Vec3Df


class UnusableDataSource(NamedTuple):
    id: DataSourceId
    status: str

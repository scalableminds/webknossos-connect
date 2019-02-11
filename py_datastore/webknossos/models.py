from __future__ import annotations

from typing import Any, List, Tuple

from ..utils.types import Box3D, Vec3D


class DataStoreStatus:
    ok: bool
    url: str

    def __init__(self, ok: bool, url: str, **kwargs: Any) -> None:
        self.ok = ok
        self.url = url


class DataRequest:
    position: Vec3D
    zoomStep: int
    cubeSize: int
    fourBit: bool

    def __init__(
        self,
        position: Vec3D,
        zoomStep: int,
        cubeSize: int,
        fourBit: bool,
        **kwargs: Any
    ) -> None:
        self.position = position
        self.zoomStep = zoomStep
        self.cubeSize = cubeSize
        self.fourBit = fourBit


class BoundingBox:
    topLeft: Vec3D
    width: int
    height: int
    depth: int

    def __init__(self, topLeft: Vec3D, width: int, height: int, depth: int) -> None:
        self.topLeft = topLeft
        self.width = width
        self.height = height
        self.depth = depth

    def box(self) -> Box3D:
        return Box3D.from_size(self.topLeft, Vec3D(self.width, self.height, self.depth))


class DataSourceId:
    team: str
    name: str

    def __init__(self, team: str, name: str, **kwargs: Any) -> None:
        self.team = team
        self.name = name

    def __hash__(self) -> int:
        return hash((self.team, self.name))


class DataLayer:
    suppoeted_categories = ["color", "segmentation"]
    suppoeted_element_classes = ["uint8", "uint16", "uint32", "uint64"]

    name: str
    category: str
    boundingBox: BoundingBox
    resolutions: List[Vec3D]
    elementClass: str

    def __init__(
        self,
        name: str,
        category: str,
        boundingBox: BoundingBox,
        resolutions: List[Vec3D],
        elementClass: str,
        **kwargs: Any
    ) -> None:
        self.name = name
        self.category = category
        self.boundingBox = boundingBox
        self.resolutions = resolutions
        self.elementClass = elementClass

        assert self.category in self.suppoeted_categories
        assert self.elementClass in self.suppoeted_element_classes


class DataSource:
    id: DataSourceId
    dataLayers: List[DataLayer]
    scale: Tuple[float, float, float]

    def __init__(
        self,
        id: DataSourceId,
        dataLayers: List[DataLayer],
        scale: Tuple[float, float, float],
    ) -> None:
        self.id = id
        self.dataLayers = dataLayers
        self.scale = scale

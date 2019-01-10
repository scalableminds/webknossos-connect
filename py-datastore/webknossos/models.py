from typing import List, Tuple


class DataStoreStatus:
    ok: bool
    url: str

    def __init__(self, ok, url, **kwargs):
        self.ok = ok
        self.url = url


class DataRequest:
    position: Tuple[int, int, int]
    zoomStep: int
    cubeSize: int
    fourBit: bool

    def __init__(self, position, zoomStep, cubeSize, fourBit, **kwargs):
        self.position = position
        self.zoomStep = zoomStep
        self.cubeSize = cubeSize
        self.fourBit = fourBit


class BoundingBox:
    topLeft: Tuple[int, int, int]
    width: int
    height: int
    depth: int

    def __init__(self, topLeft, width, height, depth):
        self.topLeft = topLeft
        self.width = width
        self.height = height
        self.depth = depth

        self.shape = self.size = (self.width, self.height, self.depth)
        self.bottomRight = tuple(map(sum, zip(self.topLeft, self.shape)))

    def union(self, other):
        topLeft = tuple(map(min, zip(self.topLeft, other.topLeft)))
        bottomRight = tuple(map(max, zip(self.bottomRight, other.bottomRight)))
        shape = tuple([high - low for low, high in zip(topLeft, bottomRight)])
        return BoundingBox(topLeft, *shape)


class DataSourceId:
    team: str
    name: str

    def __init__(self, team, name, **kwargs):
        self.team = team
        self.name = name


class DataLayer:
    suppoeted_categories = ["color", "segmentation"]
    suppoeted_element_classes = ["uint8", "uint16", "uint32", "uint64"]

    name: str
    category: str
    boundingBox: BoundingBox
    resolutions: List[Tuple[int, int, int]]
    elementClass: str

    def __init__(
        self, name, category, boundingBox, resolutions, elementClass, **kwargs
    ):
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

    def __init__(self, id, dataLayers, scale):
        self.id = id
        self.dataLayers = dataLayers
        self.scale = scale

from typing import List

from dataclasses import dataclass

from ...utils.types import Box3D, Vec3D
from ...webknossos.models import BoundingBox as WkBoundingBox
from ...webknossos.models import DataLayer as WkDataLayer
from ...webknossos.models import DataSource as WkDataSource
from ...webknossos.models import DataSourceId as WkDataSourceId
from ..backend import DatasetInfo


@dataclass(unsafe_hash=True)
class Channel:
    channel_name: str
    datatype: str
    resolutions: List[Vec3D]
    type: str

    def __post_init__(self) -> None:
        supported_datatypes = {
            "color": ["uint8", "uint16"],
            "segmentation": ["uint32", "uint64"],
        }
        if self.type == "image":
            self.type = "color"
        elif self.type == "annotation":
            self.type = "segmentation"
        assert self.type in supported_datatypes
        assert self.datatype in supported_datatypes[self.type]
        assert all(max(res) == 2 ** i for i, res in enumerate(self.resolutions))

    def wk_datatype(self) -> str:
        return "uint8" if self.type == "color" else "uint32"

    def to_webknossos(self, wk_bounding_box: WkBoundingBox) -> WkDataLayer:
        return WkDataLayer(
            self.channel_name,
            self.type,
            wk_bounding_box,
            self.resolutions,
            self.wk_datatype(),
        )


@dataclass(frozen=True)
class Experiment:
    collection_name: str
    experiment_name: str
    channels: List[Channel]


@dataclass(frozen=True)
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    domain: str
    experiment: Experiment
    username: str
    password: str
    bounding_box: Box3D
    global_scale: Vec3D

    def to_webknossos(self) -> WkDataSource:
        bounding_box = WkBoundingBox.from_box(self.bounding_box)
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            [
                channel.to_webknossos(bounding_box)
                for channel in self.experiment.channels
            ],
            self.global_scale,
        )

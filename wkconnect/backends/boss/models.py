from dataclasses import dataclass
from typing import Dict, List, Tuple

from ...utils.types import Box3D, Vec3D, Vec3Df
from ...webknossos.models import BoundingBox as WkBoundingBox
from ...webknossos.models import DataLayer as WkDataLayer
from ...webknossos.models import DataSource as WkDataSource
from ...webknossos.models import DataSourceId as WkDataSourceId
from ..backend import DatasetInfo
from .token_repository import TokenKey


class BossResolutionsError(RuntimeError):
    pass


@dataclass(unsafe_hash=True)
class Channel:
    datatype: str
    resolutions: Tuple[Vec3D, ...]
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
        if not all(max(res) == 2 ** i for i, res in enumerate(self.resolutions)):
            error_msg = "\n".join(
                [
                    "Boss resolutions do not follow power of two factors:",
                    str(self.resolutions),
                    f"Maximum dimensions should be {[2 ** i for i in range(len(self.resolutions))]}.",
                ]
            )
            raise BossResolutionsError(error_msg)

    def wk_datatype(self) -> str:
        return "uint8" if self.type == "color" else "uint32"


@dataclass(frozen=True)
class Experiment:
    collection_name: str
    experiment_name: str
    channels: Dict[str, Channel]

    def webknossos_layers(self, wk_bounding_box: WkBoundingBox) -> List[WkDataLayer]:
        return [
            WkDataLayer(
                channel_name,
                channel.type,
                wk_bounding_box,
                list(channel.resolutions),
                channel.wk_datatype(),
            )
            for channel_name, channel in self.channels.items()
        ]


@dataclass(frozen=True)
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    domain: str
    experiment: Experiment
    bounding_box: Box3D
    global_scale: Vec3Df
    token_key: TokenKey

    def to_webknossos(self) -> WkDataSource:
        bounding_box = WkBoundingBox.from_box(self.bounding_box)
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            self.experiment.webknossos_layers(bounding_box),
            self.global_scale,
        )

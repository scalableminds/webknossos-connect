from dataclasses import InitVar, dataclass, field

from ..backend import DatasetInfo
from ...webknossos.models import DataLayer as WkDataLayer

@dataclass
class Dataset(DatasetInfo):


    def to_webknossos(self, layer_name: str) -> WkDataLayer:
        return WkDataLayer(
            layer_name,
            "color",
            bounding_box(),
            [scale.resolution for scale in self.scales],
            self.wk_data_type(),
        )
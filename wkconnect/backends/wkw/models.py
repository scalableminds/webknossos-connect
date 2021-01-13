from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from wkcuber.api.Dataset import WKDataset
from wkcuber.api.Properties.LayerProperties import LayerProperties
from wkcuber.mag import Mag

from wkconnect.utils.types import Vec3D, Vec3Df

from ...webknossos.models import BoundingBox as WkBoundingBox
from ...webknossos.models import DataLayer as WkDataLayer
from ...webknossos.models import DataSource as WkDataSource
from ...webknossos.models import DataSourceId as WkDataSourceId
from ..backend import DatasetInfo


@dataclass(frozen=True)
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    dataset_handle: WKDataset = None

    def to_webknossos(self) -> WkDataSource:
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            [
                self.layer_to_webknossos(layer_name, layer)
                for layer_name, layer in self.dataset_handle.properties.data_layers.items()
            ],
            Vec3Df(*self.dataset_handle.properties.scale),
        )

    def layer_to_webknossos(
        self, layer_name: str, layer_properties: LayerProperties
    ) -> WkDataLayer:
        return WkDataLayer(
            layer_name,
            layer_properties._category,
            WkBoundingBox(
                topLeft=layer_properties._bounding_box["topLeft"],
                width=layer_properties._bounding_box["width"],
                height=layer_properties._bounding_box["height"],
                depth=layer_properties._bounding_box["depth"],
            ),
            [
                Vec3D(*mag.mag.to_array())
                for mag in layer_properties._wkw_magnifications
            ],
            layer_properties._element_class,
        )

    @lru_cache(maxsize=2 ** 12)
    def read_data(
        self, layer_name: str, zoom_step: int, wk_offset: Vec3D, shape: Vec3D
    ) -> Optional[np.ndarray]:
        layer = self.dataset_handle.get_layer(layer_name)
        available_mags = sorted([Mag(mag).mag for mag in layer.mags.keys()])
        mag = available_mags[zoom_step]
        mag_dataset = layer.get_mag(mag)
        offset = (
            np.array([wk_offset.x, wk_offset.y, wk_offset.z]) / np.array(Mag(mag).mag)
        ).astype(np.uint32)
        if not mag_dataset.view._is_opened:
            mag_dataset.open()

        return mag_dataset.read(tuple(offset), shape)

    def clear_cache(self) -> None:
        self.read_data.cache_clear()  # pylint: disable=no-member
        for layer_handle in self.dataset_handle.layers.values():
            for mag in layer_handle.mags.values():
                if mag.view._is_opened:
                    mag.close()

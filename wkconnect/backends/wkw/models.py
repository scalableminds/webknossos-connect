from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from async_lru import alru_cache
from wkcuber.api.Dataset import WKDataset
from wkcuber.api.Properties.LayerProperties import LayerProperties
from wkcuber.mag import Mag

from wkconnect.utils.types import Vec3D, Vec3Df

from ...fast_wkw import DatasetCache, DatasetHandle  # pylint: disable=no-name-in-module
from ...webknossos.models import BoundingBox as WkBoundingBox
from ...webknossos.models import DataLayer as WkDataLayer
from ...webknossos.models import DataSource as WkDataSource
from ...webknossos.models import DataSourceId as WkDataSourceId
from ..backend import DatasetInfo


@dataclass(frozen=True)
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    dataset_handle: WKDataset
    wkw_cache: DatasetCache

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
                topLeft=Vec3D(*layer_properties._bounding_box["topLeft"]),
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

    @lru_cache(maxsize=1000)
    def get_data_handle(
        self, layer_name: str, zoom_step: int
    ) -> Optional[Tuple[DatasetHandle, Mag]]:
        layer = self.dataset_handle.get_layer(layer_name)
        # Finding the right mag for the zoom_step:
        # 2 ** zoomStep = max(mag.x, mag.y, mag.z)
        mag_datasets = [
            mag_dataset
            for mag_dataset in layer.mags.values()
            if (2 ** zoom_step) == Mag(mag_dataset.name).as_np().max()
        ]
        if len(mag_datasets) < 1:
            return None
        mag_dataset = mag_datasets[0]

        data_handle = self.wkw_cache.get_dataset(str(mag_dataset.view.path))
        return (data_handle, Mag(mag_dataset.name))

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def read_data(
        self, layer_name: str, zoom_step: int, wk_offset: Vec3D, shape: Vec3D
    ) -> Optional[np.ndarray]:
        assert shape == Vec3D(
            32, 32, 32
        ), "Only buckets of 32 edge length are supported"
        assert shape % 32 == Vec3D(0, 0, 0), "Only 32-aligned buckets are supported"
        data_handle_opt = self.get_data_handle(layer_name, zoom_step)
        if data_handle_opt is None:
            return None
        data_handle, mag = data_handle_opt
        offset = (
            np.array([wk_offset.x, wk_offset.y, wk_offset.z]) / mag.as_np()
        ).astype(np.uint32)
        block = await data_handle.read_block(tuple(offset))
        return np.frombuffer(block.buf, dtype=np.dtype(block.dtype)).reshape(
            block.shape, order="F"
        )

    def clear_cache(self) -> None:
        self.read_data.cache_clear()  # pylint: disable=no-member
        self.get_data_handle.cache_clear()  # pylint: disable=no-member
        self.wkw_cache.clear_cache_prefix(str(self.dataset_handle.path))

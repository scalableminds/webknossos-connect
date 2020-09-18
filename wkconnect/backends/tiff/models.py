import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tifffile import TiffFile

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
    scale: Vec3Df

    def to_webknossos(self) -> WkDataSource:
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            [self.layer_to_webknossos()],
            self.scale,
        )

    def layer_to_webknossos(self) -> WkDataLayer:
        layer_name = "color"
        filename = self.path() / layer_name / "0.tif"
        mags, tile_shape, page_shapes, dtype, _ = self.read_properties(filename)

        return WkDataLayer(
            layer_name,
            "color",
            WkBoundingBox(
                topLeft=Vec3D(0, 0, 0),
                width=page_shapes[0][0],
                height=page_shapes[0][1],
                depth=1,
            ),
            mags,
            "uint8",
        )

    @lru_cache(maxsize=2 ** 12)
    def read_data(
        self, layer_name: str, zoom_step: int, wk_offset: Vec3D, shape: Vec3D
    ) -> Optional[np.ndarray]:
        filename = self.path() / layer_name / "0.tif"

        mag_factor = 2 ** zoom_step
        wk_offset_scaled = wk_offset // mag_factor

        tile_coordinate_offset, tile_data = self.read_tile(
            str(filename), zoom_step, (wk_offset_scaled.x, wk_offset_scaled.y)
        )
        left_in_tile = wk_offset_scaled[0] - tile_coordinate_offset[0]
        top_in_tile = wk_offset_scaled[1] - tile_coordinate_offset[1]
        right_in_tile = left_in_tile + shape.x
        bottom_in_tile = top_in_tile + shape.y

        cropout = tile_data[
            top_in_tile:bottom_in_tile, left_in_tile:right_in_tile
        ].transpose()

        bucket = np.zeros(shape, dtype=cropout.dtype)
        bucket[:, :, 0] = cropout
        return bucket

    def path(self) -> Path:
        return Path("data", "binary", self.organization_name, self.dataset_name)

    def clear_cache(self) -> None:
        self.read_properties.cache_clear()  # pylint: disable=no-member
        self.read_mmapped.cache_clear()  # pylint: disable=no-member

    @lru_cache(5)
    def read_properties(
        self, filename: str
    ) -> Tuple[
        List[Vec3D], Tuple[int, int], List[Tuple[int, int]], np.dtype, List[List[int]]
    ]:
        with TiffFile(filename) as tif:
            mags = [Vec3D(2 ** mag, 2 ** mag, 1) for mag in range(len(tif.pages))]
            assert (
                len(mags) > 0
            ), f"no magnifications found. empty tif file at {filename}?"
            tags = tif.pages[0].tags
            tile_shape = (tags[322].value, tags[323].value)
            page_shapes = [
                (page.tags[256].value, page.tags[257].value) for page in tif.pages
            ]
            dtype = tif.byteorder + tif.pages[0].dtype.char
            tile_byte_offsets = [page.tags[324].value for page in tif.pages]
            return mags, tile_shape, page_shapes, dtype, tile_byte_offsets

    def read_tile(
        self, filename: str, page: int, target_offset: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        _, tile_shape, page_shapes, dtype, tile_byte_offsets = self.read_properties(
            filename
        )

        target_tile = (
            target_offset[0] // tile_shape[0],
            target_offset[1] // tile_shape[1],
        )
        tile_coordinate_offset = (
            target_tile[0] * tile_shape[0],
            target_tile[1] * tile_shape[1],
        )

        tiles_per_row = math.ceil(page_shapes[page][0] / tile_shape[0])
        target_tile_index = target_tile[0] + target_tile[1] * tiles_per_row
        tile_byte_offset = tile_byte_offsets[page][target_tile_index]

        data = self.read_mmapped(filename, dtype, tile_byte_offset, tile_shape)
        return tile_coordinate_offset, data

    @lru_cache(maxsize=2 ** 9)
    def read_mmapped(
        self, filename: str, dtype: np.dtype, byte_offset: int, shape: Tuple[int, int]
    ) -> np.ndarray:
        image = np.zeros(shape, dtype=dtype)
        mapped = np.memmap(filename, dtype, "r", byte_offset, shape, "C")
        image[:] = mapped
        del mapped
        return image

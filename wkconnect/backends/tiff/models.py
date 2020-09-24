import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from os import listdir
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

logger = logging.getLogger()

# Accessing the tiff header tags via their numbers:
TAG_TILE_WIDTH = 322
TAG_TILE_HEIGHT = 323
TAG_TILE_BYTE_OFFSETS = 324
TAG_PAGE_WIDTH = 256
TAG_PAGE_HEIGHT = 257


@dataclass(frozen=True)
class Dataset(DatasetInfo):
    organization_name: str
    dataset_name: str
    scale: Vec3Df

    def to_webknossos(self) -> WkDataSource:
        return WkDataSource(
            WkDataSourceId(self.organization_name, self.dataset_name),
            [self.layer_to_webknossos(layer_name) for layer_name in self.layers()],
            self.scale,
        )

    def layers(self) -> List[str]:
        layers = []
        path = str(self.path())
        for filename in listdir(path):
            filepath = Path(filename)
            if filepath.suffix == ".tif":
                layers.append(filepath.stem)
        if len(layers) == 0:
            logger.info(f"No valid layers found for tif dataset at {path}")
        return layers

    def layer_to_webknossos(self, layer_name: str) -> WkDataLayer:
        mags, _, page_shapes, dtype, _ = self.read_properties(layer_name)

        return WkDataLayer(
            layer_name,
            "segmentation" if layer_name.startswith("segmentation") else "color",
            WkBoundingBox(
                topLeft=Vec3D(0, 0, 0),
                width=page_shapes[0][0],
                height=page_shapes[0][1],
                depth=1,
            ),
            mags,
            str(self.repair_dtype(dtype)),
        )

    @lru_cache(maxsize=2 ** 12)
    def read_data(
        self, layer_name: str, zoom_step: int, wk_offset: Vec3D, shape: Vec3D
    ) -> Optional[np.ndarray]:
        mag_factor = 2 ** zoom_step
        wk_offset_scaled = wk_offset // mag_factor

        tile_coordinate_offset, tile_data = self.read_tile(
            layer_name,
            zoom_step,
            (wk_offset_scaled.x, wk_offset_scaled.y),
            (shape.x, shape.y),
        )
        left_in_tile = wk_offset_scaled[0] - tile_coordinate_offset[0]
        top_in_tile = wk_offset_scaled[1] - tile_coordinate_offset[1]
        right_in_tile = left_in_tile + shape.x
        bottom_in_tile = top_in_tile + shape.y

        cropout = tile_data[
            top_in_tile:bottom_in_tile, left_in_tile:right_in_tile
        ].transpose()

        bucket = np.zeros(shape, dtype=cropout.dtype)
        bucket[0 : cropout.shape[0], 0 : cropout.shape[1], 0] = cropout
        return bucket

    def path(self) -> Path:
        return Path("data", "binary", self.organization_name, self.dataset_name)

    def layer_filepath(self, layer_name: str) -> Path:
        return self.path() / f"{layer_name}.tif"

    def clear_cache(self) -> None:
        self.read_properties.cache_clear()  # pylint: disable=no-member
        self.read_mmapped.cache_clear()  # pylint: disable=no-member

    @lru_cache(5)
    def read_properties(
        self, layer_name: str
    ) -> Tuple[
        List[Vec3D],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        np.dtype,
        List[List[int]],
    ]:
        filepath = self.layer_filepath(layer_name)
        with TiffFile(str(filepath)) as tif:
            assert (
                len(tif.pages) > 0
            ), f"No magnifications found. Empty tiff at {str(filepath)}?"

            mags = [Vec3D(2 ** mag, 2 ** mag, 1) for mag in range(len(tif.pages))]
            for i, page in enumerate(tif.pages):
                assert (
                    TAG_TILE_WIDTH in page.tags
                    and TAG_TILE_HEIGHT in page.tags
                    and TAG_TILE_BYTE_OFFSETS in page.tags
                    and TAG_PAGE_WIDTH in page.tags
                    and TAG_PAGE_HEIGHT in page.tags
                ), (
                    f"Incomplete tags in tif file at {str(filepath)} for page {i}. "
                    + "Only tiled tifs with resolution pyramid are supported."
                )
            page_shapes = [
                (page.tags[TAG_PAGE_WIDTH].value, page.tags[TAG_PAGE_HEIGHT].value)
                for page in tif.pages
            ]
            tile_byte_offsets = [
                page.tags[TAG_TILE_BYTE_OFFSETS].value for page in tif.pages
            ]
            tile_shapes = [
                (page.tags[TAG_TILE_WIDTH].value, page.tags[TAG_TILE_HEIGHT].value)
                for page in tif.pages
            ]
            for i, tile_shape in enumerate(tile_shapes):
                assert (
                    math.log2(tile_shape[0]).is_integer()
                    and math.log2(tile_shape[1]).is_integer()
                    and tile_shape[0] >= 32
                    and tile_shape[1] >= 32
                    and tile_shape[0] <= 2048
                    and tile_shape[1] <= 2048
                ), f"Tiff tile shapes must be power of two, min 32 and max 2048. Found {tile_shape} in page {i} at {str(filepath)}."
            dtype = tif.byteorder + tif.pages[0].dtype.char
            return mags, tile_shapes, page_shapes, dtype, tile_byte_offsets

    def read_tile(
        self,
        layer_name: str,
        page: int,
        target_offset: Tuple[int, int],
        target_shape: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        _, tile_shapes, page_shapes, dtype, tile_byte_offsets = self.read_properties(
            layer_name
        )
        tile_shape = tile_shapes[page]

        target_tile = (
            target_offset[0] // tile_shape[0],
            target_offset[1] // tile_shape[1],
        )
        target_tile_bottomright = (
            (target_offset[0] + target_shape[0] - 1) // tile_shape[0],
            (target_offset[1] + target_shape[1] - 1) // tile_shape[1],
        )
        assert (
            target_tile == target_tile_bottomright
        ), f"Requested tif data at {target_offset} from page {page} of {str(self.path())} that cannot be served by loading a single tile."

        tile_coordinate_offset = (
            target_tile[0] * tile_shape[0],
            target_tile[1] * tile_shape[1],
        )

        tiles_per_row = math.ceil(page_shapes[page][0] / tile_shape[0])
        target_tile_index = target_tile[0] + target_tile[1] * tiles_per_row
        tile_byte_offset = tile_byte_offsets[page][target_tile_index]

        data = self.read_mmapped(layer_name, dtype, tile_byte_offset, tile_shape)
        return tile_coordinate_offset, data

    @lru_cache(maxsize=2 ** 9)
    def read_mmapped(
        self, layer_name: str, dtype: np.dtype, byte_offset: int, shape: Tuple[int, int]
    ) -> np.ndarray:
        image = np.zeros(shape, dtype=self.repair_dtype(dtype))
        mapped = np.memmap(
            str(self.layer_filepath(layer_name)), dtype, "r", byte_offset, shape, "C"
        )
        # copy content from mmap handle to array in order to avoid holding open thousands of handles.
        image[:] = mapped
        del mapped
        return image

    @lru_cache(5)
    def repair_dtype(self, dtype: np.dtype) -> np.dtype:
        # wk does not support signed datasets. We convert their data to the respective unsigned counterparts.
        dtype = np.dtype(dtype)
        if dtype == np.int8:
            return np.uint8(0).dtype
        if dtype == np.int16:
            return np.uint16(0).dtype
        if dtype == np.int32:
            return np.uint32(0).dtype
        return dtype

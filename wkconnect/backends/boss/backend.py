import asyncio
import logging
from typing import Dict, Optional, Tuple, cast

import blosc
import numpy as np
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientResponseError
from async_lru import alru_cache

from ...utils.si import convert_si_units
from ...utils.types import JSON, Box3D, HashableDict, Vec3D, Vec3Df
from ..backend import Backend, DatasetInfo
from .client import Client
from .models import Channel, Dataset, Experiment
from .token_repository import TokenKey, TokenRepository

logger = logging.getLogger()


class Boss(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        self.tokens = TokenRepository(http_client)
        self.http_client = http_client
        self.client = Client(http_client, self.tokens)

    async def __handle_new_channel(
        self,
        domain: str,
        collection: str,
        experiment: str,
        channel: str,
        token_key: TokenKey,
    ) -> Tuple[str, Channel]:
        channel_info = await self.client.get_channel(
            domain, collection, experiment, channel, token_key
        )
        assert channel_info["base_resolution"] == 0
        if channel_info["downsample_status"] != "DOWNSAMPLED":
            logger.warn(
                f"BOSS did not finish downsampling for \"{'/'.join([domain, collection, experiment])}\", current status is {channel_info['downsample_status']}."
            )

        downsample_info = await self.client.get_downsample(
            domain, collection, experiment, channel, token_key
        )
        resolutions = tuple(
            Vec3Df(*i) for i in sorted(downsample_info["voxel_size"].values())
        )
        normalized_resolutions = tuple(
            (i / resolutions[0]).to_int() for i in resolutions
        )

        return (
            channel,
            Channel(
                channel_info["datatype"], normalized_resolutions, channel_info["type"]
            ),
        )

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        domain = dataset_info["domain"]
        assert domain.startswith("https://api") or domain.startswith("http://api")

        collection = dataset_info["collection"]
        experiment = dataset_info["experiment"]

        token_key = TokenKey(
            domain,
            dataset_info.get("username", None),
            dataset_info.get("password", None),
            dataset_info.get("token", None),
        )

        experiment_info = await self.client.get_experiment(
            domain, collection, experiment, token_key
        )
        assert experiment_info["num_time_samples"] == 1

        coord_info = await self.client.get_coord(
            domain, experiment_info["coord_frame"], token_key
        )
        bounding_box = Box3D(
            Vec3D(coord_info["x_start"], coord_info["y_start"], coord_info["z_start"]),
            Vec3D(coord_info["x_stop"], coord_info["y_stop"], coord_info["z_stop"]),
        )
        global_scale = Vec3Df(
            *(
                convert_si_units(
                    coord_info[f"{dim}_voxel_size"],
                    coord_info["voxel_unit"],
                    "nanometers",
                )
                for dim in "xyz"
            )
        )

        channels = await asyncio.gather(
            *(
                self.__handle_new_channel(
                    domain, collection, experiment, channel, token_key
                )
                for channel in experiment_info["channels"]
            )
        )

        experiment = Experiment(collection, experiment, HashableDict(channels))

        return Dataset(
            organization_name,
            dataset_name,
            domain,
            experiment,
            bounding_box,
            global_scale,
            token_key,
        )

    @alru_cache(maxsize=2 ** 12, cache_exceptions=False)
    async def __cached_read_data(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        dataset = cast(Dataset, abstract_dataset)
        channel = dataset.experiment.channels[layer_name]

        channel_offset = wk_offset // channel.resolutions[zoom_step]
        box = Box3D.from_size(channel_offset, shape)

        compressed_data = await self.client.get_cutout(
            dataset.domain,
            dataset.experiment.collection_name,
            dataset.experiment.experiment_name,
            layer_name,
            zoom_step,
            box,
            dataset.token_key,
        )

        byte_data = blosc.decompress(compressed_data)
        if channel.type == "color" and channel.datatype == "uint16":
            # this will be downscaled to uint8,
            # we want to take the higher-order bits here
            data = np.frombuffer(byte_data, dtype=">u2")
        else:
            data = np.frombuffer(byte_data, dtype=channel.datatype)
        return data.astype(channel.wk_datatype()).reshape(shape, order="F")

    async def read_data(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> Optional[np.ndarray]:
        try:
            return await self.__cached_read_data(
                abstract_dataset, layer_name, zoom_step, wk_offset, shape
            )
        except ClientResponseError:
            # will be reported in MISSING-BUCKETS, frontend will retry
            return None

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        pass

    async def on_shutdown(self) -> None:
        await self.tokens.logout_all()

from typing import Dict, cast

import blosc
import numpy as np
from aiohttp import ClientResponse, ClientSession
from aiohttp.client_exceptions import ClientResponseError

from ...utils.si import convert_si_units
from ...utils.types import JSON, Box3D, Vec3D
from ..backend import Backend, DatasetInfo
from .client import Client
from .models import Channel, Dataset, Experiment
from .token_repository import TokenRepository


class Boss(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        self.tokens = TokenRepository(http_client)
        self.http_client = http_client
        self.client = Client(http_client, self.tokens)

    async def auth_get(
        self, url: str, dataset: TokenRepository.DatasetDescriptor
    ) -> ClientResponse:
        return await self.http_client.get(
            url, headers=await self.tokens.get_header(dataset)
        )

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        domain = dataset_info["domain"]
        assert domain.startswith("https://api") or domain.startswith("http://api")

        username = dataset_info["username"]
        password = dataset_info["password"]

        collection = dataset_info["collection"]
        experiment = dataset_info["experiment"]

        token_key = (domain, username, password)

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
        global_scale = Vec3D(
            *(
                int(
                    convert_si_units(
                        coord_info[f"{dim}_voxel_size"],
                        coord_info["voxel_unit"],
                        "nanometers",
                    )
                )
                for dim in "xyz"
            )
        )

        channels = []
        # TODO use extra method & asyncio.gather
        for channel in experiment_info["channels"]:
            channel_info = await self.client.get_channel(
                domain, collection, experiment, channel, token_key
            )
            if channel_info["type"] != "image":
                # TODO add issue for annotations
                continue
            assert channel_info["base_resolution"] == 0
            assert channel_info["downsample_status"] == "DOWNSAMPLED"
            datatype = channel_info["datatype"]
            assert datatype in ["uint8", "uint16"]

            downsample_info = await self.client.get_downsample(
                domain, collection, experiment, channel, token_key
            )
            # TODO consider using cuboid size to download & for caching
            resolutions = [
                Vec3D(*map(int, i))
                for i in sorted(downsample_info["voxel_size"].values())
            ]
            resolutions = [i // resolutions[0] for i in resolutions]
            assert all(max(res) == 2 ** i for i, res in enumerate(resolutions))

            channels.append(Channel(channel, datatype, resolutions))

        experiment = Experiment(collection, experiment, channels)

        return Dataset(
            organization_name,
            dataset_name,
            domain,
            experiment,
            username,
            password,
            bounding_box,
            global_scale,
        )

    async def read_data(
        self,
        abstract_dataset: DatasetInfo,
        layer_name: str,
        zoom_step: int,
        wk_offset: Vec3D,
        shape: Vec3D,
    ) -> np.ndarray:
        dataset = cast(Dataset, abstract_dataset)

        channel = [
            i for i in dataset.experiment.channels if i.channel_name == layer_name
        ][0]

        channel_offset = wk_offset // channel.resolutions[zoom_step]
        box = Box3D.from_size(channel_offset, shape)

        try:
            compressed_data = await self.client.get_cutout(
                dataset.domain,
                dataset.experiment.collection_name,
                dataset.experiment.experiment_name,
                layer_name,
                zoom_step,
                box,
                dataset,
            )
        except ClientResponseError:
            # will be reported in MISSING-BUCKETS, frontend will retry
            return None

        data = blosc.decompress(compressed_data)

        data_8bit = (
            # TODO issue for 16 bit
            np.frombuffer(data, dtype="uint8")
            if channel.datatype == "uint8"
            else np.frombuffer(data, dtype=">u2").astype("uint8")
        )
        return data_8bit.reshape(shape, order="F")

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        pass

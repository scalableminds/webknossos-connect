from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import blosc
import numpy as np
from aiohttp import ClientResponse, ClientSession
from aiohttp.client_exceptions import ClientResponseError

from dataclasses import dataclass

from ...utils.si import convert_si_units
from ...utils.types import JSON, Box3D, Vec3D
from ...webknossos.models import BoundingBox as WKBoundingBox
from ...webknossos.models import DataLayer as WKDataLayer
from ...webknossos.models import DataSource as WKDataSource
from ...webknossos.models import DataSourceId as WKDataSourceId
from ..backend import Backend, DatasetInfo

DecoderFn = Callable[[bytes, str, Vec3D, Optional[Vec3D]], np.ndarray]
Chunk = Tuple[Box3D, np.ndarray]


@dataclass(frozen=True)
class Channel:
    channel_name: str
    datatype: str
    resolutions: List[Vec3D]

    def to_webknossos(self, wk_bounding_box: WKBoundingBox) -> WKDataLayer:
        return WKDataLayer(
            self.channel_name, "color", wk_bounding_box, self.resolutions, self.datatype
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

    def to_webknossos(self) -> WKDataSource:
        bounding_box = WKBoundingBox.from_box(self.bounding_box)
        return WKDataSource(
            WKDataSourceId(self.organization_name, self.dataset_name),
            [
                channel.to_webknossos(bounding_box)
                for channel in self.experiment.channels
            ],
            self.global_scale,
        )


class TokenRepository:
    def __init__(self, http_client: ClientSession):
        self.http_client = http_client
        self.token_infos: Dict = {}

    DatasetDescriptor = Union[Dataset, Tuple[str, str, str]]

    async def get(self, dataset: DatasetDescriptor) -> str:
        if isinstance(dataset, Dataset):
            domain, username, password = (
                dataset.domain,
                dataset.username,
                dataset.password,
            )
        else:
            domain, username, password = dataset
        key = (domain, username, password)

        request_new = True
        if key in self.token_infos:
            token_info = self.token_infos[key]
            if time.monotonic() - token_info["time"] < token_info["expires_in"] - 10:
                # TODO: maybe refresh via refresh_token if still valid?
                # see https://stackoverflow.com/questions/51386337/refresh-access-token-via-refresh-token-in-keycloak
                request_new = False

        if request_new:
            client_id = "endpoint"
            login_realm = "boss"

            # domain:   https://api.boss.neurodata.io
            # auth_url: https://auth.boss.neurodata.io/auth
            protocol = domain.split("://", 1)[0]
            domain_end = domain.split(".", 1)[1]
            auth_url = f"{protocol}://auth.{domain_end}/auth"

            url = f"{auth_url}/realms/{login_realm}/protocol/openid-connect/token"
            data = {
                "grant_type": "password",
                "client_id": client_id,
                "username": username,
                "password": password,
            }
            now = time.monotonic()
            async with await self.http_client.post(url, data=data) as r:
                token_info = await r.json()
            # token_info:
            # {
            #     "access_token": "…",
            #     "expires_in": 1800,
            #     "refresh_expires_in": 3600,
            #     "refresh_token": "…",
            #     "token_type": "bearer",
            #     "id_token": "…",
            #     "not-before-policy": 0,
            #     "session_state": "…"
            # }

            token_info["time"] = now
            self.token_infos[key] = token_info

        return "Bearer " + token_info["access_token"]

    async def get_header(self, dataset: DatasetDescriptor) -> Dict[str, str]:
        return {"Authorization": await self.get(dataset)}


class Boss(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        self.tokens = TokenRepository(http_client)
        self.http_client = http_client

    async def auth_get(
        self, url: str, dataset: TokenRepository.DatasetDescriptor
    ) -> ClientResponse:
        return await self.http_client.get(
            url, headers=await self.tokens.get_header(dataset)
        )

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        # TODO get them from dataset_info
        domain = "https://api.boss.neurodata.io"
        assert domain.startswith("https://api") or domain.startswith("http://api")

        username = "jstriebel"
        password = "{,vEzT7J?-5_"

        # coll_name = "ara_2016"
        # exp_name = "sagittal_50um"
        coll_name = "kasthuri"
        exp_name = "kasthuri11"

        experiment_url = "/".join(
            [domain, "v1", "collection", coll_name, "experiment", exp_name]
        )
        token_key = (domain, username, password)
        async with await self.auth_get(experiment_url, token_key) as r:
            experiment_info = await r.json()

        # experiment_info:
        # {
        #     "channels": ["annotation_50um", "average_50um", "nissl_50um"],
        #     "name": "sagittal_50um",
        #     "description": "Allen Reference Atlases ingested from the source at 50 um resolution",
        #     "collection": "ara_2016",
        #     "coord_frame": "ara_2016_50um",
        #     "num_hierarchy_levels": 1,
        #     "hierarchy_method": "isotropic",
        #     "num_time_samples": 1,
        #     "time_step": None,
        #     "time_step_unit": "",
        #     "creator": "vikramc"
        # }

        assert experiment_info["num_time_samples"] == 1

        coord_url = f"{domain}/v1/coord/{experiment_info['coord_frame']}"
        async with await self.auth_get(coord_url, token_key) as r:
            coord_info = await r.json()

        # coord_info:
        # {
        #     "name": "ara_2016_50um",
        #     "description": "50 um Allen reference atlas",
        #     "x_start": 0,
        #     "x_stop": 264,
        #     "y_start": 0,
        #     "y_stop": 160,
        #     "z_start": 0,
        #     "z_stop": 228,
        #     "x_voxel_size": 50.0,
        #     "y_voxel_size": 50.0,
        #     "z_voxel_size": 50.0,
        #     "voxel_unit": "micrometers"
        # }

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
        for channel_name in experiment_info["channels"]:
            channel_url = "/".join(
                [
                    domain,
                    "v1",
                    "collection",
                    coll_name,
                    "experiment",
                    exp_name,
                    "channel",
                    channel_name,
                ]
            )
            async with await self.auth_get(channel_url, token_key) as r:
                channel_info = await r.json()

            # channel_info:
            # {
            #     "name": "nissl_50um",
            #     "description": "",
            #     "experiment": "sagittal_50um",
            #     "default_time_sample": 0,
            #     "type": "image",                   | "annotation"
            #     "base_resolution": 0,
            #     "datatype": "uint16",
            #     "creator": "vikramc",
            #     "sources": [],
            #     "downsample_status": "DOWNSAMPLED",
            #     "related": []
            # }

            if channel_info["type"] != "image":
                # TODO add issue for annotations
                continue
            assert channel_info["base_resolution"] == 0
            assert channel_info["downsample_status"] == "DOWNSAMPLED"
            datatype = channel_info["datatype"]
            assert datatype in ["uint8", "uint16"]

            downsample_url = "/".join(
                [domain, "v1", "downsample", coll_name, exp_name, channel_name]
            )
            async with await self.auth_get(downsample_url, token_key) as r:
                downsample_info = await r.json()

            # downsample_info:
            # {
            #     "num_hierarchy_levels": 1,
            #     "cuboid_size": { "0": [512, 512, 16] },
            #     "extent": { "0": [264, 160, 228] },
            #     "status": "DOWNSAMPLED",
            #     "voxel_size": { "0": [50.0, 50.0, 50.0] }
            # }

            resolutions = [
                Vec3D(*map(int, i))
                for i in sorted(downsample_info["voxel_size"].values())
            ]
            assert resolutions[0] == global_scale
            resolutions = [i // resolutions[0] for i in resolutions]
            assert all(max(res) == 2 ** i for i, res in enumerate(resolutions))

            channels.append(Channel(channel_name, datatype, resolutions))

        experiment = Experiment(coll_name, exp_name, channels)

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
        # TODO assert layer_name in dataset channels
        url = "/".join(
            [
                dataset.domain,
                "v1",
                "cutout",
                dataset.experiment.collection_name,
                dataset.experiment.experiment_name,
                layer_name,
                str(zoom_step),
                *(f"{l}:{r}" for l, r in zip(*box)),
            ]
        )
        try:
            async with await self.auth_get(url, dataset) as r:
                compressed_data = await r.read()
        except ClientResponseError:
            # report in MISSING-BUCKETS, frontend will retry
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

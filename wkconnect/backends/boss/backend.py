from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, cast

import blosc
import numpy as np
from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientResponseError

from dataclasses import InitVar, dataclass, field

from ...utils.si import convert_si_units
from ...utils.types import JSON, Box3D, Vec3D
from ...webknossos.models import BoundingBox as WKBoundingBox
from ...webknossos.models import DataLayer as WKDataLayer
from ...webknossos.models import DataSource as WKDataSource
from ...webknossos.models import DataSourceId as WKDataSourceId
from ..backend import Backend, DatasetInfo

DecoderFn = Callable[[bytes, str, Vec3D, Optional[Vec3D]], np.ndarray]
Chunk = Tuple[Box3D, np.ndarray]


@dataclass
class Experiment:
    collection_name: str
    experiment_name: str
    # InitVar allows to consume mesh argument in init without storing it
    channel_names: InitVar[List[str]]
    channels: List["Channel"] = field(init=False)

    def __post_init__(self, channel_names: List[str]) -> None:
        self.channels = [Channel(self, channel_name) for channel_name in channel_names]


@dataclass(frozen=True)
# TODO maybe remove this class and only use channel_names in Experiment
class Channel:
    """
    Must not be initialized manually, use Experiment instead.
    """

    experiment: Experiment
    channel_name: str


@dataclass
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
        # TODO all
        return WKDataSource(
            WKDataSourceId(self.organization_name, self.dataset_name),
            [
                WKDataLayer(
                    self.experiment.channels[0].channel_name,
                    "color",
                    WKBoundingBox.from_box(self.bounding_box),
                    [Vec3D(1, 1, 1)],
                    "uint8",
                )
            ],
            self.global_scale,
        )


class TokenRepository:
    def __init__(self, http_client):
        self.http_client = http_client
        self.token_infos = {}

    async def get(self, dataset):
        if isinstance(dataset, Dataset):
            domain, username, password = (
                dataset.domain,
                dataset.username,
                dataset.password,
            )
        else:
            domain, username, password = dataset
        key = (domain, username, password)
        if key in self.token_infos:
            # TODO check if token still valid
            token_info = self.token_infos[key]
        else:
            client_id = "endpoint"
            login_realm = "boss"

            # TODO
            # from ng: let authServer = `https://auth${baseHostName[1]}/auth`;
            authdomain = "https://auth.boss.neurodata.io/auth"
            url = f"{authdomain}/realms/{login_realm}/protocol/openid-connect/token"
            data = {
                "grant_type": "password",
                "client_id": client_id,
                "username": username,
                "password": password,
            }
            async with await self.http_client.post(url, data=data) as r:
                token_info = await r.json()
            self.token_infos[key] = token_info

        return "Bearer " + token_info["access_token"]

    async def get_header(self, dataset):
        return {"Authorization": await self.get(dataset)}


class Boss(Backend):
    def __init__(self, config: Dict, http_client: ClientSession) -> None:
        self.tokens = TokenRepository(http_client)
        self.http_client = http_client

    async def auth_get(self, url, dataset):
        return await self.http_client.get(
            url, headers=await self.tokens.get_header(dataset)
        )

    async def handle_new_dataset(
        self, organization_name: str, dataset_name: str, dataset_info: JSON
    ) -> DatasetInfo:
        domain = "https://api.boss.neurodata.io"
        assert domain.startswith("https://api") or domain.startswith("http://api")

        username = "jstriebel"
        password = "{,vEzT7J?-5_"

        coll_name = "ara_2016"
        exp_name = "sagittal_50um"

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
                int(convert_si_units(
                    coord_info[f"{dim}_voxel_size"],
                    coord_info["voxel_unit"],
                    "nanometers",
                ))
                for dim in "xyz"
            )
        )

        chan_name = "nissl_50um"
        for channel_name in experiment_info["channels"]:
            pass

        experiment = Experiment(coll_name, exp_name, [chan_name])

        return Dataset(
            organization_name, dataset_name, domain, experiment, username, password, bounding_box, global_scale
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

        box = Box3D.from_size(wk_offset, shape)
        # TODO assert layer_name in dataset channels
        resolution = 0
        url = "/".join(
            [
                dataset.domain,
                "v1",
                "cutout",
                dataset.experiment.collection_name,
                dataset.experiment.experiment_name,
                layer_name,
                str(resolution),
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
        return (
            np.frombuffer(data, dtype=">u2").astype("uint8").reshape(shape, order="F")
        )

    def clear_dataset_cache(self, dataset: DatasetInfo) -> None:
        pass

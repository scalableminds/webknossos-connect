from aiohttp import ClientResponse, ClientSession

from ...utils.types import JSON, Box3D
from .token_repository import TokenKey, TokenRepository


class Client:
    def __init__(self, http_client: ClientSession, tokens: TokenRepository) -> None:
        self.http_client = http_client
        self.tokens = tokens

    async def auth_get(self, url: str, token_key: TokenKey) -> ClientResponse:
        return await self.http_client.get(
            url, headers=await self.tokens.get_header(token_key)
        )

    async def get_experiment(
        self, domain: str, collection: str, experiment: str, token_key: TokenKey
    ) -> JSON:
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
        experiment_url = "/".join(
            [domain, "v1", "collection", collection, "experiment", experiment]
        )
        async with await self.auth_get(experiment_url, token_key) as r:
            return await r.json()

    async def get_coord(
        self, domain: str, coord_frame: str, token_key: TokenKey
    ) -> JSON:
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
        coord_url = f"{domain}/v1/coord/{coord_frame}"
        async with await self.auth_get(coord_url, token_key) as r:
            return await r.json()

    async def get_channel(
        self,
        domain: str,
        collection: str,
        experiment: str,
        channel: str,
        token_key: TokenKey,
    ) -> JSON:
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
        channel_url = "/".join(
            [
                domain,
                "v1",
                "collection",
                collection,
                "experiment",
                experiment,
                "channel",
                channel,
            ]
        )
        async with await self.auth_get(channel_url, token_key) as r:
            return await r.json()

    async def get_downsample(
        self,
        domain: str,
        collection: str,
        experiment: str,
        channel: str,
        token_key: TokenKey,
    ) -> JSON:
        # {
        #     "num_hierarchy_levels": 1,
        #     "cuboid_size": { "0": [512, 512, 16] },
        #     "extent": { "0": [264, 160, 228] },
        #     "status": "DOWNSAMPLED",
        #     "voxel_size": { "0": [50.0, 50.0, 50.0] }
        # }
        downsample_url = "/".join(
            [domain, "v1", "downsample", collection, experiment, channel]
        )
        async with await self.auth_get(downsample_url, token_key) as r:
            return await r.json()

    async def get_cutout(
        self,
        domain: str,
        collection: str,
        experiment: str,
        channel: str,
        resolution: int,
        range_box: Box3D,
        token_key: TokenKey,
    ) -> bytes:

        url = "/".join(
            [
                domain,
                "v1",
                "cutout",
                collection,
                experiment,
                channel,
                str(resolution),
                *(f"{left}:{right}" for left, right in zip(*range_box)),
            ]
        )
        async with await self.auth_get(url, token_key) as r:
            return await r.read()

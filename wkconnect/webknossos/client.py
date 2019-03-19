from typing import List, Union

from aiohttp import ClientSession
from sanic.config import Config

from ..utils.caching import atlru_cache
from ..utils.json import from_json, to_json
from .access import AccessAnswer, AccessRequest
from .models import DataSource, DataStoreStatus, UnusableDataSource


class WebKnossosClient:
    def __init__(self, config: Config, http_client: ClientSession) -> None:
        self.webknossos_url = config["webknossos"]["url"]
        self.datastore_url = config["server"]["url"]
        self.datastore_name = config["datastore"]["name"]
        self.datastore_key = config["datastore"]["key"]

        self.http_client = http_client

        self.headers = {"Content-Type": "application/json"}
        self.params = {"key": self.datastore_key}

    async def report_status(self, ok: bool = True) -> None:
        url = f"{self.webknossos_url}/api/datastores/{self.datastore_name}/status"
        status = DataStoreStatus(ok, self.datastore_url)
        await self.http_client.patch(
            url, headers=self.headers, params=self.params, json=to_json(status)
        )

    async def report_dataset(
        self, dataset: Union[DataSource, UnusableDataSource]
    ) -> None:
        url = f"{self.webknossos_url}/api/datastores/{self.datastore_name}/datasource"
        await self.http_client.put(
            url, headers=self.headers, params=self.params, json=to_json(dataset)
        )

    async def report_all_datasets(
        self, datasets: List[Union[DataSource, UnusableDataSource]]
    ) -> None:
        url = f"{self.webknossos_url}/api/datastores/{self.datastore_name}/datasources"
        await self.http_client.put(
            url, headers=self.headers, params=self.params, json=to_json(datasets)
        )

    @atlru_cache(seconds_to_use=2 * 60)
    async def request_access(
        self, token: str, access_request: AccessRequest
    ) -> AccessAnswer:
        url = f"{self.webknossos_url}/api/datastores/{self.datastore_name}/validateUserAccess"
        params = {"token": token, **self.params}
        async with await self.http_client.post(
            url,
            headers=self.headers,
            params=params,
            json=to_json(access_request),
            raise_for_status=False,
        ) as r:
            return from_json(await r.json(), AccessAnswer)

from ..utils.json import from_json, to_json
from .models import DataStoreStatus


class WebKnossosClient:
    def __init__(self, config, http_client):
        self.webknossos_url = config["webknossos"]["url"]
        self.datastore_url = config["server"]["url"]
        self.datastore_name = config["datastore"]["name"]
        self.datastore_key = config["datastore"]["key"]

        self.http_client = http_client

        self.headers = {"Content-Type": "application/json"}
        self.params = {"key": self.datastore_key}

    async def report_status(self, ok=True):
        url = f"{self.webknossos_url}/api/datastores/{self.datastore_name}/status"
        status = DataStoreStatus(ok, self.datastore_url)
        await self.http_client.patch(
            url, headers=self.headers, params=self.params, json=to_json(status)
        )

    async def report_dataset(self, dataset):
        url = f"{self.webknossos_url}/api/datastores/{self.datastore_name}/datasource"
        headers = {"Content-Type": "application/json"}
        await self.http_client.put(
            url, headers=self.headers, params=self.params, json=to_json(dataset)
        )

    async def request_access(self, token, access_request):
        pass
        # rpc(s"$webKnossosUrl/api/datastores/$dataStoreName/validateUserAccess")
        #  .addQueryString("key" -> dataStoreKey)
        #  .addQueryString("token" -> token)
        #  .postWithJsonResponse[UserAccessRequest, UserAccessAnswer](accessRequest)

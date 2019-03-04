import time
from typing import Dict, Tuple, Union

from aiohttp import ClientSession

from .models import Dataset


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
            assert token_info["token_type"] == "bearer"
            self.token_infos[key] = token_info

        return "Bearer " + token_info["access_token"]

    async def get_header(self, dataset: DatasetDescriptor) -> Dict[str, str]:
        return {"Authorization": await self.get(dataset)}

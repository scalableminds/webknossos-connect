from __future__ import annotations

import asyncio
import time
from typing import Dict, NamedTuple, Tuple, Union

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientResponseError

from ...utils.json import JSON
from .models import Dataset


class BossAuthenticationError(RuntimeError):
    pass


class TokenRepository:
    class Key(NamedTuple):
        domain: str
        username: str
        password: str

    DatasetDescriptor = Union[Dataset, Tuple[str, str, str], Key]

    def __init__(self, http_client: ClientSession):
        self.http_client = http_client
        self.token_infos: Dict[TokenRepository.Key, JSON] = {}

        self.client_id = "endpoint"
        self.login_realm = "boss"

    def __make_key(self, dataset: DatasetDescriptor) -> TokenRepository.Key:
        if isinstance(dataset, Dataset):
            return TokenRepository.Key(
                dataset.domain, dataset.username, dataset.password
            )
        else:
            return TokenRepository.Key(*dataset)

    def _openid_url(self, key: Key) -> str:
        # domain:   https://api.boss.neurodata.io
        # auth_url: https://auth.boss.neurodata.io/auth
        protocol = key.domain.split("://", 1)[0]
        domain_end = key.domain.split(".", 1)[1]
        auth_url = f"{protocol}://auth.{domain_end}/auth"

        return f"{auth_url}/realms/{self.login_realm}/protocol/openid-connect/"

    async def get(self, dataset: DatasetDescriptor) -> str:
        key = self.__make_key(dataset)

        request_new = True
        if key in self.token_infos:
            token_info = self.token_infos[key]
            if time.monotonic() - token_info["time"] < token_info["expires_in"] - 10:
                request_new = False

        if request_new:
            url = self._openid_url(key) + "token"
            data = {
                "grant_type": "password",
                "client_id": self.client_id,
                "username": key.username,
                "password": key.password,
            }
            now = time.monotonic()
            try:
                async with await self.http_client.post(url, data=data) as r:
                    token_info = await r.json()
            except ClientResponseError as e:
                if e.status == 401:  # Unauthorized
                    raise BossAuthenticationError(
                        f'Could not authorize user "{username}" at {auth_url}.'
                    )
                else:
                    raise e
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

    async def logout(self, dataset: DatasetDescriptor) -> None:
        key = self.__make_key(dataset)
        url = self._openid_url(key) + "logout"
        data = {
            "refresh_token": self.token_infos[key]["refresh_token"],
            "client_id": self.client_id,
        }
        try:
            await self.http_client.post(url, data=data)
            del self.token_infos[key]
        except ClientResponseError:
            pass

    async def logout_all(self) -> None:
        await asyncio.gather(
            *(self.logout(token_key) for token_key in self.token_infos)
        )

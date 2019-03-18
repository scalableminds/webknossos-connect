from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

from aiohttp import ClientSession
from aiohttp.client_exceptions import ClientResponseError

from ...utils.json import JSON


class BossAuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenKey:
    domain: str
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None

    def __post_init__(self) -> None:
        if self.token is None:
            assert self.username is not None
            assert self.password is not None
        else:
            assert self.username is None
            assert self.password is None


class TokenRepository:
    def __init__(self, http_client: ClientSession):
        self.http_client = http_client
        self.token_infos: Dict[TokenKey, JSON] = {}

        self.client_id = "endpoint"
        self.login_realm = "boss"

    def _openid_url(self, key: TokenKey) -> str:
        # domain:   https://api.boss.neurodata.io
        # auth_url: https://auth.boss.neurodata.io/auth
        protocol = key.domain.split("://", 1)[0]
        domain_end = key.domain.split(".", 1)[1]
        auth_url = f"{protocol}://auth.{domain_end}/auth"

        return f"{auth_url}/realms/{self.login_realm}/protocol/openid-connect/"

    async def get(self, key: TokenKey) -> str:
        if key.token is not None:
            return "Token " + key.token

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
                        f'Could not authorize user "{key.username}" at {key.domain}.'
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

    async def get_header(self, key: TokenKey) -> Dict[str, str]:
        return {"Authorization": await self.get(key)}

    async def logout(self, key: TokenKey) -> None:
        if key.token is not None:
            return
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

import aiohttp
from typing import Optional, List

class Token:
    def __init__(
        self,
        service_file: Optional[str] = None,
        session: aiohttp.ClientSession = None,
        scopes: List[str] = None,
    ) -> None: ...
    async def get(self) -> Optional[str]: ...

import aiohttp

class HttpClient:

  async def __aenter__(self):
    self.http_session = await aiohttp.ClientSession().__aenter__()
    return self


  async def __aexit__(self, exc_type, exc, tb):
    await self.http_session.__aexit__(exc_type, exc, tb)


  async def __perform_request(self, request_fn, *args, **kwargs):
    if 'response_fn' in kwargs:
      response_fn = kwargs['response_fn']
      del kwargs['response_fn']
    else:
      response_fn = None

    async with request_fn(*args, **kwargs) as response:
      if response.status is not 200:
        text = await response.text()
        raise RuntimeError((response.status, text))
      return await response_fn(response) if response_fn else None


  async def patch(self, *args, **kwargs):
    return await self.__perform_request(self.http_session.patch, *args, **kwargs)


  async def get(self, *args, **kwargs):
    return await self.__perform_request(self.http_session.get, *args, **kwargs)


  async def put(self, *args, **kwargs):
    return await self.__perform_request(self.http_session.put, *args, **kwargs)


  async def post(self, *args, **kwargs):
    return await self.__perform_request(self.http_session.post, *args, **kwargs)

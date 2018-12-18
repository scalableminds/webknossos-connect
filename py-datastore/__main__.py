import asyncio
import base64
import json
import numpy as np

from io import BytesIO
from PIL import Image
from sanic import Sanic
from sanic import response
from sanic_cors import cross_origin
from typing import List

from .backends.neuroglancer.backend import NeuroglancerBackend as Neuroglancer
from .repository import Repository
from .utils.http import HttpClient
from .utils.json import from_json, to_json
from .webknossos.client import WebKnossosClient as WebKnossos
from .webknossos.models import DataRequest as WKDataRequest


app = Sanic()

app.config.update({
  'server': {
    'host': '0.0.0.0',
    'port': 8000,
    'url': 'http://localhost:8000'
  },

  'datastore': {
    'name': 'py-datastore',
    'key': 'k'
  },

  'webknossos': {
    'url': 'http://docker.for.mac.host.internal:9000',
    'ping_interval_minutes': 10
  },

  'backends': {
    'neuroglancer': {}
  }
})

app.available_backends = [
  Neuroglancer
]


## TASKS ##

@app.listener('before_server_start')
async def setup(app, loop):

  def instanciate_backend(backend_class):
    backend_name = backend_class.name()
    config = app.config['backends'][backend_name]
    return (backend_name, backend_class(config, app.http_client))

  app.http_client = await HttpClient().__aenter__()
  app.repository = Repository()
  app.webknossos = WebKnossos(app.config, app.http_client)
  app.backends = dict(map(instanciate_backend, app.available_backends))


@app.listener('after_server_stop')
async def close_http_client(app, loop):
  await app.http_client.__aexit__(None, None, None)


async def ping_webknossos(app):
  ping_interval_seconds = app.config['webknossos']['ping_interval_minutes'] * 60
  while True:
    await app.webknossos.report_status()
    await asyncio.sleep(ping_interval_seconds)


## ROUTES ##


@app.route('/data/health')
async def health(request):
  return response.text('Ok')


@app.route('/api/buildinfo')
async def build_info(request):
  return response.json({
    'py-datastore': {
      'name': 'py-datastore',
      'version': '0.1',
      'datastoreApiVersion': '1.0'
    }
  })


@app.route('/api/<backend_name>/<organization_name>/<dataset_name>', methods=['POST'])
async def add_dataset(request, backend_name, organization_name, dataset_name):
  backend = app.backends[backend_name]
  dataset = await backend.handle_new_dataset(organization_name, dataset_name, request.json)
  app.repository.add_dataset(backend_name, dataset)
  await app.webknossos.report_dataset(dataset.to_webknossos())
  return response.text('Ok')


@app.route('/data/datasets/<organization_name>/<dataset_name>/layers/<layer_name>/data', methods=['OPTIONS'])
@cross_origin(app)
async def get_data(request, organization_name, dataset_name, layer_name):
  return response.text('Ok')


@app.route('/data/datasets/<organization_name>/<dataset_name>/layers/<layer_name>/data', methods=['POST'])
@cross_origin(app)
async def get_data(request, organization_name, dataset_name, layer_name):
  (backend_name, dataset) = app.repository.get_dataset(organization_name, dataset_name)
  backend = app.backends[backend_name]

  bucket_requests = from_json(request.json, List[WKDataRequest])
  assert all(not request.fourBit for request in bucket_requests)

  buckets = await asyncio.gather(*(backend.read_data(dataset, layer_name, r.zoomStep, r.position, (r.cubeSize, r.cubeSize, r.cubeSize)) for r in bucket_requests))
  missing_buckets = [ index for index, data in enumerate(buckets) if data is None ]
  existing_buckets = [ data.flatten(order='F') for data in buckets if data is not None ]
  data = np.concatenate(existing_buckets) if len(existing_buckets) > 0 else b''

  headers = { 'Access-Control-Expose-Headers': 'MISSING-BUCKETS', 'MISSING-BUCKETS': json.dumps(missing_buckets) }
  return response.raw(data, headers=headers)


@app.route('/data/datasets/<organization_name>/<dataset_name>/layers/<layer_name>/thumbnail.json')
async def get_thumbnail(request, organization_name, dataset_name, layer_name):
  width = int(request.args.get('width'))
  height = int(request.args.get('height'))

  (backend_name, dataset) = app.repository.get_dataset(organization_name, dataset_name)
  backend = app.backends[backend_name]
  data = await backend.read_data(dataset, layer_name, (30, 6, 6), (6400, 6400, 640), (width, height, 1))
  thumbnail = Image.fromarray(data[:,:,0])
  with BytesIO() as output:
    thumbnail.save(output, 'JPEG')
    return response.json({ 'mimeType': 'image/jpeg', 'value': base64.b64encode(output.getvalue()) })



## MAIN ##

if __name__ == '__main__':
  app.add_task(ping_webknossos(app))
  app.run(host=app.config['server']['host'], port=app.config['server']['port'], access_log=False)

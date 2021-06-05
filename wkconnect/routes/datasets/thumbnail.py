import base64
from io import BytesIO

from PIL import Image
from sanic import Blueprint, response
from sanic.request import Request

from ...utils.colors import color_bytes
from ...utils.types import Vec3D
from ...webknossos.access import AccessRequest, authorized

thumbnail = Blueprint(__name__)


@thumbnail.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/thumbnail.json"
)
@authorized(AccessRequest.read_dataset)
async def get_thumbnail(
    request: Request, organization_name: str, dataset_name: str, layer_name: str
) -> response.HTTPResponse:
    width = int(request.args.get("width"))
    height = int(request.args.get("height"))

    (backend_name, dataset) = request.app.ctx.repository.get_dataset(
        organization_name, dataset_name
    )
    backend = request.app.ctx.backends[backend_name]
    layer = [i for i in dataset.to_webknossos().dataLayers if i.name == layer_name][0]
    scale = min(3, len(layer.resolutions) - 1)
    center = layer.boundingBox.box().center()
    size = Vec3D(width, height, 1)
    data = (
        await backend.read_data(dataset, layer_name, scale, center - size // 2, size)
    )[:, :, 0]
    if layer.category == "segmentation":
        data = data.astype("uint8")
        thumbnail = Image.fromarray(data, mode="P")
        color_list = list(color_bytes.values())[: 2 ** 8]
        thumbnail.putpalette(b"".join(color_list))
        with BytesIO() as output:
            thumbnail.save(output, "PNG", transparency=0)
            return response.json(
                {"mimeType": "image/png", "value": base64.b64encode(output.getvalue())}
            )
    else:
        thumbnail = Image.fromarray(data)
        with BytesIO() as output:
            thumbnail.save(output, "JPEG")
            return response.json(
                {"mimeType": "image/jpeg", "value": base64.b64encode(output.getvalue())}
            )

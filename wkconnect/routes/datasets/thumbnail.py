from io import BytesIO

from PIL import Image
from sanic import Blueprint, response
from sanic.request import Request

from ...utils.colors import color_bytes
from ...utils.types import Vec3D
from ...webknossos.access import AccessRequest, authorized

thumbnail = Blueprint(__name__)


@thumbnail.route(
    "/<organization_name>/<dataset_name>/layers/<layer_name>/thumbnail.jpg"
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
    mag: Vec3D = sorted(layer.resolutions, key=lambda m: m.max_dim())[
        min(3, len(layer.resolutions) - 1)
    ]
    center = layer.boundingBox.box().center()
    size = Vec3D(width, height, 1)
    data = (
        await backend.read_data(dataset, layer_name, mag, center - size // 2, size)
    )[:, :, 0]
    if layer.category == "segmentation":
        data = data.astype("uint8")
        thumbnail = Image.fromarray(data.T, mode="P")
        color_list = list(color_bytes.values())[: 2 ** 8]
        thumbnail.putpalette(b"".join(color_list))
        with BytesIO() as output:
            thumbnail.save(output, "PNG", transparency=0)
            return response.raw(output.getvalue(), content_type="image/png")
    else:
        thumbnail = Image.fromarray(data.T)
        with BytesIO() as output:
            thumbnail.save(output, "JPEG")
            return response.raw(output.getvalue(), content_type="image/jpeg")

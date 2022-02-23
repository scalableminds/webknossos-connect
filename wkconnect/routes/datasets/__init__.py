from sanic import Blueprint

from .connectomes import connectomes
from .datasource_properties import datasource_properties
from .histogram import histogram
from .meshes import meshes
from .read_data import read_data
from .thumbnail import thumbnail
from .upload import upload

datasets = Blueprint.group(
    datasource_properties,
    read_data,
    thumbnail,
    histogram,
    upload,
    meshes,
    connectomes,
    url_prefix="/datasets",
)

from sanic import Blueprint

from .datasource_properties import datasource_properties
from .read_data import read_data
from .thumbnail import thumbnail

datasets = Blueprint.group(
    datasource_properties, read_data, thumbnail, url_prefix="/datasets"
)

from sanic import Blueprint

from .read_data import read_data
from .thumbnail import thumbnail

datasets = Blueprint.group(read_data, thumbnail, url_prefix="/datasets")

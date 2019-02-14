from sanic import Blueprint

from .read_data import read_data
from .thumbnail import thumbnail
from .upload import upload

datasets = Blueprint.group(read_data, thumbnail, upload, url_prefix="/datasets")

from sanic import Blueprint

from .datasets import datasets
from .triggers import triggers
from .trace import trace

data = Blueprint.group(datasets, triggers, url_prefix="/data")
routes = Blueprint.group(data, trace)

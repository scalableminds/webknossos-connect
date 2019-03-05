from sanic import Blueprint

from .datasets import datasets
from .trace import trace
from .triggers import triggers

data = Blueprint.group(datasets, triggers, url_prefix="/data")
routes = Blueprint.group(data, trace)

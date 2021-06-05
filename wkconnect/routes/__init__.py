from sanic import Blueprint
from sanic.blueprint_group import BlueprintGroup

from .datasets import datasets
from .trace import trace
from .triggers import triggers

data = Blueprint.group(datasets, triggers, url_prefix="/data")
routes: BlueprintGroup = Blueprint.group(data, trace)

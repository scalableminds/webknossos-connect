import inspect
import json

from typing import Dict, List


def from_json(data, cls):
  if hasattr(cls, '__annotations__'):
    for name, value in data.items():
      field_type = cls.__annotations__.get(name)
      data[name] = from_json(value, field_type)
    return cls(**data)
  elif hasattr(cls, '__origin__'):
    if issubclass(cls.__origin__, list):
      list_type = cls.__args__[0]
      return [ from_json(value, list_type) for value in data]
    elif issubclass(cls.__origin__, tuple):
      return tuple([
        from_json(value, element_type) for value, element_type in zip(data, cls.__args__)])
    elif issubclass(cls.__origin__, dict):
      key_type = cls.__args__[0]
      value_type = cls.__args__[1]
      return dict([
        (from_json(key, key_type), from_json(value, value_type)) for key, value in data.items()])
  else:
    return data


def to_json(obj):
  cls = type(obj)
  if hasattr(cls, '__annotations__'):
    return dict(map(lambda name: (name, to_json(obj.__dict__[name])), cls.__annotations__))
  elif issubclass(cls, list):
    return list(map(to_json, obj))
  elif issubclass(cls, tuple):
    return tuple(map(to_json, obj))
  elif issubclass(cls, dict):
    return dict([ (name, to_json(value)) for name, value in dict.items()])
  else:
    return obj

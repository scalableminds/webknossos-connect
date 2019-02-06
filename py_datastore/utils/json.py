import inspect
import json

from typing import Any, Dict, List, Optional

from .types import JSON


def from_json(data: JSON, cls: Optional[type]) -> Any:
    cls_annotations = getattr(cls, "__annotations__", None)
    cls_origin = getattr(cls, "__origin__", None)
    cls_args = getattr(cls, "__args__", None)
    if cls_annotations is not None:
        assert cls is not None
        if isinstance(data, dict):
            for name, value in data.items():
                field_type = cls_annotations.get(name)
                data[name] = from_json(value, field_type)
            return cls(**data)
        else:
            return cls(
                *(
                    from_json(field_data, field_type)
                    for field_data, field_type in zip(data, cls_annotations.values())
                )
            )
    elif isinstance(cls_origin, type) and cls_args is not None:
        if issubclass(cls_origin, list):
            list_type = cls_args[0]
            return [from_json(value, list_type) for value in data]
        elif issubclass(cls_origin, tuple):
            return tuple(
                from_json(value, element_type)
                for value, element_type in zip(data, cls_args)
            )
        elif issubclass(cls_origin, dict):
            key_type = cls_args[0]
            value_type = cls_args[1]
            return {
                from_json(key, key_type): from_json(value, value_type)
                for key, value in data.items()
            }
    else:
        return data


def to_json(obj: Any) -> JSON:
    cls = type(obj)
    if issubclass(cls, list):
        return list(map(to_json, obj))
    elif issubclass(cls, tuple):
        return tuple(map(to_json, obj))
    elif issubclass(cls, dict):
        return dict([(name, to_json(value)) for name, value in obj.items()])
    elif hasattr(cls, "__annotations__"):
        return dict(
            map(lambda name: (name, to_json(obj.__dict__[name])), cls.__annotations__)
        )
    else:
        return obj

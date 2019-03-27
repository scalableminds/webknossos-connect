from dataclasses import InitVar
from typing import Any, Iterable, Optional, Tuple, Union, get_type_hints

from .types import JSON, Vec3D, Vec3Df


def from_json(data: JSON, cls: Optional[type]) -> Any:
    cls_annotations = getattr(cls, "__annotations__", None)
    cls_origin = getattr(cls, "__origin__", None)
    cls_args = getattr(cls, "__args__", None)
    if cls_origin == Union:
        for unioned_cls in cls_args:
            try:
                return from_json(data, unioned_cls)
            except Exception:
                pass
        return data
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
            if cls_args[-1] == Ellipsis:
                tuple_type = cls_args[0]
                return tuple(from_json(value, tuple_type) for value in data)
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


def yield_jsons(
    obj_iter: Iterable, keys: Optional[Iterable[str]] = None
) -> Iterable[Union[JSON, Tuple[str, JSON]]]:
    if keys is None:
        for obj in obj_iter:
            json = to_json(obj)
            if json is not None:
                yield json
    else:
        for key, obj in zip(keys, obj_iter):
            json = to_json(obj)
            if json is not None:
                yield key, json


def to_json(obj: Any) -> JSON:
    cls = type(obj)
    if issubclass(cls, (Vec3D, Vec3Df)):
        return tuple(obj)
    elif hasattr(cls, "__annotations__"):
        annotations = get_type_hints(cls)
        annotations = {k: v for k, v in annotations.items() if v != InitVar}
        return dict(
            yield_jsons(
                (getattr(obj, name) for name in annotations), keys=annotations.keys()
            )
        )
    elif issubclass(cls, list):
        return list(yield_jsons(obj))
    elif issubclass(cls, tuple):
        return tuple(yield_jsons(obj))
    elif issubclass(cls, dict):
        return dict(yield_jsons(obj.values(), keys=obj.keys()))
    else:
        return obj

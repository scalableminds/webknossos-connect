from operator import add, sub, mul, truediv, floordiv
from typing import Any, Callable, NamedTuple


BaseVec3D = NamedTuple('Point', [('x', int), ('y', int), ('z', int)])


class Vec3D(BaseVec3D):
    def _element_wise(self, other: Any, fn: Callable[[int, Any], int]):
        if isinstance(other, tuple):
            return Vec3D(*(fn(a, b) for a, b in zip(self, other)))
        return Vec3D(*(fn(a, other) for a in self))

    def __add__(self, other: Any) -> "Vec3D":
        return self._element_wise(other, add)

    def __sub__(self, other: Any) -> "Vec3D":
        return self._element_wise(other, sub)

    def __mul__(self, other: Any) -> "Vec3D":
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Any) -> "Vec3D":
        return self._element_wise(other, floordiv)


# TODO refine this when recursive types are possible:
# see https://github.com/python/mypy/issues/731
JSON = Any

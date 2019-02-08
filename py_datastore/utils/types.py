from operator import add, floordiv, mul, sub
from typing import Any, Callable, NamedTuple, Union

import numpy as np

_BaseVec3D = NamedTuple("_BaseVec3D", [("x", int), ("y", int), ("z", int)])


class Vec3D(_BaseVec3D):
    def _element_wise(self, other: Any, fn: Callable[[int, Any], int]) -> "Vec3D":
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

    def ceildiv(self, other: Any) -> "Vec3D":
        return (self + other - 1) // other

    def pairmax(self, other: Any) -> "Vec3D":
        return self._element_wise(other, max)

    def pairmin(self, other: Any) -> "Vec3D":
        return self._element_wise(other, min)


_BaseBox3D = NamedTuple("_BaseBox3D", [("left", Vec3D), ("right", Vec3D)])


class Box3D(_BaseBox3D):
    """
    3-dimensional Box [left, right)
    """

    def size(self) -> Vec3D:
        return self.right - self.left

    def _element_wise(
        self, other: Union[Vec3D, int], fn: Callable[[Vec3D, Union[Vec3D, int]], Vec3D]
    ) -> "Box3D":
        return Box3D(*(fn(a, other) for a in self))

    def __add__(self, other: Any) -> "Box3D":
        return self._element_wise(other, add)

    def __sub__(self, other: Any) -> "Box3D":
        return self._element_wise(other, sub)

    def __mul__(self, other: Any) -> "Box3D":
        return self._element_wise(other, mul)

    def div(self, other: Any) -> "Box3D":
        return Box3D(self.left // other, self.right.ceildiv(other))

    def union(self, other: "Box3D") -> "Box3D":
        return Box3D(self.left.pairmin(other.left), self.right.pairmax(other.right))

    def intersect(self, other: "Box3D") -> "Box3D":
        return Box3D(self.left.pairmax(other.left), self.right.pairmin(other.right))

    def np_slice(self) -> np.lib.index_tricks.IndexExpression:
        return np.index_exp[
            self.left.x : self.right.x,
            self.left.y : self.right.y,
            self.left.z : self.right.z,
        ]


# TODO refine this when recursive types are possible:
# see https://github.com/python/mypy/issues/731
JSON = Any

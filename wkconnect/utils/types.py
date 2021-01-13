from __future__ import annotations

from operator import add, floordiv, mod, mul, sub, truediv
from typing import Any, Callable, Iterator, NamedTuple, Union

import numpy as np


class Vec3D(NamedTuple):
    x: int
    y: int
    z: int

    def _element_wise(self, other: Any, fn: Callable[[int, Any], int]) -> Vec3D:
        if isinstance(other, tuple):
            return Vec3D(*(fn(a, b) for a, b in zip(self, other)))
        return Vec3D(*(fn(a, other) for a in self))  # pylint: disable=not-an-iterable

    def __add__(self, other: Any) -> Vec3D:
        return self._element_wise(other, add)

    def __sub__(self, other: Any) -> Vec3D:
        return self._element_wise(other, sub)

    def __mul__(self, other: Any) -> Vec3D:
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Any) -> Vec3D:
        return self._element_wise(other, floordiv)

    def __mod__(self, other: Any) -> Vec3D:
        return self._element_wise(other, mod)

    def ceildiv(self, other: Any) -> Vec3D:
        return (self + other - 1) // other

    def pairmax(self, other: Any) -> Vec3D:
        return self._element_wise(other, max)

    def pairmin(self, other: Any) -> Vec3D:
        return self._element_wise(other, min)

    def to_float(self) -> Vec3Df:
        return Vec3Df(*map(float, self))

    def to_int(self) -> Vec3D:
        return Vec3D(*map(int, self))

    @classmethod
    def zeros(cls) -> Vec3D:
        return cls(0, 0, 0)

    @classmethod
    def ones(cls) -> Vec3D:
        return cls(1, 1, 1)


class Vec3Df(NamedTuple):
    x: float
    y: float
    z: float

    def _element_wise(self, other: Any, fn: Callable[[float, Any], float]) -> Vec3Df:
        if isinstance(other, tuple):
            return Vec3Df(*(fn(a, b) for a, b in zip(self, other)))
        return Vec3Df(*(fn(a, other) for a in self))  # pylint: disable=not-an-iterable

    def __add__(self, other: Any) -> Vec3Df:
        return self._element_wise(other, add)

    def __sub__(self, other: Any) -> Vec3Df:
        return self._element_wise(other, sub)

    def __mul__(self, other: Any) -> Vec3Df:
        return self._element_wise(other, mul)

    def __floordiv__(self, other: Any) -> Vec3Df:
        return self._element_wise(other, floordiv)

    def __truediv__(self, other: Any) -> Vec3Df:
        return self._element_wise(other, truediv)

    def __mod__(self, other: Any) -> Vec3Df:
        return self._element_wise(other, mod)

    def to_int(self) -> Vec3D:
        return Vec3D(*map(int, self))

    @classmethod
    def zeros(cls) -> Vec3Df:
        return cls(0.0, 0.0, 0.0)


class Box3D(NamedTuple):
    """
    3-dimensional Box [left, right)
    """

    left: Vec3D
    right: Vec3D

    @classmethod
    def from_size(cls, left: Vec3D, size: Vec3D) -> Box3D:
        return cls(left, left + size)

    def to_int(self) -> Box3D:
        return Box3D(self.left.to_int(), self.right.to_int())

    def size(self) -> Vec3D:
        return self.right - self.left

    def center(self) -> Vec3D:
        return self.left + self.size() // 2

    def _element_wise(
        self, other: Union[Vec3D, int], fn: Callable[[Vec3D, Union[Vec3D, int]], Vec3D]
    ) -> Box3D:
        return Box3D(*(fn(a, other) for a in self))  # pylint: disable=not-an-iterable

    def __add__(self, other: Any) -> Box3D:
        return self._element_wise(other, add)

    def __sub__(self, other: Any) -> Box3D:
        return self._element_wise(other, sub)

    def __mul__(self, other: Any) -> Box3D:
        return self._element_wise(other, mul)

    def div(self, other: Any) -> Box3D:
        return Box3D(self.left // other, self.right.ceildiv(other))

    def union(self, other: Box3D) -> Box3D:
        return Box3D(self.left.pairmin(other.left), self.right.pairmax(other.right))

    def intersect(self, other: Box3D) -> Box3D:
        return Box3D(self.left.pairmax(other.left), self.right.pairmin(other.right))

    def range(self, offset: Vec3D = Vec3D.ones()) -> Iterator[Vec3D]:
        for x in range(self.left.x, self.right.x, offset.x):
            for y in range(self.left.y, self.right.y, offset.y):
                for z in range(self.left.z, self.right.z, offset.z):
                    yield Vec3D(x, y, z)

    def np_slice(self) -> np.lib.index_tricks.IndexExpression:
        return np.index_exp[
            self.left.x : self.right.x,
            self.left.y : self.right.y,
            self.left.z : self.right.z,
        ]


# TODO refine this when recursive types are possible:
# see https://github.com/python/mypy/issues/731
JSON = Any


class HashableDict(dict):
    def __hash__(self) -> int:  # type: ignore  # dict returns None, we need int
        return hash(tuple(sorted(self.items())))

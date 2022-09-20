from abc import ABC, abstractmethod

from typing import TypeVar, Generic

from jetnet.config import Config


X = TypeVar("X")
Y = TypeVar("Y")


__all__ = ["Model"]


class Model(ABC, Generic[X, Y]):

    @abstractmethod
    def __call__(self, x: X) -> Y:
        raise NotImplementedError

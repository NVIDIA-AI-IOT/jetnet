from abc import ABC, abstractmethod

from typing import TypeVar, Generic

from pydantic.generics import GenericModel


X = TypeVar("X")
Y = TypeVar("Y")


__all__ = ["Model"]


class Model(GenericModel, Generic[X, Y]):
    
    def init(self):
        pass

    def build(self):
        clone = self.copy(deep=True)
        clone.init()
        return clone

    @abstractmethod
    def __call__(self, x: X) -> Y:
        raise NotImplementedError

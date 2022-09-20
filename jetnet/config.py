from typing import Generic, TypeVar
from pydantic.generics import GenericModel


__all__ = ["Config"]


T = TypeVar("T")


class Config(GenericModel, Generic[T]):

    def build(self) -> T:
        raise NotImplementedError
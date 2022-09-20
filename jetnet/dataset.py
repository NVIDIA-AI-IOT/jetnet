from abc import abstractmethod

from typing import TypeVar, Generic
from pydantic.generics import GenericModel

T = TypeVar("T")


__all__ = ["Dataset"]


class Dataset(GenericModel, Generic[T]):
    
    def init(self):
        pass

    def build(self):
        clone = self.copy(deep=True)
        clone.init()
        return clone
        
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        raise NotImplementedError

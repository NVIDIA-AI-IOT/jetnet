from abc import abstractmethod

from typing import TypeVar, Generic


T = TypeVar("T")


__all__ = ["Dataset"]


class Dataset(Generic[T]):

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> T:
        raise NotImplementedError

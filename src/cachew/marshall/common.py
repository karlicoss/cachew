from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Tuple,
    TypeVar,
    Union,
)

Json = Union[Dict[str, Any], Tuple[Any, ...], str, float, int, bool, None]


T = TypeVar('T')


class AbstractMarshall(Generic[T]):
    @abstractmethod
    def dump(self, obj: T) -> Json:
        raise NotImplementedError

    @abstractmethod
    def load(self, dct: Json) -> T:
        raise NotImplementedError

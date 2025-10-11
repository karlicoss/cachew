from abc import abstractmethod
from typing import Any

type Json = dict[str, Any] | tuple[Any, ...] | str | float | int | bool | None


class AbstractMarshall[T]:
    @abstractmethod
    def dump(self, obj: T) -> Json:
        raise NotImplementedError

    @abstractmethod
    def load(self, dct: Json) -> T:
        raise NotImplementedError

from dataclasses import dataclass
from typing import Type


class CachewException(RuntimeError):
    pass


@dataclass
class TypeNotSupported(CachewException):
    type_: Type

    def __str__(self) -> str:
        return f"{self.type_} isn't supported by cachew. See https://github.com/karlicoss/cachew#features for the list of supported types."


# https://stackoverflow.com/a/2166841/706389
def is_namedtuple(t) -> bool:
    b = getattr(t, '__bases__', None)
    if b is None:
        return False
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    # pylint: disable=unidiomatic-typecheck
    return all(type(n) == str for n in f)  # noqa: E721

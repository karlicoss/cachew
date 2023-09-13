from dataclasses import dataclass
from datetime import date, datetime
from typing import (
    NamedTuple,
    Optional,
    Union,
    Tuple,
    Type,
)


def get_union_args(cls) -> Optional[Tuple[Type]]:
    if getattr(cls, '__origin__', None) != Union:
        return None

    args = cls.__args__
    args = [e for e in args if e != type(None)]
    assert len(args) > 0
    return args


def is_union(cls) -> bool:
    return get_union_args(cls) is not None


class CachewException(RuntimeError):
    pass


@dataclass
class TypeNotSupported(CachewException):
    type_: Type

    def __str__(self) -> str:
        return f"{self.type_} isn't supported by cachew. See https://github.com/karlicoss/cachew#features for the list of supported types."


Types = Union[
    Type[str],
    Type[int],
    Type[float],
    Type[bool],
    Type[datetime],
    Type[date],
    Type[dict],
    Type[list],
    Type[Exception],
    Type[NamedTuple],
]

Values = Union[
    str,
    int,
    float,
    bool,
    datetime,
    date,
    dict,
    list,
    Exception,
    NamedTuple,
]
# TODO assert all PRIMITIVES are also in Types/Values?


PRIMITIVE_TYPES = {
    str,
    int,
    float,
    bool,
    datetime,
    date,
    dict,
    list,
    Exception,
}


def is_primitive(cls: Type) -> bool:
    """
    >>> from typing import Dict, Any
    >>> is_primitive(int)
    True
    >>> is_primitive(set)
    False
    >>> is_primitive(dict)
    True
    """
    return cls in PRIMITIVE_TYPES


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

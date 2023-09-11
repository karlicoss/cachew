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

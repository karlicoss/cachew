import typing
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import chain, islice
from pathlib import Path
from typing import (
    Any,
    Generic,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)

import sqlalchemy
from sqlalchemy import Column

from .pytest import parametrize
from .utils import CachewException


def get_union_args(cls) -> Optional[tuple[type]]:
    if getattr(cls, '__origin__', None) != Union:
        return None

    args = cls.__args__
    args = tuple(e for e in args if e is not type(None))
    assert len(args) > 0
    return args


def is_union(cls) -> bool:
    return get_union_args(cls) is not None


Types = Union[
    type[str],
    type[int],
    type[float],
    type[bool],
    type[datetime],
    type[date],
    type[dict],
    type[list],
    type[Exception],
    type[NamedTuple],
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


def is_primitive(cls: type) -> bool:
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


class IsoDateTime(sqlalchemy.TypeDecorator):
    # in theory could use something more effecient? e.g. blob for encoded datetime and tz?
    # but practically, the difference seems to be pretty small, so perhaps fine for now
    impl = sqlalchemy.String

    cache_ok = True

    @property
    def python_type(self):
        return datetime

    def process_literal_param(self, value, dialect):
        raise NotImplementedError()  # make pylint happy

    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[str]:  # noqa: ARG002
        if value is None:
            return None
        # ok, it's a bit hacky... attempt to preserve pytz infromation
        iso = value.isoformat()
        tz = getattr(value, 'tzinfo', None)
        if tz is None:
            return iso
        try:
            import pytz
        except ImportError:
            self.warn_pytz()
            return iso
        else:
            if isinstance(tz, pytz.BaseTzInfo):
                zone = tz.zone
                # should be present: https://github.com/python/typeshed/blame/968fd6d01d23470e0c8368e7ee7c43f54aaedc0e/stubs/pytz/pytz/tzinfo.pyi#L6
                assert zone is not None, tz
                return iso + ' ' + zone
            else:
                return iso

    def process_result_value(self, value: Optional[str], dialect) -> Optional[datetime]:  # noqa: ARG002
        if value is None:
            return None
        spl = value.split(' ')
        dt = datetime.fromisoformat(spl[0])
        if len(spl) <= 1:
            return dt
        zone = spl[1]
        # else attempt to decypher pytz tzinfo
        try:
            import pytz
        except ImportError:
            self.warn_pytz()
            return dt
        else:
            tz = pytz.timezone(zone)
            return dt.astimezone(tz)

    def warn_pytz(self) -> None:
        warnings.warn('install pytz for better timezone support while serializing with cachew', stacklevel=2)


# a bit hacky, but works...
class IsoDate(IsoDateTime):
    impl = sqlalchemy.String

    cache_ok = True

    @property
    def python_type(self):
        return date

    def process_literal_param(self, value, dialect):
        raise NotImplementedError()  # make pylint happy

    def process_result_value(self, value: Optional[str], dialect) -> Optional[date]:  # type: ignore[override]
        res = super().process_result_value(value, dialect)
        if res is None:
            return None
        return res.date()


jtypes = (int, float, bool, type(None))


class ExceptionAdapter(sqlalchemy.TypeDecorator):
    '''
    Enables support for caching Exceptions. Exception is treated as JSON and serialized.

    It's useful for defensive error handling, in case of cachew in particular for preserving error state.

    I elaborate on it here: [mypy-driven error handling](https://beepb00p.xyz/mypy-error-handling.html#kiss).
    '''

    impl = sqlalchemy.JSON

    cache_ok = True

    @property
    def python_type(self):
        return Exception

    def process_literal_param(self, value, dialect):
        raise NotImplementedError()  # make pylint happy

    def process_bind_param(self, value: Optional[Exception], dialect) -> Optional[list[Any]]:  # noqa: ARG002
        if value is None:
            return None
        sargs: list[Any] = []
        for a in value.args:
            if any(isinstance(a, t) for t in jtypes):
                sargs.append(a)
            elif isinstance(a, date):
                sargs.append(a.isoformat())
            else:
                sargs.append(str(a))
        return sargs

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Exception]:  # noqa: ARG002
        if value is None:
            return None
        # sadly, can't do much to convert back from the strings? Unless I serialize the type info as well?
        return Exception(*value)


# fmt: off
PRIMITIVES = {
    str      : sqlalchemy.String,
    int      : sqlalchemy.Integer,
    float    : sqlalchemy.Float,
    bool     : sqlalchemy.Boolean,
    datetime : IsoDateTime,
    date     : IsoDate,
    dict     : sqlalchemy.JSON,
    list     : sqlalchemy.JSON,
    Exception: ExceptionAdapter,
}
# fmt: on
assert set(PRIMITIVES.keys()) == PRIMITIVE_TYPES


def strip_optional(cls) -> tuple[type, bool]:
    """
    >>> from typing import Optional, NamedTuple
    >>> strip_optional(Optional[int])
    (<class 'int'>, True)
    >>> class X(NamedTuple):
    ...     x: int
    >>> strip_optional(X)
    (<class 'cachew.legacy.X'>, False)
    """
    is_opt: bool = False

    args = get_union_args(cls)
    if args is not None and len(args) == 1:
        cls = args[0]  # meh
        is_opt = True

    return (cls, is_opt)


def strip_generic(tp):
    """
    >>> from typing import List
    >>> strip_generic(List[int])
    <class 'list'>
    >>> strip_generic(str)
    <class 'str'>
    """
    GA = getattr(typing, '_GenericAlias')  # ugh, can't make both mypy and pylint happy here?
    if isinstance(tp, GA):
        return tp.__origin__
    return tp


NT = TypeVar('NT')
# sadly, bound=NamedTuple is not working yet in mypy
# https://github.com/python/mypy/issues/685
# also needs to support dataclasses?


@dataclass
class NTBinder(Generic[NT]):
    """
    >>> class Job(NamedTuple):
    ...    company: str
    ...    title: Optional[str]
    >>> class Person(NamedTuple):
    ...     name: str
    ...     age: int
    ...     job: Optional[Job]

    NTBinder is a helper class for inteacting with sqlite database.
    Hierarchy is flattened:
    >>> binder = NTBinder.make(Person)
    >>> [(c.name, type(c.type)) for c in binder.columns]
    ... # doctest: +NORMALIZE_WHITESPACE
    [('name',         <class 'sqlalchemy.sql.sqltypes.String'>),
     ('age',          <class 'sqlalchemy.sql.sqltypes.Integer'>),
     ('_job_is_null', <class 'sqlalchemy.sql.sqltypes.Boolean'>),
     ('job_company',  <class 'sqlalchemy.sql.sqltypes.String'>),
     ('job_title',    <class 'sqlalchemy.sql.sqltypes.String'>)]


    >>> person = Person(name='alan', age=40, job=None)

    to_row converts object to a sql-friendly tuple. job=None, so we end up with True in _job_is_null field
    >>> tuple(binder.to_row(person))
    ('alan', 40, True, None, None)

    from_row does reverse conversion
    >>> binder.from_row(('alan', 40, True, None, None))
    Person(name='alan', age=40, job=None)

    >>> binder.from_row(('ann', 25, True, None, None, 'extra'))
    Traceback (most recent call last):
    ...
    cachew.utils.CachewException: unconsumed items in iterator ['extra']
    """

    name: Optional[str]  # None means toplevel
    type_: Types
    span: int  # not sure if span should include optional col?
    primitive: bool
    optional: bool
    union: Optional[type]  # helper, which isn't None if type is Union
    fields: Sequence[Any]  # mypy can't handle cyclic definition at this point :(

    @staticmethod
    def make(tp: type[NT], name: Optional[str] = None) -> 'NTBinder[NT]':
        tp, optional = strip_optional(tp)
        union: Optional[type]
        fields: tuple[Any, ...]
        primitive: bool

        union_args = get_union_args(tp)
        if union_args is not None:
            CachewUnion = NamedTuple('_CachewUnionRepr', [(x.__name__, Optional[x]) for x in union_args])  # type: ignore[misc]
            union = CachewUnion
            primitive = False
            fields = (NTBinder.make(tp=CachewUnion, name='_cachew_union_repr'),)
            span = 1
        else:
            union = None
            tp = strip_generic(tp)
            primitive = is_primitive(tp)

            if primitive:
                if name is None:
                    name = '_cachew_primitive'  # meh. presumably, top level
            if primitive:
                fields = ()
                span = 1
            else:
                annotations = typing.get_type_hints(tp)
                if annotations == {}:
                    raise CachewException(
                        f"{tp} (field '{name}'): doesn't look like a supported type to cache. See https://github.com/karlicoss/cachew#features for the list of supported types."
                    )
                fields = tuple(NTBinder.make(tp=ann, name=fname) for fname, ann in annotations.items())
                span = sum(f.span for f in fields) + (1 if optional else 0)
        return NTBinder(
            name=name,
            type_=tp,  # type: ignore[arg-type]
            span=span,
            primitive=primitive,
            optional=optional,
            union=union,
            fields=fields,
        )

    @property
    def columns(self) -> list[Column]:
        return list(self.iter_columns())

    # TODO not necessarily namedtuple? could be primitive type
    def to_row(self, obj: NT) -> tuple[Optional[Values], ...]:
        return tuple(self._to_row(obj))

    def from_row(self, row: Iterable[Any]) -> NT:
        riter = iter(row)
        res = self._from_row(riter)
        remaining = list(islice(riter, 0, 1))
        if len(remaining) != 0:
            raise CachewException(f'unconsumed items in iterator {remaining}')
        assert res is not None  # nosec # help mypy; top level will not be None
        return res

    def _to_row(self, obj) -> Iterator[Optional[Values]]:
        if self.primitive:
            yield obj
        elif self.union is not None:
            CachewUnion = self.union
            (uf,) = self.fields
            # TODO assert only one of them matches??
            union = CachewUnion(**{f.name: obj if isinstance(obj, f.type_) else None for f in uf.fields})
            yield from uf._to_row(union)
        else:
            if self.optional:
                is_none = obj is None
                yield is_none
            else:
                is_none = False
                assert obj is not None  # TODO hmm, that last assert is not very symmetric...

            if is_none:
                for _ in range(self.span - 1):
                    yield None
            else:
                yield from chain.from_iterable(f._to_row(getattr(obj, f.name)) for f in self.fields)

    def _from_row(self, row_iter):
        if self.primitive:
            return next(row_iter)
        elif self.union is not None:
            CachewUnion = self.union  # noqa: F841
            (uf,) = self.fields
            # TODO assert only one of them is not None?
            union_params = [r for r in uf._from_row(row_iter) if r is not None]
            assert len(union_params) == 1, union_params
            return union_params[0]
        else:
            if self.optional:
                is_none = next(row_iter)
            else:
                is_none = False

            if is_none:
                for _ in range(self.span - 1):
                    x = next(row_iter)
                    assert x is None, x  # huh. assert is kinda opposite of producing value
                return None
            else:
                return self.type_(*(f._from_row(row_iter) for f in self.fields))

    # TODO not sure if we want to allow optionals on top level?
    def iter_columns(self) -> Iterator[Column]:
        used_names: set[str] = set()

        def col(name: str, tp) -> Column:
            while name in used_names:
                name = '_' + name
            used_names.add(name)
            return Column(name, tp)

        if self.primitive:
            if self.name is None:
                raise AssertionError
            yield col(self.name, PRIMITIVES[self.type_])
        else:
            prefix = '' if self.name is None else self.name + '_'
            if self.optional:
                yield col(f'_{prefix}is_null', sqlalchemy.Boolean)
            for f in self.fields:
                for c in f.iter_columns():
                    yield col(f'{prefix}{c.name}', c.type)

    def __str__(self):
        lines = ['  ' * level + str(x.name) + ('?' if x.optional else '') + f' <span {x.span}>' for level, x in self.flatten()]
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)

    def flatten(self, level=0):
        yield (level, self)
        for f in self.fields:
            yield from f.flatten(level=level + 1)


def test_mypy_annotations() -> None:
    # mypy won't handle, so this has to be dynamic
    vs = []
    for t in Types.__args__:  # type: ignore[attr-defined]
        (arg,) = t.__args__
        vs.append(arg)

    def types(ts):
        return sorted(ts, key=lambda t: str(t))

    assert types(vs) == types(Values.__args__)  # type: ignore[attr-defined]

    for p in PRIMITIVE_TYPES:
        assert p in Values.__args__  # type: ignore[attr-defined]


@parametrize(
    ('tp', 'val'),
    [
        (int, 22),
        (bool, False),
        (Optional[str], 'abacaba'),
        (Union[str, int], 1),
    ],
)
def test_ntbinder_primitive(tp, val) -> None:
    b = NTBinder.make(tp, name='x')
    row = b.to_row(val)
    vv = b.from_row(list(row))
    assert vv == val


def test_unique_columns(tmp_path: Path) -> None:  # noqa: ARG001
    class Job(NamedTuple):
        company: str
        title: Optional[str]

    class Breaky(NamedTuple):
        job_title: int
        job: Optional[Job]

    assert [c.name for c in NTBinder.make(Breaky).columns] == [
        'job_title',
        '_job_is_null',
        'job_company',
        '_job_title',
    ]

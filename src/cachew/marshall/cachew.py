from __future__ import annotations

import types
from abc import abstractmethod
from collections import abc
from collections.abc import Sequence
from dataclasses import dataclass, is_dataclass
from datetime import date, datetime, timezone
from numbers import Real
from typing import (  # noqa: UP035
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from zoneinfo import ZoneInfo

from ..utils import TypeNotSupported, is_namedtuple
from .common import (
    AbstractMarshall,
    Json,
    T,
)


class CachewMarshall(AbstractMarshall[T]):
    def __init__(self, Type_: type[T]) -> None:
        self.schema = build_schema(Type_)

    def dump(self, obj: T) -> Json:
        return self.schema.dump(obj)

    def load(self, dct: Json) -> T:
        return self.schema.load(dct)


# TODO add generic types later?


# NOTE: using slots gives a small speedup (maybe 5%?)
# I suppose faster access to fields or something..


@dataclass(slots=True)
class Schema:
    type: Any

    @abstractmethod
    def dump(self, obj):
        raise NotImplementedError

    @abstractmethod
    def load(self, dct):
        raise NotImplementedError


@dataclass(slots=True)
class SPrimitive(Schema):
    def dump(self, obj):
        # NOTE: returning here directly (instead of calling identity lambda) gives about 20% speedup
        # I think custom types should have their own Schema subclass
        return obj
        # prim = primitives_to.get(self.type)
        # assert prim is not None
        # return prim(o)

    def load(self, dct):
        return dct
        # prim = primitives_from.get(self.type)
        # assert prim is not None
        # return prim(d)


@dataclass(slots=True)
class SDataclass(Schema):
    # using list of tuples instead of dict gives about 5% speedup
    fields: tuple[tuple[str, Schema], ...]

    def dump(self, obj):
        # TODO would be nice if we didn't create a dictionary here
        # considering it is going to be serialized to json anyway
        # maybe we need to yield json bits actually?
        return {
            # would be kinda nice if we didn't have to use getattr here
            # but I think for dataclass this is actually the fastest way
            # TODO for NamedTuples could just use them as tuples.. think about separating
            k: ks.dump(getattr(obj, k))
            for k, ks in self.fields
        }

    def load(self, dct):
        # dict comprehension is meh, but not sure if there is a faster way?
        return self.type(**{
            k: ks.load(dct[k])
            for k, ks in self.fields
        })  # fmt: skip


@dataclass(slots=True)
class SUnion(Schema):
    # it's a bit faster to cache indices here, gives about 15% speedup
    args: tuple[tuple[int, Schema], ...]

    def dump(self, obj):
        if obj is None:
            # if it's a None, then doesn't really matter how to serialize and deserialize it
            return (0, None)

        # TODO could do a bit of magic here and remember the last index that worked?
        # that way if some objects dominate the Union, the first isinstance would always work
        for tidx, a in self.args:
            if isinstance(obj, a.type):  # this takes quite a lot of time (sort of expected?)
                # using lists instead of dicts gives a bit of a speedup (about 15%)
                # so probably worth it even though a bit cryptic
                # also could add a tag or something?
                # NOTE: using tuple instead of list gives a tiiny speedup
                jj = a.dump(obj)
                return (tidx, jj)
                # {
                #     '__union_index__': tidx,
                #     '__value__': jj,
                # }
        raise RuntimeError(f"shouldn't happen: {self.args} {obj}")

    def load(self, dct):
        # tidx = d['__union_index__']
        # s = self.args[tidx]
        # return s.load(d['__value__'])
        tidx, val = dct
        if val is None:
            # counterpart for None handling in .dump method
            return None

        _, s = self.args[tidx]
        return s.load(val)


@dataclass(slots=True)
class SList(Schema):
    arg: Schema

    def dump(self, obj):
        return tuple(self.arg.dump(i) for i in obj)

    def load(self, dct):
        return [self.arg.load(i) for i in dct]


@dataclass(slots=True)
class STuple(Schema):
    args: tuple[Schema, ...]

    def dump(self, obj):
        return tuple(a.dump(i) for a, i in zip(self.args, obj, strict=True))

    def load(self, dct):
        return tuple(a.load(i) for a, i in zip(self.args, dct, strict=True))


@dataclass(slots=True)
class SSequence(Schema):
    arg: Schema

    def dump(self, obj):
        return tuple(self.arg.dump(i) for i in obj)

    def load(self, dct):
        return tuple(self.arg.load(i) for i in dct)


@dataclass(slots=True)
class SDict(Schema):
    ft: SPrimitive
    tt: Schema

    def dump(self, obj):
        return {
            k: self.tt.dump(v)
            for k, v in obj.items()
        }  # fmt: skip

    def load(self, dct):
        return {
            k: self.tt.load(v)
            for k, v in dct.items()
        }  # fmt: skip


# TODO unify with primitives?
JTypes = {int, str, type(None), float, bool}


def _exc_helper(args):
    for a in args:
        at = type(a)
        if at in JTypes:
            yield a
        elif issubclass(at, date):
            # TODO would be nice to restore datetime from cache too
            # maybe generally save exception as a union? or intact and let orjson save it?
            yield a.isoformat()
        else:
            yield str(a)  # not much we can do..


@dataclass(slots=True)
class SException(Schema):
    def dump(self, obj: Exception) -> Json:
        return tuple(_exc_helper(obj.args))

    def load(self, dct: Json):
        return self.type(*dct)


try:
    # defensive to avoid dependency on pytz when we switch to python >= 3.9
    import pytz
except ModuleNotFoundError:
    # dummy, this is only needed for isinstance check below
    class pytz_BaseTzInfo:
        zone: str

    def make_tz_pytz(zone: str):
        raise RuntimeError(f"Install pytz to deserialize {zone}")

else:
    pytz_BaseTzInfo = pytz.BaseTzInfo  # type: ignore[misc,assignment]

    make_tz_pytz = pytz.timezone


# just ints to avoid inflating db size
# for now, we try to preserve actual timezone object just in case since they do have somewhat incompatible apis
_TZTAG_ZONEINFO = 1
_TZTAG_PYTZ = 2


@dataclass(slots=True)
class SDatetime(Schema):
    def dump(self, obj: datetime) -> Json:
        iso = obj.isoformat()
        tz = obj.tzinfo
        if tz is None:
            return (iso, None, None)

        if isinstance(tz, ZoneInfo):
            return (iso, tz.key, _TZTAG_ZONEINFO)
        elif isinstance(tz, pytz_BaseTzInfo):
            zone = tz.zone
            # should be present: https://github.com/python/typeshed/blame/968fd6d01d23470e0c8368e7ee7c43f54aaedc0e/stubs/pytz/pytz/tzinfo.pyi#L6
            assert zone is not None, (obj, tz)
            return (iso, zone, _TZTAG_PYTZ)
        else:
            return (iso, None, None)

    def load(self, dct: tuple):
        iso, zone, zone_tag = dct
        dt = datetime.fromisoformat(iso)
        if zone is None:
            return dt

        make_tz = ZoneInfo if zone_tag == _TZTAG_ZONEINFO else make_tz_pytz
        tz = make_tz(zone)
        return dt.astimezone(tz)


@dataclass(slots=True)
class SDate(Schema):
    def dump(self, obj: date) -> Json:
        return obj.isoformat()

    def load(self, dct: str):
        return date.fromisoformat(dct)


PRIMITIVES = {
    # int and float are handled a bit differently to allow implicit casts
    # isinstance(.., Real) works both for int and for float
    # Real can't be serialized back, but if you look in SPrimitive, it leaves the values intact anyway
    # since the actual serialization of primitives is handled by orjson
    int: Real,
    float: Real,
    str: str,
    type(None): type(None),
    bool: bool,
    # if type is Any, there isn't much we can do to dump it -- just dump into json and rely on the best
    # so in this sense it works exacly like primitives
    Any: Any,
}


def build_schema(Type) -> Schema:
    assert not isinstance(Type, str), Type  # just to avoid confusion in case of weirdness with stringish type annotations

    ptype = PRIMITIVES.get(Type)
    if ptype is not None:
        return SPrimitive(type=ptype)

    origin = get_origin(Type)
    # origin is 'unsubscripted/erased' version of type
    # if origin is NOT None, it's some sort of generic type

    if origin is None:
        if issubclass(Type, Exception):
            return SException(type=Type)

        if issubclass(Type, datetime):
            return SDatetime(type=Type)

        if issubclass(Type, date):
            return SDate(type=Type)

        if not (is_dataclass(Type) or is_namedtuple(Type)):
            raise TypeNotSupported(type_=Type, reason='unknown type')
        try:
            hints = get_type_hints(Type)
        except TypeError as te:
            # this can happen for instance on 3.9 if pipe syntax was used for Union types
            # would be nice to provide a friendlier error though
            raise TypeNotSupported(type_=Type, reason='failed to get type hints') from te
        fields = tuple((k, build_schema(t)) for k, t in hints.items())
        return SDataclass(
            type=Type,
            fields=fields,
        )

    args = get_args(Type)
    is_union = origin is Union or origin is types.UnionType

    if is_union:
        # We 'erasing' types (since generic types don't work with isinstance checks).
        # So we need to make sure the types are unique to make sure we can deserialise them.
        schemas = [build_schema(a) for a in args]
        union_types = [s.type for s in schemas if s.type is not Real]
        if len(set(union_types)) != len(union_types):
            raise TypeNotSupported(type_=Type, reason=f'runtime union arguments are not unique: {union_types}')
        return SUnion(
            type=origin,
            args=tuple(
                (tidx, s)
                for tidx, s in enumerate(schemas)
            ),
        )  # fmt: skip

    is_listish = origin is list
    if is_listish:
        (t,) = args
        return SList(
            type=origin,
            arg=build_schema(t),
        )

    # hmm check for is typing.Sequence doesn't pass for some reason
    # perhaps because it's a deprecated alias?
    is_tuplish = origin is tuple or origin is abc.Sequence
    if is_tuplish:
        if origin is tuple:
            # this is for Tuple[()], which is the way to represent empty tuple
            # before python 3.11, get_args for that gives ((),) instead of an empty tuple () as one might expect
            if args == ((),):
                args = ()
            return STuple(
                type=origin,
                args=tuple(build_schema(a) for a in args),
            )
        else:
            (t,) = args
            return SSequence(
                type=origin,
                arg=build_schema(t),
            )

    is_dictish = origin is dict
    if is_dictish:
        (ft, tt) = args
        fts = build_schema(ft)
        tts = build_schema(tt)
        assert isinstance(fts, SPrimitive)
        return SDict(
            type=origin,
            ft=fts,
            tt=tts,
        )

    raise RuntimeError(f"unsupported: {Type=} {origin=} {args=}")


######### tests


def _test_identity(obj, Type_, expected=None):
    if expected is None:
        expected = obj

    m = CachewMarshall(Type_)

    j = m.dump(obj)
    obj2 = m.load(j)

    # Exception's don't support equality normally, so we need to do some hacks..
    def normalise(x):
        if isinstance(x, Exception):
            return (type(x), x.args)
        if type(x) is list:
            return [(type(i), i.args) if isinstance(i, Exception) else i for i in x]
        return x

    # ugh that doesn't work
    # def exc_eq(s, other):
    #     return (type(s), s.args) == (type(other), other.args)
    # Exception.__eq__ = exc_eq

    assert normalise(expected) == normalise(obj2), (expected, obj2)
    return (j, obj2)


# TODO customise with cattrs
def test_serialize_and_deserialize() -> None:
    import pytest

    helper = _test_identity

    # primitives
    helper(1, int)
    helper('aaa', str)
    helper(None, type(None))
    # TODO emit other value as none type? not sure what should happen

    # implicit casts, simple version
    helper(None, int)
    helper(None, str)
    helper(1, float)

    # implicit casts, inside other types
    # technically not type safe, but might happen in practice
    # doesn't matter how to deserialize None anyway so let's allow this
    helper(None, str | int)
    # old syntax
    helper(None, Union[str, int])  # noqa: UP007

    # even though 1 is not isinstance(float), often it ends up as float in data
    # see https://github.com/karlicoss/cachew/issues/54
    helper(1, float | str)
    helper(2, float | int)
    helper(2.0, float | int)
    helper((1, 2), tuple[int, float])

    # optionals
    helper('aaa', str | None)
    helper(None, str | None)
    # old syntax
    helper('aaa', Optional[str])  # noqa: UP045
    helper('aaa', Union[str, None])  # noqa: UP007
    helper(None, Union[str, None])  # noqa: UP007

    # lists/tuples/sequences
    # TODO test with from __future__ import annotations..
    helper([1, 2, 3], list[int])
    helper([1, 2, 3], Optional[List[int]])  # noqa: UP006,UP045
    helper([1, 2, 3], Sequence[int], expected=(1, 2, 3))
    helper((1, 2, 3), Sequence[int])
    helper((1, 2, 3), tuple[int, int, int])
    # old syntax
    helper([1, 2, 3], List[int])  # noqa: UP006
    helper((1, 2, 3), Tuple[int, int, int])  # noqa: UP006
    helper((1, 2, 3), Optional[tuple[int, int, int]])  # noqa: UP045

    # dicts
    helper({'a': 'aa', 'b': 'bb'}, dict[str, str])
    helper({'a': None, 'b': 'bb'}, dict[str, str | None])
    helper({'a': 'aa', 'b': 'bb'}, dict[str, str])
    # old syntax
    helper({'a': None, 'b': 'bb'}, Dict[str, Optional[str]])  # noqa: UP006,UP045

    # unions
    helper('aaa', str | int)
    # old syntax
    helper(1, Union[str, int])  # noqa: UP007

    # compounds of simple types
    helper(['1', 2, '3'], list[str | int])
    # old syntax
    helper(['1', 2, '3'], list[Union[str, int]])  # noqa: UP007

    # TODO need to add test for equivalent dataclasses

    @dataclass
    class Point:
        x: int
        y: int

    # dataclasses
    helper(Point(x=1, y=2), Point)

    # Namedtuple
    class NT(NamedTuple):
        first: str
        last: str

    helper(NT(first='aaa', last='bbb'), NT)

    @dataclass
    class WithJson:
        id: int
        raw_data: dict[str, Any]

    # json-ish stuff
    helper({}, dict[str, Any])
    helper(WithJson(id=123, raw_data={'payload': 'whatever', 'tags': ['a', 'b', 'c']}), WithJson)
    helper([], list[Any])

    # exceptions
    helper(RuntimeError('whatever!'), RuntimeError)
    # fmt: off
    helper([
        RuntimeError('I', 'am', 'exception', 123),
        Point(x=1, y=2),
        Point(x=11, y=22),
        RuntimeError('more stuff'),
        RuntimeError(),
    ], list[RuntimeError | Point])

    exc_with_datetime     = Exception('I happenned on', datetime.fromisoformat('2021-04-03T10:11:12'))
    exc_with_datetime_exp = Exception('I happenned on', '2021-04-03T10:11:12')
    helper(exc_with_datetime, Exception, expected=exc_with_datetime_exp)
    # fmt: on

    # datetimes
    import pytz

    tz_london = pytz.timezone('Europe/London')
    dwinter = datetime.strptime('20200203 01:02:03', '%Y%m%d %H:%M:%S')
    dsummer = datetime.strptime('20200803 01:02:03', '%Y%m%d %H:%M:%S')
    dwinter_tz = tz_london.localize(dwinter)
    dsummer_tz = tz_london.localize(dsummer)

    dates_tz = [
        dwinter_tz,
        dsummer_tz,
    ]

    tz_sydney = ZoneInfo('Australia/Sydney')
    ## these will have same local time (2025-04-06 02:01:00) in Sydney due to DST shift!
    ## the second one will have fold=1 set to disambiguate
    utc_before_shift = datetime.fromisoformat('2025-04-05T15:01:00+00:00')
    utc_after__shift = datetime.fromisoformat('2025-04-05T16:01:00+00:00')
    ##
    sydney_before = utc_before_shift.astimezone(tz_sydney)
    sydney__after = utc_after__shift.astimezone(tz_sydney)

    dates_tz.extend([sydney_before, sydney__after])

    dates = [
        *dates_tz,
        dwinter,
        dsummer,
        dsummer.replace(tzinfo=timezone.utc),
    ]
    for d in dates:
        jj, dd = helper(d, datetime)
        assert str(d) == str(dd)

        # test that we preserve zone names
        if d in dates_tz:
            # this works both with pytz and zoneinfo without getting .zone or .key attributes
            assert str(d.tzinfo) == str(dd.tzinfo)

    assert helper(dsummer_tz, datetime)[0] == ('2020-08-03T01:02:03+01:00', 'Europe/London', _TZTAG_PYTZ)
    assert helper(dwinter, datetime)[0] == ('2020-02-03T01:02:03', None, None)

    assert helper(sydney_before, datetime)[0] == ('2025-04-06T02:01:00+11:00', 'Australia/Sydney', _TZTAG_ZONEINFO)
    assert helper(sydney__after, datetime)[0] == ('2025-04-06T02:01:00+10:00', 'Australia/Sydney', _TZTAG_ZONEINFO)

    assert helper(dwinter.date(), date)[0] == '2020-02-03'

    # unsupported types
    class NotSupported:
        pass

    with pytest.raises(RuntimeError, match=".*NotSupported.* isn't supported by cachew"):
        helper([NotSupported()], list[NotSupported])

    # edge cases
    helper((), tuple[()])

    # unions of generic sequences and such
    # these don't work because the erased type of both is just 'list'..
    # so there is no way to tell which one we need to construct :(
    with pytest.raises(TypeNotSupported, match=".*runtime union arguments are not unique"):
        helper([1, 2, 3], list[int] | list[Exception])
    with pytest.raises(TypeNotSupported, match=".*runtime union arguments are not unique"):
        helper([1, 2, 3], list[Exception] | list[int])


# TODO test type aliases and such??

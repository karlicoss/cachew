from __future__ import annotations

from abc import abstractmethod
from collections import abc
from dataclasses import dataclass, is_dataclass
from datetime import date, datetime, timezone
import sys
import types
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import pytz

from .common import (
    AbstractMarshall,
    Json,
    T,
)
from ..utils import TypeNotSupported, is_namedtuple


class CachewMarshall(AbstractMarshall[T]):
    def __init__(self, Type_: Type[T]) -> None:
        self.schema = build_schema(Type_)

    def dump(self, obj: T) -> Json:
        return self.schema.dump(obj)

    def load(self, dct: Json) -> T:
        return self.schema.load(dct)


# TODO add generic types later?


# NOTE: using slots gives a small speedup (maybe 5%?)
# I suppose faster access to fields or something..

SLOTS: Dict[str, bool]
if sys.version_info[:2] >= (3, 10):
    SLOTS = dict(slots=True)
else:
    # not available :(
    SLOTS = dict()


@dataclass(**SLOTS)
class Schema:
    type: Any

    @abstractmethod
    def dump(self, obj):
        raise NotImplementedError

    @abstractmethod
    def load(self, dct):
        raise NotImplementedError


@dataclass(**SLOTS)
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


@dataclass(**SLOTS)
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
        # fmt: off
        return self.type(**{
            k: ks.load(dct[k])
            for k, ks in self.fields
        })
        # fmt: on


@dataclass(**SLOTS)
class SUnion(Schema):
    # it's a bit faster to cache indixes here, gives about 15% speedup
    args: tuple[tuple[int, Schema], ...]

    def dump(self, obj):
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
        else:
            assert False, "shouldn't happen!"

    def load(self, dct):
        # tidx = d['__union_index__']
        # s = self.args[tidx]
        # return s.load(d['__value__'])
        tidx, val = dct
        _, s = self.args[tidx]
        return s.load(val)


@dataclass(**SLOTS)
class SList(Schema):
    arg: Schema

    def dump(self, obj):
        return tuple(self.arg.dump(i) for i in obj)

    def load(self, dct):
        return [self.arg.load(i) for i in dct]


@dataclass(**SLOTS)
class STuple(Schema):
    args: tuple[Schema, ...]

    def dump(self, obj):
        return tuple(a.dump(i) for a, i in zip(self.args, obj))

    def load(self, dct):
        return tuple(a.load(i) for a, i in zip(self.args, dct))


@dataclass(**SLOTS)
class SSequence(Schema):
    arg: Schema

    def dump(self, obj):
        return tuple(self.arg.dump(i) for i in obj)

    def load(self, dct):
        return tuple(self.arg.load(i) for i in dct)


@dataclass(**SLOTS)
class SDict(Schema):
    ft: SPrimitive
    tt: Schema

    def dump(self, obj):
        # fmt: off
        return {
            k: self.tt.dump(v)
            for k, v in obj.items()
        }
        # fmt: on

    def load(self, dct):
        # fmt: off
        return {
            k: self.tt.load(v)
            for k, v in dct.items()
        }
        # fmt: on


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


@dataclass(**SLOTS)
class SException(Schema):
    def dump(self, obj: Exception) -> Json:
        return tuple(_exc_helper(obj.args))

    def load(self, dct: Json):
        return self.type(*dct)


@dataclass(**SLOTS)
class SDatetime(Schema):
    def dump(self, obj: datetime) -> Json:
        iso = obj.isoformat()
        tz = obj.tzinfo
        if tz is None:
            return (iso, None)

        if isinstance(tz, pytz.BaseTzInfo):
            zone = tz.zone
            # should be present: https://github.com/python/typeshed/blame/968fd6d01d23470e0c8368e7ee7c43f54aaedc0e/stubs/pytz/pytz/tzinfo.pyi#L6
            assert zone is not None, (obj, tz)
            return (iso, zone)
        else:
            return (iso, None)

    def load(self, dct: tuple):
        iso, zone = dct
        dt = datetime.fromisoformat(iso)
        if zone is None:
            return dt

        tz = pytz.timezone(zone)
        return dt.astimezone(tz)


@dataclass(**SLOTS)
class SDate(Schema):
    def dump(self, obj: date) -> Json:
        return obj.isoformat()

    def load(self, dct: str):
        return date.fromisoformat(dct)


PRIMITIVES = {
    int,
    str,
    type(None),
    float,
    bool,
    # if type is Any, there isn't much we can do to dump it -- just dump into json and rely on the best
    # so in this sense it works exacly like primitives
    Any,
}


def build_schema(Type) -> Schema:
    if Type in PRIMITIVES:
        return SPrimitive(type=Type)

    origin = get_origin(Type)

    # if origin not none, it's some sort of generic type?
    if origin is None:
        if issubclass(Type, Exception):
            return SException(type=Type)

        if issubclass(Type, datetime):
            return SDatetime(type=Type)

        if issubclass(Type, date):
            return SDate(type=Type)

        if not (is_dataclass(Type) or is_namedtuple(Type)):
            raise TypeNotSupported(type_=Type)
        hints = get_type_hints(Type)
        fields = tuple((k, build_schema(t)) for k, t in hints.items())
        return SDataclass(
            type=Type,
            fields=fields,
        )

    args = get_args(Type)

    if sys.version_info[:2] >= (3, 10):
        is_uniontype = origin is types.UnionType
    else:
        is_uniontype = False

    is_union = origin is Union or is_uniontype

    if is_union:
        return SUnion(
            type=Type,
            # fmt: off
            args=tuple(
                (tidx, build_schema(a))
                for tidx, a in enumerate(args)
            ),
            # fmt: on
        )

    is_listish = origin is list
    if is_listish:
        (t,) = args
        return SList(
            type=Type,
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
                type=Type,
                args=tuple(build_schema(a) for a in args),
            )
        else:
            (t,) = args
            return SSequence(
                type=Type,
                arg=build_schema(t),
            )

    is_dictish = origin is dict
    if is_dictish:
        (ft, tt) = args
        fts = build_schema(ft)
        tts = build_schema(tt)
        assert isinstance(fts, SPrimitive)
        return SDict(
            type=Type,
            ft=fts,
            tt=tts,
        )

    assert False, f"unsupported: {Type} {origin} {args}"


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
        if type(x) is list:  # noqa: E721
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

    # unions
    helper(1, Union[str, int])
    if sys.version_info[:2] >= (3, 10):
        helper('aaa', str | int)

    # optionals
    helper('aaa', Optional[str])
    helper('aaa', Union[str, None])
    helper(None, Union[str, None])

    # lists
    helper([1, 2, 3], List[int])
    helper([1, 2, 3], List[int])
    helper([1, 2, 3], Sequence[int], expected=(1, 2, 3))
    helper((1, 2, 3), Sequence[int])
    helper((1, 2, 3), Tuple[int, int, int])
    helper((1, 2, 3), Tuple[int, int, int])

    # dicts
    helper({'a': 'aa', 'b': 'bb'}, Dict[str, str])
    helper({'a': None, 'b': 'bb'}, Dict[str, Optional[str]])

    # compounds of simple types
    helper(['1', 2, '3'], List[Union[str, int]])

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
        raw_data: Dict[str, Any]

    # json-ish stuff
    helper({}, Dict[str, Any])
    helper(WithJson(id=123, raw_data=dict(payload='whatever', tags=['a', 'b', 'c'])), WithJson)
    helper([], List[Any])

    # exceptions
    helper(RuntimeError('whatever!'), RuntimeError)
    # fmt: off
    helper([
        RuntimeError('I', 'am', 'exception', 123),
        Point(x=1, y=2),
        Point(x=11, y=22),
        RuntimeError('more stuff'),
        RuntimeError(),
    ], List[Union[RuntimeError, Point]])

    exc_with_datetime     = Exception('I happenned on', datetime.fromisoformat('2021-04-03T10:11:12'))
    exc_with_datetime_exp = Exception('I happenned on', '2021-04-03T10:11:12')
    helper(exc_with_datetime, Exception, expected=exc_with_datetime_exp)
    # fmt: on

    # datetimes
    tz = pytz.timezone('Europe/London')
    dwinter = datetime.strptime('20200203 01:02:03', '%Y%m%d %H:%M:%S')
    dsummer = datetime.strptime('20200803 01:02:03', '%Y%m%d %H:%M:%S')
    dwinter_tz = tz.localize(dwinter)
    dsummer_tz = tz.localize(dsummer)

    dates_pytz = [
        dwinter_tz,
        dsummer_tz,
    ]
    dates = [
        *dates_pytz,
        dwinter,
        dsummer,
        dsummer.replace(tzinfo=timezone.utc),
    ]
    for d in dates:
        jj, dd = helper(d, datetime)
        assert d.tzinfo == dd.tzinfo

        # test that we preserve pytz zone names
        if d in dates_pytz:
            assert getattr(d.tzinfo, 'zone') == getattr(dd.tzinfo, 'zone')

    assert helper(dsummer_tz, datetime)[0] == ('2020-08-03T01:02:03+01:00', 'Europe/London')
    assert helper(dwinter, datetime)[0] == ('2020-02-03T01:02:03', None)

    assert helper(dwinter.date(), date)[0] == '2020-02-03'

    # unsupported types
    class NotSupported:
        pass

    with pytest.raises(RuntimeError, match=".*NotSupported.* isn't supported by cachew"):
        helper([NotSupported()], List[NotSupported])

    # edge cases
    helper((), Tuple[()])


# TODO test type aliases and such??

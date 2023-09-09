#!/usr/bin/env python3
from abc import abstractmethod
from collections import abc
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
import os
from pathlib import Path
import sys
import types
from typing import Union, get_origin, get_args, Optional, List, Sequence, Tuple, get_type_hints, NamedTuple, Any


import orjson

from codetiming import Timer

def timer(name: str) -> Timer:
    return Timer(name=name, text=name + ': ' + '{:.2f}s')


PROFILES = Path(__file__).absolute().parent / 'profiles'


@contextmanager
def profile(name: str):
    # ugh. seems like pyinstrument slows down code quite a bit?
    if os.environ.get('PYINSTRUMENT') is None:
        yield
        return

    from pyinstrument import Profiler

    with Profiler() as profiler:
        yield

    PROFILES.mkdir(exist_ok=True)
    results_file = PROFILES / f"{name}.html"

    print("results for " + name, file=sys.stderr)
    profiler.print()

    results_file.write_text(profiler.output_html())


Json = dict[str, Any] | tuple[Any, ...]


ident = lambda x: x


primitives_to = {
    int: ident,
    str: ident,
    type(None): ident,
    float: ident,
    bool: ident,
    # if type is Any, there isn't much we can do to dump it -- just dump into json and rely on the best
    # so in this sense it works exacly like primitives
    Any: ident,
}


primitives_from = {
    int: ident,
    str: ident,
    type(None): ident,
    float: ident,
    bool: ident,
    Any: ident,
}


# https://stackoverflow.com/a/2166841/706389
def is_namedtuple(t) -> bool:
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    # pylint: disable=unidiomatic-typecheck
    return all(type(n) == str for n in f)




# NOTE: using slots gives a small speedup (maybe 5%?)
# I suppose faster access to fields or something..
@dataclass(slots=True)
class Schema:
    type: Any

    @abstractmethod
    def to_json(self, o):
        pass

    @abstractmethod
    def from_json(self, d):
        pass


@dataclass(slots=True)
class Primitive(Schema):
    def to_json(self, o):
        # NOTE: returning here directly (instead of calling identity lambda) gives about 20% speedup
        # I think custom types should have their own Schema subclass
        return o
        # prim = primitives_to.get(self.type)
        # assert prim is not None
        # return prim(o)

    def from_json(self, d):
        return d
        # prim = primitives_from.get(self.type)
        # assert prim is not None
        # return prim(d)


@dataclass(slots=True)
class Dataclass(Schema):
    # using list of tuples instead of dict gives about 5% speedup
    fields: tuple[tuple[str, Schema], ...]

    def to_json(self, o):
        # TODO would be nice if we didn't create a dictionary here
        # considering it is going to be serialized to json anyway
        # maybe we need to yield json bits actually?
        return {
            # would be kinda nice if we didn't have to use getattr here
            # but I think for dataclass this is actually the fastest way
            # TODO for NamedTuples could just use them as tuples.. think about separating
            k: ks.to_json(getattr(o, k))
            for k, ks in self.fields
        }

    def from_json(self, d):
        # dict comprehension is meh, but not sure if there is a faster way?
        return self.type(**{
            k: ks.from_json(d[k])
            for k, ks in self.fields
        })


@dataclass(slots=True)
class XUnion(Schema):
    # it's a bit faster to cache indixes here, gives about 15% speedup
    args: tuple[tuple[int, Schema], ...]

    def to_json(self, o):
        # TODO could do a bit of magic here and remember the last index that worked?
        # that way if some objects dominate the Union, the first isinstance would always work
        for tidx, a in self.args:
            if isinstance(o, a.type):  # this takes quite a lot of time (sort of expected?)
                # using lists instead of dicts gives a bit of a speedup (about 15%)
                # so probably worth it even though a bit cryptic
                # also could add a tag or something?
                # NOTE: using tuple instead of list gives a tiiny speedup
                jj = a.to_json(o)
                return (tidx, jj)
                # {
                #     '__union_index__': tidx,
                #     '__value__': jj,
                # }
        else:
            assert False, "shouldn't happen!"

    def from_json(self, d):
        # tidx = d['__union_index__']
        # s = self.args[tidx]
        # return s.from_json(d['__value__'])
        tidx, val = d
        _, s = self.args[tidx]
        return s.from_json(val)



@dataclass(slots=True)
class XList(Schema):
    arg: Schema

    def to_json(self, o):
        return tuple(self.arg.to_json(i) for i in o)

    def from_json(self, d):
        return [self.arg.from_json(i) for i in d]


@dataclass(slots=True)
class XTuple(Schema):
    args: tuple[Schema, ...]

    def to_json(self, o):
        return tuple(a.to_json(i) for a, i in zip(self.args, o))

    def from_json(self, d):
        return tuple(a.from_json(i) for a, i in zip(self.args, d))


@dataclass(slots=True)
class XSequence(Schema):
    arg: Schema

    def to_json(self, o):
        return tuple(self.arg.to_json(i) for i in o)

    def from_json(self, d):
        return tuple(self.arg.from_json(i) for i in d)


@dataclass(slots=True)
class XDict(Schema):
    ft: Primitive
    tt: Schema

    def to_json(self, o):
        return {
            k: self.tt.to_json(v)
            for k, v in o.items()
        }

    def from_json(self, d):
        return {
            k: self.tt.from_json(v)
            for k, v in d.items()
        }


# TODO unify with primitives?
JTypes = {int, str, type(None), float, bool}


def _exc_helper(args):
    for a in args:
        at = type(a)
        assert at in JTypes, (a, at)
        yield a

@dataclass(slots=True)
class XException(Schema):
    def to_json(self, o: Exception) -> Json:
        return tuple(_exc_helper(o.args))

    def from_json(self, d: Json):
        return self.type(*d)


def build_schema(Type) -> Schema:
    prim = primitives_from.get(Type)
    if prim is not None:
        return Primitive(type=Type)

    origin = get_origin(Type)

    # if origin not none, it's some sort of generic type?
    if origin is None:
        if issubclass(Type, Exception):
            return XException(type=Type)

        assert is_dataclass(Type) or is_namedtuple(Type)
        hints = get_type_hints(Type)
        fields = tuple((k, build_schema(t)) for k, t in hints.items())
        return Dataclass(
            type=Type,
            fields=fields,
        )

    args = get_args(Type)
    is_union = origin is Union or origin is types.UnionType
    if is_union:
        return XUnion(
            type=Type,
            args=tuple(
                (tidx, build_schema(a))
                for tidx, a in enumerate(args)
            ),
        )

    is_listish = origin is list
    if is_listish:
        (t,) = args
        return XList(
            type=Type,
            arg=build_schema(t),
        )

    # hmm check for is typing.Sequence doesn't pass for some reason
    # perhaps because it's a deprecated alias?
    is_tuplish = origin is tuple or origin is abc.Sequence
    if is_tuplish:
        if origin is tuple:
            return XTuple(
                type=Type,
                args=tuple(build_schema(a) for a in args),
            )
        else:
            (t, ) = args
            return XSequence(
                type=Type,
                arg=build_schema(t),
            )

    is_dictish = origin is dict
    if is_dictish:
        (ft, tt) = args
        fts = build_schema(ft)
        tts = build_schema(tt)
        assert isinstance(fts, Primitive)
        return XDict(
            type=Type,
            ft=fts,
            tt=tts,
        )

    assert False, f"unsupported: {Type} {origin} {args}"


def do_json(o, T, expected=None):
    if expected is None:
        expected = o

    schema = build_schema(T)

    # print('-----')
    # print("type", T)
    # print("schema", schema)
    # print("original", o, T)
    j = schema.to_json(o)
    # print("json    ", j)
    o2 = schema.from_json(j)
    # print("restored", o2, T)
    # print('-----')

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

    assert normalise(expected) == normalise(o2), (expected, o2)


@dataclass
class P:
    x: int
    y: int


class Name(NamedTuple):
    first: str
    last: str


IdType = int

@dataclass
class WithJson:
    id: IdType
    raw_data: dict[str, Any]


def test_basic() -> None:
    # primitives
    do_json(1, int)
    do_json('aaa', str)
    do_json(None, type(None))
    # TODO emit other value as none type? not sure what should happen

    # unions
    do_json(1, Union[str, int])
    do_json('aaa', str | int)

    # optionals
    do_json('aaa', Optional[str])
    do_json('aaa', str | None)
    do_json('aaa', str | None)

    # lists
    do_json([1, 2, 3], list[int])
    do_json([1, 2, 3], List[int])
    do_json([1, 2, 3], Sequence[int], expected=(1, 2, 3))
    do_json((1, 2, 3), Sequence[int])
    do_json((1, 2, 3), Tuple[int, int, int])
    do_json((1, 2, 3), tuple[int, int, int])


    # dicts
    do_json({'a': 'aa', 'b': 'bb'}, dict[str, str])
    do_json({'a': None, 'b': 'bb'}, dict[str, Optional[str]])


    # compounds of simple types
    do_json(['1', 2, '3'], list[str | int])


    # dataclasses
    do_json(P(x=1, y=2), P)

    # Namedtuple
    do_json(Name(first='aaa', last='bbb'), Name)

    # json-ish stuff
    do_json({}, dict[str, Any])
    do_json(WithJson(id=123, raw_data=dict(payload='whatever', tags=['a', 'b', 'c'])), WithJson)
    do_json([], list[Any])

    # exceptions
    do_json(RuntimeError('whatever!'), RuntimeError)
    do_json([
        RuntimeError('I', 'am', 'exception', 123),
        P(x=1, y=2),
        P(x=11, y=22),
        RuntimeError('more stuff'),
        RuntimeError(),
    ], list[RuntimeError | P])



# TODO not sure about this..
import gc
gc.disable()


import pytest


BType = Union[str, Name]

def prepare_union_str_namedtuple(N):
    objects: list[BType] = []
    for i in range(N):
        if i % 2 == 0:
            objects.append(str(i))
        else:
            objects.append(Name(first=f'first {i}', last=f'last {i}'))
    return objects


def do_test_union_str_namedtuple(*, count: int, test_name: str) -> None:
    objects = prepare_union_str_namedtuple(N=count)
    N = len(objects)

    schema = build_schema(BType)

    jsons = [None for _ in range(N)]
    with profile(test_name + ':serialize'), timer(f'serializing   {N} objects of type {BType}'):
        for i in range(N):
            jsons[i] = schema.to_json(objects[i])
    print(len(jsons))

    res = [None for _ in range(N)]
    with profile(test_name + ':deserialize'), timer(f'deserializing {N} objects of type {BType}'):
        for i in range(N):
            res[i] = schema.from_json(jsons[i])
    print(len(res))


@pytest.mark.parametrize('count', [
    50_000,
    1_000_000,
    5_000_000,
])
def test_union_str_namedtuple(count: int, request) -> None:
    do_test_union_str_namedtuple(count=count, test_name=request.node.name)

# OK, performance with calling this manually (not via pytest) is the same
# do_test_union_str_namedtuple(count=1_000_000, test_name='adhoc')


# TODO next test should probs be runtimeerror?

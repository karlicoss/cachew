#!/usr/bin/env python3
from dataclasses import dataclass, is_dataclass
import inspect
from typing import Union, get_origin, get_args, Optional, List, Sequence, Tuple, get_type_hints, NamedTuple

import orjson

from codetiming import Timer

def timer(name: str) -> Timer:
    return Timer(name=name, text=name + ': ' + '{:.2f}s')


@dataclass
class Comment:
    msg: str

@dataclass
class Point:
    x: int
    y: int
    comments: list[Comment]


items = [
    Point(x=1, y=2, comments=[Comment('a'), Comment('b')]),
]

ident = lambda x: x


primitives_to = {
    int: ident,
    float: ident,
    str: ident,
    type(None): ident,
}


primitives_from = {
    int: ident,
    float: ident,
    str: ident,
    type(None): ident,
}

import types


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


from functools import lru_cache


def memoize_cls(fun):
    return lru_cache(None)(fun)


missing = object()
def memoize_cls_by_str(fun):  # ok, this seems quite a bit slower than hash
    cache = {}
    def wrapper(cls):
        key = str(cls)
        v = cache.get(key, missing)
        if v is not missing:
            return v
        v = fun(cls)
        print("BEFORE", len(cache))
        cache[key] = v
        print("AFTER ", len(cache))
        print("CALLING", fun, cls, cache)
        return v
    return wrapper

# gives a big speedup, these are pretty slow
_get_type_hints = memoize_cls(get_type_hints)
_get_origin     = memoize_cls(get_origin)
_get_args       = memoize_cls(get_args)
_is_dataclass   = memoize_cls(is_dataclass)
_is_namedtuple  = memoize_cls(is_namedtuple)



from typing import Any

from abc import abstractmethod

# TODO frozen?
#
@dataclass
class Schema:
    type: Any

    @abstractmethod
    def to_json(self, o):
        pass

    @abstractmethod
    def from_json(self, d):
        pass


@dataclass
class Primitive(Schema):
    def to_json(self, o):
        prim = primitives_to.get(self.type)
        assert prim is not None
        return prim(o)

    def from_json(self, d):
        prim = primitives_from.get(self.type)
        assert prim is not None
        return prim(d)


@dataclass
class Dataclass(Schema):
    fields: dict[str, Schema]

    def to_json(self, o):
        return {
            # TODO would be nice to get rid of getattr here?
            k: ks.to_json(getattr(o, k))
            for k, ks in self.fields.items()
        }

    def from_json(self, d):
        # dict comprehension is meh, but not sure if there is a faster way?
        return self.type(**{
            k: ks.from_json(d[k])
            for k, ks in self.fields.items()
        })


@dataclass
class XUnion(Schema):
    # it's a bit faster to cache indixes here, gives about 15% speedup
    args: tuple[tuple[int, Schema], ...]

    def to_json(self, o):
        for tidx, a in self.args:
            if isinstance(o, a.type):  # this takes quite a lot of time (sort of expected?)
                # using lists instead of dicts gives a bit of a speedup (about 15%)
                # so probably worth it even though a bit cryptic
                # also could add a tag or something?
                # TODO could try returning tuples instead of lists?
                # yeah, it's a little faster -- should do it
                jj = a.to_json(o)
                return [tidx, jj]
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



@dataclass
class XList(Schema):
    arg: Schema

    def to_json(self, o):
        return [self.arg.to_json(i) for i in o]

    def from_json(self, d):
        return [self.arg.from_json(i) for i in d]


@dataclass
class XTuple(Schema):
    args: tuple[Schema, ...]

    def to_json(self, o):
        return [a.to_json(i) for a, i in zip(self.args, o)]

    def from_json(self, d):
        return tuple(a.from_json(i) for a, i in zip(self.args, d))


@dataclass
class XSequence(Schema):
    arg: Schema

    def to_json(self, o):
        return [self.arg.to_json(i) for i in o]

    def from_json(self, d):
        return tuple(self.arg.from_json(i) for i in d)


@dataclass
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


def build_schema(Type) -> Schema:
    prim = primitives_from.get(Type)
    if prim is not None:
        return Primitive(type=Type)

    origin = get_origin(Type)
    if origin is None:
        assert is_dataclass(Type) or is_namedtuple(Type)
        hints = _get_type_hints(Type)
        fields = {k: build_schema(t) for k, t in hints.items()}
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


def to_json(o, Type):
    prim = primitives_to.get(Type)
    if prim is not None:
        return prim(o)

    origin = _get_origin(Type)
    args = _get_args(Type)

    if origin is None:
        assert _is_dataclass(Type) or _is_namedtuple(Type)
        hints = _get_type_hints(Type)
        return {
            k: to_json(getattr(o, k), t)
            for k, t in hints.items()
            # todo could add type info?
        }

    is_union = origin is Union or origin is types.UnionType
    if is_union:
        for ti, t in enumerate(args):
            if isinstance(o, t):
                jj = to_json(o, t)
                # TODO __type__ here isn't really used.. but could keep some debug info? dunno
                return {'__type__': 'union', '__index__': ti, '__value__': jj}
        else:
            assert False, "shouldn't happen"
    is_listish = origin is list
    if is_listish:
        (t,) = args
        return [to_json(i, t) for i in o]
    # hmm check for is typing.Sequence doesn't pass for some reason
    # perhaps because it's a deprecated alias?
    is_tuplish = origin is tuple or origin is abc.Sequence
    if is_tuplish:
        if origin is tuple:
            return [to_json(i, t) for i, t in zip(o, args)]
        else:
            # for sequence.. a bit meh
            (t,) = args
            return [to_json(i, t) for i in o]

    is_dictish = origin is dict
    if is_dictish:
        (ft, tt) = args
        return {k: to_json(v, tt) for k, v in o.items()}

    assert False, f"unsupported: {o} {Type} {origin} {args}"

from collections import abc
def from_json(d, Type):
    prim = primitives_from.get(Type)
    if prim is not None:
        return prim(d)

    origin = get_origin(Type)
    args = get_args(Type)

    if origin is None:
        assert is_dataclass(Type) or is_namedtuple(Type)
        hints = _get_type_hints(Type)
        return Type(**{  # meh, but not sure if there is a faster way?
            k: from_json(d[k], t)
            for k, t in hints.items()
        })


    is_union = origin is Union or origin is types.UnionType
    if is_union:
        ti = d['__index__']
        t = args[ti]
        return from_json(d['__value__'], t)

    is_listish = origin is list
    if is_listish:
        (t,) = args
        return [from_json(i, t) for i in d]

    is_tuplish = origin is tuple or origin is abc.Sequence
    if is_tuplish:
        if origin is tuple:
            return tuple(from_json(i, t) for i, t in zip(d, args))
        else:
            # meh
            (t,) = args
            return tuple(from_json(i, t) for i in d)

    is_dictish = origin is dict
    if is_dictish:
        (ft, tt) = args
        return {k: from_json(v, tt) for k, v in d.items()}

    assert False, f"unsupported: {d} {Type} {origin} {args}"


Type = str | int

item: Type = 3


def do_json(o, T, expected=None):
    if expected is None:
        expected = o

    schema = build_schema(T)

    print('-----')
    print("type", T)
    print("schema", schema)
    print("original", o, T)
    j = schema.to_json(o)
    print("json    ", j)
    o2 = schema.from_json(j)
    print("restored", o2, T)
    print('-----')

    assert expected == o2, (expected, o2)


@dataclass
class P:
    x: int
    y: int


class Name(NamedTuple):
    first: str
    last: str


def test():
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


BType = Union[str, Name]

def benchmark():
    import gc
    gc.disable()

    N = 5_000_000
    objects: list[BType] = []
    for i in range(N):
        if i % 2 == 0:
            objects.append(str(i))
        else:
            objects.append(Name(first=f'first {i}', last=f'last {i}'))


    schema = build_schema(BType)

    jsons = [None for _ in range(N)]
    with timer(f'serializing   {N} objects of type {BType}'):
        for i in range(N):
            jsons[i] = schema.to_json(objects[i])
    print(len(jsons))

    # res = [None for _ in range(N)]
    # with timer(f'deserializing {N} objects of type {BType}'):
    #     for i in range(N):
    #         res[i] = schema.from_json(jsons[i])
    # print(len(res))


# test()
benchmark()

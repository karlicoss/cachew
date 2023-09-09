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


def to_json(o, Type):
    prim = primitives_to.get(Type)
    if prim is not None:
        return prim(o)

    origin = get_origin(Type)
    args = get_args(Type)

    if origin is None:
        assert is_dataclass(Type) or is_namedtuple(Type)
        hints = get_type_hints(Type)
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
            # meh
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
        hints = get_type_hints(Type)
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

    print("original", o, T)
    j = to_json(o, T)
    print("json    ", j)
    o2 = from_json(j, T)
    print("restored", o2, T)

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

    N = 1_000_000
    objects: list[BType] = []
    for i in range(N):
        if i % 2 == 0:
            objects.append(str(i))
        else:
            objects.append(Name(first=f'first {i}', last=f'last {i}'))

    jsons = [None for _ in range(N)]
    with timer(f'serializing   {N} objects of type {BType}'):
        for i in range(N):
            jsons[i] = to_json(objects[i], BType)

    res = [None for _ in range(N)]
    with timer(f'deserializing {N} objects of type {BType}'):
        for i in range(N):
            res[i] = from_json(jsons[i], BType)



benchmark()

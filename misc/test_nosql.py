#!/usr/bin/env python3
from dataclasses import dataclass
import inspect
from typing import Union, get_origin, get_args, Optional

import orjson

from cachew import is_union

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


def to_json(o, Type):
    prim = primitives_to.get(Type)
    if prim is not None:
        return prim(o)

    origin = get_origin(Type)
    args = get_args(Type)
    is_union = origin is Union or origin is types.UnionType
    if is_union:
        for ti, t in enumerate(args):
            if isinstance(o, t):
                jj = to_json(o, t)
                # TODO __type__ here isn't really used.. but could keep some debug info? dunno
                return {'__type__': 'union', '__index__': ti, '__value__': jj}
        else:
            assert False, "shouldn't happen"

        # TODO need to strip off generic??
        # need to figure out which of the union members it actually is?

    assert False, f"unsupported: {o} {Type}"


def from_json(d, Type):
    prim = primitives_from.get(Type)
    if prim is not None:
        return prim(d)

    origin = get_origin(Type)
    args = get_args(Type)
    is_union = origin is Union or origin is types.UnionType
    if is_union:
        ti = d['__index__']
        t = args[ti]
        return from_json(d['__value__'], t)
    assert False


Type = str | int

item: Type = 3


def do_json(o, T):
    print("original", o, T)
    j = to_json(o, T)
    print("json    ", j)
    o2 = from_json(j, T)
    print("restored", o2, T)
    assert o == o2, (o, o2)


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



# doit(item)
# print(orjson.dumps(item))

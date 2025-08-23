# ruff: noqa: ARG001  # ruff thinks pytest fixtures are unused arguments
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import orjson
import pytest

from ..marshall.cachew import CachewMarshall
from ..marshall.common import Json
from .utils import (
    gc_control,  # noqa: F401
    profile,
    running_on_ci,
    timer,
)

Impl = Literal[
    'cachew',  # our custom deserialization
    'cattrs',
    'legacy',  # our legacy deserialization
]
# don't include legacy by default, it's only here just for the sake of comparing once before switch
Impls: list[Impl] = ['cachew', 'cattrs']


def do_test(*, test_name: str, Type, factory, count: int, impl: Impl = 'cachew') -> None:
    if count > 100 and running_on_ci:
        pytest.skip("test too heavy for CI, only meant to run manually")

    to_json: Any
    from_json: Any
    if impl == 'cachew':
        marshall = CachewMarshall(Type_=Type)
        to_json = marshall.dump
        from_json = marshall.load
    elif impl == 'legacy':
        from ..legacy import NTBinder

        # NOTE: legacy binder emits a tuple which can be inserted directly into the database
        # so 'json dump' and 'json load' should really be disregarded for this flavor
        # if you're comparing with <other> implementation, you should compare
        # legacy serializing as the sum of <other> serializing + <other> json dump
        # that said, this way legacy will have a bit of an advantage since custom types (e.g. datetime)
        # would normally be handled by sqlalchemy instead
        binder = NTBinder.make(Type)
        to_json = binder.to_row
        from_json = binder.from_row
    elif impl == 'cattrs':
        from cattrs import Converter

        converter = Converter()

        from typing import get_args

        # TODO use later
        # from typing import Union, get_origin
        # import types
        # def is_union(type_) -> bool:
        #     origin = get_origin(type_)
        #     return origin is Union or origin is types.UnionType

        def union_structure_hook_factory(_):
            def union_hook(data, type_):
                args = get_args(type_)

                if data is None:  # we don't try to coerce None into anything
                    return None

                for t in args:
                    try:
                        res = converter.structure(data, t)
                    except Exception:
                        continue
                    else:
                        return res
                raise ValueError(f"Could not cast {data} to {type_}")

            return union_hook

        # borrowed from https://github.com/python-attrs/cattrs/issues/423
        # uhh, this doesn't really work straightaway...
        # likely need to combine what cattr does with configure_tagged_union
        # converter.register_structure_hook_factory(is_union, union_structure_hook_factory)
        # configure_tagged_union(
        #     union=Type,
        #     converter=converter,
        # )
        # NOTE: this seems to give a bit of speedup... maybe raise an issue or something?
        # fmt: off
        unstruct_func = converter._unstructure_func.dispatch(Type)  # type: ignore[call-arg, misc]  # about 20% speedup
        struct_func   = converter._structure_func  .dispatch(Type)  # type: ignore[call-arg, misc]  # TODO speedup
        # fmt: on

        to_json = unstruct_func
        # todo would be nice to use partial? but how do we bind a positional arg?
        from_json = lambda x: struct_func(x, Type)
    else:
        raise RuntimeError(impl)

    print(file=sys.stderr)  # kinda annoying, pytest starts printing on the same line as test name

    with profile(test_name + ':baseline'), timer(f'building      {count} objects of type {Type}'):
        objects = list(factory(count=count))

    jsons: list[Json] = [None for _ in range(count)]
    with profile(test_name + ':serialize'), timer(f'serializing   {count} objects of type {Type}'):
        for i in range(count):
            jsons[i] = to_json(objects[i])

    strs: list[bytes] = [None for _ in range(count)]  # type: ignore[misc]
    with profile(test_name + ':json_dump'), timer(f'json dump     {count} objects of type {Type}'):
        for i in range(count):
            # TODO any orjson options to speed up?
            strs[i] = orjson.dumps(jsons[i])

    db = Path('/tmp/cachew_test/db.sqlite')
    if db.parent.exists():
        shutil.rmtree(db.parent)
    db.parent.mkdir()

    with profile(test_name + ':sqlite_dump'), timer(f'sqlite dump   {count} objects of type {Type}'):
        with sqlite3.connect(db) as conn:
            conn.execute('CREATE TABLE data (value BLOB)')
            conn.executemany('INSERT INTO data (value) VALUES (?)', [(s,) for s in strs])
        conn.close()

    strs2: list[bytes] = [None for _ in range(count)]  # type: ignore[misc]
    with profile(test_name + ':sqlite_load'), timer(f'sqlite load   {count} objects of type {Type}'):
        with sqlite3.connect(db) as conn:
            i = 0
            for (value,) in conn.execute('SELECT value FROM data'):
                strs2[i] = value
                i += 1
        conn.close()

    cache = db.parent / 'cache.jsonl'

    with profile(test_name + ':jsonl_dump'), timer(f'jsonl dump    {count} objects of type {Type}'):
        with cache.open('wb') as fw:
            for s in strs:
                fw.write(s + b'\n')

    strs3: list[bytes] = [None for _ in range(count)]  # type: ignore[misc]
    with profile(test_name + ':jsonl_load'), timer(f'jsonl load    {count} objects of type {Type}'):
        i = 0
        with cache.open('rb') as fr:
            for l in fr:
                l = l.rstrip(b'\n')
                strs3[i] = l
                i += 1

    assert strs2[:100] + strs2[-100:] == strs3[:100] + strs3[-100:]  # just in case

    jsons2: list[Json] = [None for _ in range(count)]
    with profile(test_name + ':json_load'), timer(f'json load     {count} objects of type {Type}'):
        for i in range(count):
            # TODO any orjson options to speed up?
            jsons2[i] = orjson.loads(strs2[i])

    objects2 = [None for _ in range(count)]
    with profile(test_name + ':deserialize'), timer(f'deserializing {count} objects of type {Type}'):
        for i in range(count):
            objects2[i] = from_json(jsons2[i])

    assert objects[:100] + objects[-100:] == objects2[:100] + objects2[-100:]


@dataclass
class Name:
    first: str
    last: str


@pytest.mark.parametrize('impl', Impls)
@pytest.mark.parametrize('count', [99, 1_000_000, 5_000_000])
@pytest.mark.parametrize('gc_on', [True, False], ids=['gc_on', 'gc_off'])
def test_union_str_dataclass(impl: Impl, count: int, gc_control, request) -> None:
    # NOTE: previously was union_str_namedtuple, but adapted to work with cattrs for now
    # perf difference between datacalss/namedtuple here seems negligible so old benchmark results should apply

    if impl == 'cattrs':
        pytest.skip('TODO need to adjust the handling of Union types..')

    def factory(count: int):
        objects: list[str | Name] = []
        for i in range(count):
            if i % 2 == 0:
                objects.append(str(i))
            else:
                objects.append(Name(first=f'first {i}', last=f'last {i}'))
        return objects

    do_test(test_name=request.node.name, Type=str | Name, factory=factory, count=count, impl=impl)


# OK, performance with calling this manually (not via pytest) is the same
# do_test_union_str_dataclass(count=1_000_000, test_name='adhoc')


@pytest.mark.parametrize('impl', Impls)
@pytest.mark.parametrize('count', [99, 1_000_000, 5_000_000])
@pytest.mark.parametrize('gc_on', [True, False], ids=['gc_on', 'gc_off'])
def test_datetimes(impl: Impl, count: int, gc_control, request) -> None:
    if impl == 'cattrs':
        pytest.skip('TODO support datetime with pytz for cattrs')

    import pytz

    def factory(*, count: int):
        tzs = [
            pytz.timezone('Europe/Berlin'),
            timezone.utc,
            pytz.timezone('America/New_York'),
        ]
        start = datetime.fromisoformat('1990-01-01T00:00:00')
        end = datetime.fromisoformat('2030-01-01T00:00:00')
        step = (end - start) / count
        for i in range(count):
            dt = start + step * i
            tz = tzs[i % len(tzs)]
            yield dt.replace(tzinfo=tz)

    do_test(test_name=request.node.name, Type=datetime, factory=factory, count=count, impl=impl)


@pytest.mark.parametrize('impl', Impls)
@pytest.mark.parametrize('count', [99, 1_000_000])
@pytest.mark.parametrize('gc_on', [True, False], ids=['gc_on', 'gc_off'])
def test_nested_dataclass(impl: Impl, count: int, gc_control, request) -> None:
    # NOTE: was previously named test_many_from_cachew
    @dataclass
    class UUU:
        xx: int
        yy: int

    @dataclass
    class TE2:
        value: int
        uuu: UUU
        value2: int

    def factory(*, count: int):
        for i in range(count):
            yield TE2(value=i, uuu=UUU(xx=i, yy=i), value2=i)

    do_test(test_name=request.node.name, Type=TE2, factory=factory, count=count, impl=impl)


# TODO next test should probs be runtimeerror?

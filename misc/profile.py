#!/usr/bin/env python3
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import sqlalchemy
from codetiming import Timer
from more_itertools import ilen

from cachew import cachew

# todo not sure it really helps much?
import gc  # isort: skip

gc.disable()


def timer(name: str) -> Timer:
    return Timer(name=name, text=name + ': ' + '{:.2f}s')


def test_ints() -> None:
    N = 5_000_000

    base = Path('/tmp/cachew_profiling/')
    # shutil.rmtree(base)
    base.mkdir(exist_ok=True, parents=True)

    cache_path = base / 'ints'

    def fun_nocachew(n) -> Iterator[int]:
        yield from range(n)

    @cachew(cache_path=cache_path, force_file=True)
    def fun(n) -> Iterator[int]:
        yield from range(n)

    # with timer('no caching'):
    #     ilen(fun_nocachew(N))

    # with timer('initial call'):
    #     ilen(fun(N))

    assert cache_path.exists()  # just in case
    with timer('reading directly via sqlite'):
        total = 0
        with sqlite3.connect(cache_path) as conn:
            for (_x,) in conn.execute('SELECT * FROM cache'):
                total += 1
        assert total == N  # just in case

    with timer('reading directly via sqlalchemy'):
        total = 0
        engine = sqlalchemy.create_engine(f'sqlite:///{cache_path}')

        from sqlalchemy import Column, MetaData, Table

        meta = MetaData()
        table_cache = Table('cache', meta, Column('_cachew_primitive', sqlalchemy.Integer))
        with engine.connect() as conn:
            with timer('sqlalchemy querying'):
                rows = conn.execute(table_cache.select())
                for (_x,) in rows:
                    total += 1
        engine.dispose()
        assert total == N  # just in case

    cache_size_mb = cache_path.stat().st_size / 10**6
    print(f'cache size: {cache_size_mb:.1f} Mb')

    with timer('subsequent call'):
        ilen(fun(N))


test_ints()

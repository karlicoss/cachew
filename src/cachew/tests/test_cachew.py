from datetime import datetime, date, timezone
import logging
from pathlib import Path
from random import Random
from subprocess import check_call
import string
import sys
import time
import timeit
from typing import NamedTuple, Iterator, Optional, List, Set, Tuple, cast, Iterable, Dict, Any, Union

from more_itertools import one

import pytz
import pytest  # type: ignore

from .. import cachew, get_logger, PRIMITIVES, NTBinder, CachewException, Types, Values, settings


logger = get_logger()

@pytest.fixture(autouse=True)
def throw_on_errors():
    # a more reasonable default for tests
    settings.THROW_ON_ERROR = True
    yield


@pytest.fixture
def restore_settings():
    orig = {k: v for k, v in settings.__dict__.items() if not k.startswith('__')}
    try:
        yield
    finally:
        for k, v in orig.items():
            setattr(settings, k, v)


@pytest.fixture
def tdir(tmp_path):
    # TODO just use tmp_path instead?
    yield Path(tmp_path)


@pytest.mark.skipif(sys.version_info < (3, 7), reason="""
wtf??? 
>>> from typing import Union
>>> from datetime import date, datetime
>>> Union[date, datetime]
<class 'datetime.date'>
same with bool/int apparently
""")
def test_mypy_annotations():
    # mypy won't handle, so this has to be dynamic
    from typing import Union
    vs = []
    for t in Types.__args__: # type: ignore
        (arg, ) = t.__args__
        vs.append(arg)

    def types(ts):
        return list(sorted(ts, key=lambda t: str(t)))

    assert types(vs) == types(Values.__args__)  # type: ignore

    for p in PRIMITIVES:
        assert p in Values.__args__ # type: ignore


@pytest.mark.parametrize('tp, val', [
    (int, 22),
    (bool, False),
    (Optional[str], 'abacaba'),
    (Union[str, int], 1),
])
def test_ntbinder_primitive(tp, val):
    # TODO good candidate for property tests...
    b = NTBinder.make(tp, name='x')
    row = b.to_row(val)
    vv = b.from_row(list(row)) # type: ignore[var-annotated]
    assert vv == val


class UUU(NamedTuple):
    xx: int
    yy: int


def test_custom_hash(tdir):
    """
    Demo of using argument's modification time to determine if underlying data changed
    """
    src = tdir / 'source'
    src.write_text('0')

    entities = [
        UUU(xx=1, yy=1),
        UUU(xx=2, yy=2),
        UUU(xx=3, yy=3),
    ]
    calls = 0

    @cachew(
        cache_path=tdir,
        hashf=lambda path: path.stat().st_mtime  # when path is update, underlying cache would be discarded
    )
    def data(path: Path) -> Iterable[UUU]:
        nonlocal calls
        calls += 1
        count = int(path.read_text())
        return entities[:count]

    ldata = lambda: list(data(path=src))

    assert len(ldata()) == 0
    assert len(ldata()) == 0
    assert len(ldata()) == 0
    assert calls == 1

    src.write_text('1')
    assert ldata() == entities[:1]
    assert ldata() == entities[:1]
    assert calls == 2

    src.write_text('3')
    assert ldata() == entities
    assert ldata() == entities
    assert calls == 3


def test_caching(tdir):
    @cachew(tdir)
    def data() -> Iterator[UUU]:
        time.sleep(1)
        for i in range(5):
            yield UUU(xx=i, yy=i)
            time.sleep(1)

    # https://stackoverflow.com/a/40385994/706389
    template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""
    timeit.template = template # type: ignore

    timer = timeit.Timer(lambda: len(list(data())))
    t, cnt = cast(Tuple[float, int], timer.timeit(number=1))
    assert cnt == 5
    assert t > 5.0, 'should take at least 5 seconds'

    t, cnt = cast(Tuple[float, int], timer.timeit(number=1))
    assert cnt == 5
    assert t < 2.0, 'should be pretty much instantaneous'


class TE2(NamedTuple):
    value: int
    uuu: UUU
    value2: int


# TODO also profile datetimes?
def test_many(tdir):
    COUNT = 100000
    logger = get_logger()

    src = tdir / 'source'
    src.touch()

    @cachew(cache_path=lambda path: tdir / (path.name + '.cache'), cls=TE2)
    def _iter_data(_: Path):
        for i in range(COUNT):
            yield TE2(value=i, uuu=UUU(xx=i, yy=i), value2=i)

    def iter_data():
        return _iter_data(src)

    def ilen(it):
        ll = 0
        for _ in it:
            ll += 1
        return ll
    assert ilen(iter_data()) == COUNT
    assert ilen(iter_data()) == COUNT
    logger.debug('done')

    # serializing to db
    # in-memory: 16 seconds

    # without transaction: 22secs
    # without transaction and size 100 chunks -- some crazy amount of time, as expected

    # with transaction:
    # about 17 secs to write 1M entries (just None)
    # chunking by 20K doesn't seem to help
    # chunking by 100 also gives same perf

    # with to_row binding: 21 secs for dummy NamedTuple with None inside, 22 for less trivial class

    # deserializing from db:
    # initially, took 20 secs to load 1M entries (TE2)
    # 9 secs currently
    # 6 secs if we instantiate namedtuple directly via indices
    # 3.5 secs if we just return None from row


class BB(NamedTuple):
    xx: int
    yy: int


class AA(NamedTuple):
    value: int
    b: Optional[BB]
    value2: int


def test_return_type_inference(tdir):
    """
    Tests that return type (BB) is inferred from the type annotation
    """

    @cachew(tdir)
    def data() -> Iterator[BB]:
        yield BB(xx=1, yy=2)
        yield BB(xx=3, yy=4)

    assert len(list(data())) == 2
    assert len(list(data())) == 2


def test_return_type_mismatch(tdir):
    # even though user got invalid type annotation here, they specified correct type, and it's the one that should be used
    @cachew(tdir, cls=AA)
    def data2() -> List[BB]:
        return [
            AA(value=1, b=None, value2=123),  # type: ignore[list-item]
        ]

    # TODO hmm, this is kinda a downside that it always returns
    # could preserve the original return type, but too much trouble for now

    assert list(data2()) == [AA(value=1, b=None, value2=123)]


def test_return_type_none(tdir):
    with pytest.raises(CachewException):
        @cachew(tdir)
        # pylint: disable=unused-variable
        def data():
            return []


def test_callable_cache_path(tdir):
    """
    Cache path can be function dependent on wrapped function's arguments
    """
    called: Set[str] = set()
    @cachew(cache_path=lambda kind: tdir / f'{kind}.cache')
    def get_data(kind: str) -> Iterator[BB]:
        assert kind not in called
        called.add(kind)
        if kind == 'first':
            yield BB(xx=1, yy=1)
        else:
            yield BB(xx=2, yy=2)

    assert list(get_data('first'))  == [BB(xx=1, yy=1)]
    assert list(get_data('second')) == [BB(xx=2, yy=2)]
    assert list(get_data('first'))  == [BB(xx=1, yy=1)]
    assert list(get_data('second')) == [BB(xx=2, yy=2)]


def test_nested(tdir):

    d1 = AA(
        value=1,
        b=BB(xx=2, yy=3),
        value2=4,
    )
    d2 = AA(
        value=3,
        b=None,
        value2=5,
    )
    def data():
        yield d1
        yield d2

    @cachew(cache_path=tdir, cls=AA)
    def get_data():
        yield from data()

    assert list(get_data()) == [d1, d2]
    assert list(get_data()) == [d1, d2]


class BBv2(NamedTuple):
    xx: int
    yy: int
    zz: float


def test_schema_change(tdir):
    """
    Should discard cache on schema change (BB to BBv2) in this example
    """
    b = BB(xx=2, yy=3)

    @cachew(cache_path=tdir, cls=BB)
    def get_data():
        return [b]

    assert list(get_data()) == [b]

    # TODO make type part of key?
    b2 = BBv2(xx=3, yy=4, zz=5.0)
    @cachew(cache_path=tdir, cls=BBv2)
    def get_data_v2():
        return [b2]

    assert list(get_data_v2()) == [b2]


def test_transaction(tdir):
    """
    Should keep old cache and not leave it in some broken state in case of errors
    """
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

    class TestError(Exception):
        pass

    @cachew(cache_path=tdir, cls=BB, chunk_by=1)
    def get_data(version: int):
        for i in range(3):
            yield BB(xx=2, yy=i)
            if version == 2:
                raise TestError

    exp = [BB(xx=2, yy=0), BB(xx=2, yy=1), BB(xx=2, yy=2)]
    assert list(get_data(1)) == exp
    assert list(get_data(1)) == exp

    # TODO test that hash is unchanged?
    with pytest.raises(TestError):
        list(get_data(2))

    assert list(get_data(1)) == exp


class Job(NamedTuple):
    company: str
    title: Optional[str]


def test_optional(tdir):
    """
    Tests support for typing.Optional
    """

    @cachew(tdir)
    def data() -> Iterator[Job]:
        yield Job('google'      , title='engineed')
        yield Job('selfemployed', title=None)

    list(data()) # trigger cachew
    assert list(data()) == [
        Job('google'      , title='engineed'),
        Job('selfemployed', title=None),
    ]

# TODO add test for optional for misleading type annotation


class Person(NamedTuple):
    name: str
    secondname: str
    age: int
    job: Optional[Job]


def make_people_data(count: int) -> Iterator[Person]:
    g = Random(124)
    chars = string.ascii_uppercase + string.ascii_lowercase

    randstr = lambda len_: ''.join(g.choices(chars, k=len_))

    for _ in range(count):
        has_job = g.choice([True, False])
        maybe_job: Optional[Job] = None
        if has_job:
            maybe_job = Job(company=randstr(12), title=randstr(8))

        yield Person(
            name=randstr(5),
            secondname=randstr(10),
            age=g.randint(20, 60),
            job=maybe_job,
        )


class Breaky(NamedTuple):
    job_title: int
    job: Optional[Job]


def test_unique(tdir):
    assert [c.name for c in NTBinder.make(Breaky).columns] == [
        'job_title',
        '_job_is_null',
        'job_company',
        '_job_title',
    ]

    b = Breaky(
        job_title=123,
        job=Job(company='123', title='whatever'),
    )
    @cachew(cache_path=tdir)
    def iter_breaky() -> Iterator[Breaky]:
        yield b
        yield b

    assert list(iter_breaky()) == [b, b]
    assert list(iter_breaky()) == [b, b]


def test_stats(tdir):
    cache_file = tdir / 'cache'

    # 4 + things are string lengths
    one = (4 + 5) + (4 + 10) + 4 + (4 + 12 + 4 + 8)
    N = 10000

    @cachew(cache_path=cache_file, cls=Person)
    def get_people_data() -> Iterator[Person]:
        yield from make_people_data(count=N)

    list(get_people_data())
    print(f"Cache db size for {N} entries: estimated size {one * N // 1024} Kb, actual size {cache_file.stat().st_size // 1024} Kb;")


def test_dataclass(tdir):
    from dataclasses import dataclass
    @dataclass
    class Test:
        field: int

    @cachew(tdir)
    def get_dataclasses() -> Iterator[Test]:
        yield from [Test(field=i) for i in range(5)]

    assert list(get_dataclasses()) == [Test(field=i) for i in range(5)]
    assert list(get_dataclasses()) == [Test(field=i) for i in range(5)]


def test_dates(tmp_path):
    from dataclasses import dataclass
    @dataclass
    class X:
        d1: datetime
        d2: datetime
        d3: datetime
        d4: datetime
        d5: datetime

    tz = pytz.timezone('Europe/London')
    dwinter = datetime.strptime('20200203 01:02:03', '%Y%m%d %H:%M:%S')
    dsummer = datetime.strptime('20200803 01:02:03', '%Y%m%d %H:%M:%S')

    x = X(
        d1=tz.localize(dwinter),
        d2=tz.localize(dsummer),
        d3=dwinter,
        d4=dsummer,
        d5=dsummer.replace(tzinfo=timezone.utc),
    )

    @cachew(tmp_path)
    def fun() -> Iterable[X]:
        yield x

    assert one(fun()) == x
    assert one(fun()) == x

    # make sure the actuall tzinfo is preserved... otherwise we might end up with raw offsets and lose some info
    r = one(fun())
    # attempting to preserve pytz zone names is a bit arbitrary
    # but on the other hand, they will be in python 3.9, so I guess it's ok
    assert r.d1.tzinfo.zone == x.d1.tzinfo.zone # type: ignore
    assert r.d2.tzinfo.zone == x.d2.tzinfo.zone # type: ignore
    assert r.d3.tzname() is None
    assert r.d4.tzname() is None
    assert r.d5.tzinfo is timezone.utc


def test_types(tdir):
    from dataclasses import dataclass, asdict
    @dataclass
    class Test:
        an_int : int
        a_bool : bool
        a_float: float
        a_str  : str
        a_dt   : datetime
        a_date : date
        a_json : Dict[str, Any]
        an_exc : Exception

    # pylint: disable=no-member
    assert len(Test.__annotations__) == len(PRIMITIVES) # precondition so we don't forget to update test

    tz = pytz.timezone('Europe/Berlin')
    obj = Test(
        an_int =1123,
        a_bool =True,
        a_float=3.131,
        a_str  ='abac',
        a_dt   =datetime.now(tz=tz),
        a_date =datetime.now().replace(year=2000).date(),
        a_json ={'a': True, 'x': {'whatever': 3.14}},
        an_exc =RuntimeError('error!', 123),
    )

    @cachew(tdir)
    def get() -> Iterator[Test]:
        yield obj

    def H(t: Test):
        # Exceptions can't be directly compared.. so this kinda helps
        d = asdict(t)
        d['an_exc'] = d['an_exc'].args
        return d

    assert H(one(get())) == H(obj)
    assert H(one(get())) == H(obj)


# TODO if I do perf tests, look at this https://docs.sqlalchemy.org/en/13/_modules/examples/performance/large_resultsets.html
# TODO should be possible to iterate anonymous tuples too? or just sequences of primitive types?

def test_primitive(tmp_path: Path):
    @cachew(tmp_path)
    def fun() -> Iterator[str]:
        yield 'aba'
        yield 'caba'

    assert list(fun()) == ['aba', 'caba']
    assert list(fun()) == ['aba', 'caba']


class O(NamedTuple):
    x: int

def test_default(tmp_path: Path):
    class HackHash:
        def __init__(self, x: int) -> None:
            self.x = x

        def __repr__(self):
            return repr(self.x)

    hh = HackHash(1)

    @cachew(tmp_path, hashf=lambda param: param.x)
    def fun(param=hh) -> Iterator[O]:
        yield O(hh.x)

    list(fun())
    assert list(fun()) == [O(1)]

    # now, change hash. That should cause the composite hash to invalidate and recompute
    hh.x = 2
    assert list(fun()) == [O(2)]

    # should be ok with explicitly passing
    assert list(fun(param=HackHash(2))) == [O(2)]


class U(NamedTuple):
    x: Union[str, O]

def test_union(tmp_path: Path):
    @cachew(tmp_path)
    def fun() -> Iterator[U]:
        yield U('hi')
        yield U(O(123))

    list(fun())
    assert list(fun()) == [U('hi'), U(O(123))]


def _concurrent_helper(cache_path: Path, count: int, sleep_s=0.1):
    from time import sleep
    @cachew(cache_path)
    def test(count: int) -> Iterator[int]:
        for i in range(count):
            print(f"{count}: GENERATING {i}")
            sleep(sleep_s)
            yield i * i

    return list(test(count=count))


@pytest.fixture
def fuzz_cachew_impl():
    """
    Insert random sleeps in cachew_impl to increase likelihood of concurrency issues
    """
    import patchy  # type: ignore[import]
    from .. import cachew_impl
    patch = '''\
@@ -740,6 +740,11 @@

             logger.debug('old hash: %s', prev_hash)

+            from random import random
+            rs = random() * 2
+            print("sleeping for: ", rs)
+            from time import sleep; sleep(rs)
+
             if h == prev_hash:
                 logger.debug('hash matched: loading from cache')
                 rows = conn.execute(values_table.select())
'''
    patchy.patch(cachew_impl, patch)
    yield
    patchy.unpatch(cachew_impl, patch)


# TODO fuzz when they start so they enter transaction at different times?

# TODO how to run it enough times on CI and increase likelihood of failing?
# for now, stress testing manually:
# while PYTHONPATH=src pytest -s cachew -k concurrent_writes ; do sleep 0.5; done
def test_concurrent_writes(tmp_path: Path, fuzz_cachew_impl):
    cache_path = tmp_path / 'cache.sqlite'
    from concurrent.futures import ProcessPoolExecutor as Pool

    # warm up to create the database
    # FIXME ok, that will be fixed separately with atomic move I suppose
    _concurrent_helper(cache_path, 1)

    processes = 5
    with Pool() as pool:
        futures = [
            pool.submit(_concurrent_helper, cache_path, count)
        for count in range(processes)]

        for count, f in enumerate(futures):
            assert f.result() == [i * i for i in range(count)]


# TODO ugh. need to keep two processes around to test for yield holding transaction lock

def test_concurrent_reads(tmp_path: Path, fuzz_cachew_impl):
    cache_path = tmp_path / 'cache.sqlite'
    from concurrent.futures import ProcessPoolExecutor as Pool

    count = 10
    # warm up
    _concurrent_helper(cache_path, count, sleep_s=0)

    processes = 4

    import time
    start = time.time()
    with Pool() as pool:
        futures = [
            pool.submit(_concurrent_helper, cache_path, count, 1)
        for _ in range(processes)]

        for f in futures:
            print(f.result())
    end = time.time()

    taken = end - start
    # should be pretty instantaneous
    # if it takes more, most likely means that helper was called again
    assert taken < 5


def test_mcachew(tmp_path: Path):
    # TODO how to test for defensive behaviour?
    from cachew.misc import mcachew

    # TODO check throw on error
    @mcachew(cache_path=tmp_path / 'cache')
    def func() -> Iterator[str]:
        yield 'one'
        yield 'two'

    assert list(func()) == ['one', 'two']
    assert list(func()) == ['one', 'two']


def test_defensive(restore_settings):
    '''
    Make sure that cachew doesn't crash on misconfiguration
    '''
    def orig() -> Iterator[int]:
        yield 123

    def orig2():
        yield "x"
        yield 123

    fun = cachew(bad_arg=123)(orig)
    assert list(fun()) == [123]
    assert list(fun()) == [123]

    from cachew.compat import nullcontext
    for throw in [True, False]:
        ctx = pytest.raises(Exception) if throw else nullcontext()
        settings.THROW_ON_ERROR = throw

        with ctx: # type: ignore
            fun = cachew(cache_path=lambda: 1 + 'bad_path_provider')(orig) # type: ignore
            assert list(fun()) == [123]
            assert list(fun()) == [123]

            fun = cachew(cache_path=lambda p: '/tmp/' + str(p))(orig)
            assert list(fun()) == [123]
            assert list(fun()) == [123]

            fun = cachew(orig2)
            assert list(fun()) == ['x', 123]
            assert list(fun()) == ['x', 123]


def test_recursive(tmp_path: Path):
    calls = 0
    @cachew(tmp_path)
    def factorials(n: int) -> Iterable[int]:
        nonlocal calls
        calls += 1
        if n == 0:
            yield 1
            return
        prev = factorials(n - 1)
        last = 1
        # TODO potentially quadratic? measure perf perhaps?
        for x in prev:
            yield x
            last = x
        yield last * n

    assert calls == 0
    assert list(factorials(3)) == [1, 1, 2, 6]
    assert calls == 4
    assert list(factorials(3)) == [1, 1, 2, 6]
    assert calls == 4
    assert list(factorials(5)) == [1, 1, 2, 6, 24, 120]
    assert calls == 6
    assert list(factorials(3)) == [1, 1, 2, 6]
    assert calls == 10



def test_deep_recursive(tmp_path: Path):
    @cachew(tmp_path)
    def rec(n: int) -> Iterable[int]:
        if n == 0:
            yield 0
            return
        yield from rec(n - 1)
        yield n

    rlimit = sys.getrecursionlimit()

    # NOTE in reality it has to do with the number of file descriptors (ulimit -Sn, e.g. 1024?)
    # but it seems that during the error unrolling, pytest or something else actually hits the recursion limit somehow
    # pytest ends up with an internal error in such case... which is good enough as long as tests are concerned I guess.
    try:
        sys.setrecursionlimit(2000)
        list(rec(800))
        list(rec(800))
    finally:
        sys.setrecursionlimit(rlimit)


def test_recursive_error(tmp_path: Path):
    @cachew(tmp_path)
    def rec(n: int) -> Iterable[int]:
        if n == 0:
            yield 0
            return
        yield from rec(n - 1)
        yield n

    rlimit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(50)
        list(rec(100))
        raise AssertionError('Expecting recursion error')
    except RecursionError:
        pass
    finally:
        sys.setrecursionlimit(rlimit)

    # todo not sure if cache file should exist??
    # either way, at least check that the db is not completely messed up
    assert len(list(rec(100))) == 101


def test_exceptions(tmp_path: Path):
    class X(NamedTuple):
        a: int

    d = datetime.strptime('20200102 03:04:05', '%Y%m%d %H:%M:%S')
    @cachew(tmp_path)
    def fun() -> Iterator[Exception]:
        yield RuntimeError('whatever', 123, d, X(a=123))

    list(fun())
    [e] = fun()
    # not sure if there is anything that can be done to preserve type information?
    assert type(e) == Exception
    assert e.args == ('whatever', 123, '2020-01-02T03:04:05', 'X(a=123)')


# see https://beepb00p.xyz/mypy-error-handling.html#kiss
def test_result(tmp_path: Path):
    @cachew(tmp_path)
    def fun() -> Iterator[Union[Exception, int]]:
        yield 1
        yield RuntimeError("sad!")
        yield 123
    list(fun())
    [v1, ve, v123] = fun()
    assert v1 == 1
    assert v123 == 123
    assert ve.args == ('sad!',)


def test_version_change(tmp_path: Path):
    calls = 0
    @cachew(tmp_path, logger=logger)
    def fun() -> Iterator[str]:
        nonlocal calls
        calls += 1

        yield from ['a', 'b', 'c']

    list(fun())
    list(fun())
    assert calls == 1

    # todo ugh. not sure how to do this as a relative import??
    import cachew as cachew_module
    old_version = cachew_module.CACHEW_VERSION

    try:
        cachew_module.CACHEW_VERSION = old_version + '_whatever'
        # should invalidate cachew now
        list(fun())
        assert calls == 2
        list(fun())
        assert calls == 2
    finally:
        cachew_module.CACHEW_VERSION = old_version

    # and now again, back to the old version
    list(fun())
    assert calls == 3
    list(fun())
    assert calls == 3


def dump_old_cache(tmp_path: Path):
    # call this if you want to get an sql script for version upgrade tests..
    oc = tmp_path / 'old_cache.sqlite'
    @cachew(oc)
    def fun() -> Iterator[int]:
        yield from [1, 2, 3]

    list(fun())
    assert oc.exists(), oc
    from subprocess import check_output, check_call
    sql = check_output(['sqlite3', oc, '.dump']).decode('utf8')
    print(sql, file=sys.stderr)



def test_old_cache_v0_6_3(tmp_path: Path):
    sql = '''
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE hash (
	value VARCHAR
);
INSERT INTO hash VALUES('cachew: 1, schema: {''_'': <class ''int''>}, hash: ()');
CREATE TABLE IF NOT EXISTS "table" (
	_cachew_primitive INTEGER
);
INSERT INTO "table" VALUES(1);
INSERT INTO "table" VALUES(2);
INSERT INTO "table" VALUES(3);
COMMIT;
    '''
    db = tmp_path / 'cache.sqlite'
    check_call(['sqlite3', db, sql])

    @cachew(db)
    def fun() -> Iterator[int]:
        yield from [1, 2, 3]

    # this tests that it doesn't crash
    # for actual version upgrade test see test_version_change
    assert [1, 2, 3] == list(fun())

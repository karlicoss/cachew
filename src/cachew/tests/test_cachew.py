from datetime import datetime, date
from pathlib import Path
from random import Random
import string
import sys
import time
import timeit
from typing import NamedTuple, Iterator, Optional, List, Set, Tuple, cast, Iterable, Dict, Any

import pytz
import pytest  # type: ignore

from .. import cachew, get_logger, PRIMITIVES, NTBinder, CachewException, Types, Values


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
    import pytest # type: ignore
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


def test_types(tdir):
    from dataclasses import dataclass
    @dataclass
    class Test:
        an_int : int
        a_bool : bool
        a_float: float
        a_str  : str
        a_dt   : datetime
        a_date : date
        a_json : Dict[str, Any]
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
    )

    @cachew(tdir)
    def get() -> Iterator[Test]:
        yield obj

    assert list(get()) == [obj]
    assert list(get()) == [obj]


# TODO if I do perf tests, look at this https://docs.sqlalchemy.org/en/13/_modules/examples/performance/large_resultsets.html
# TODO should be possible to iterate anonymous tuples too? or just sequences of primitive types?

@pytest.mark.skip("TODO need to support primitive types as well")
def test_primitive(tmp_path: Path):
    @cachew(tmp_path)
    def fun() -> Iterator[int]:
        yield 1
        yield 2

    assert list(fun()) == [1, 2]
    assert list(fun()) == [1, 2]


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

from datetime import datetime, date
from itertools import zip_longest
from pathlib import Path
from random import Random
import string
from typing import NamedTuple, Iterator, Optional, List, Set

import pytz
import sqlalchemy  # type: ignore
import pytest  # type: ignore

from .. import cachew, get_logger, mtime_hash, PRIMITIVES, NTBinder, CachewException, Types, Values


# TODO mypy is unhappy about inline namedtuples.. perhaps should open an issue
class TE(NamedTuple):
    dt: datetime
    value: float
    flag: bool


def test_mypy_annotations():
    # mypy won't handle, so this has to be dynamic
    for t, v in zip_longest(Types.__args__, Values.__args__): # type: ignore
        (arg, ) = t.__args__
        assert arg == v

    for p in PRIMITIVES:
        assert p in Values.__args__ # type: ignore


def test_simple(tmp_path):
    mad = pytz.timezone('Europe/Madrid')
    utc = pytz.utc


    tdir = Path(tmp_path)
    src = tdir / 'source'
    src.write_text('0')

    db_path = tdir / 'db.sqlite'

    entities = [
        TE(dt=utc.localize(datetime(year=1991, month=5, day=3, minute=1)), value=123.43242, flag=True),
        TE(dt=mad.localize(datetime(year=1997, month=7, day=4, second=5)), value=9842.4234, flag=False),
    ]

    accesses = 0
    @cachew(db_path, hashf=mtime_hash, cls=TE)
    def _get_data(path: Path):
        nonlocal accesses
        accesses += 1
        count = int(path.read_text())
        return entities[:count]

    def get_data():
        return list(_get_data(src))

    assert len(get_data()) == 0
    assert len(get_data()) == 0
    assert len(get_data()) == 0
    assert accesses == 1

    src.write_text('1')
    assert get_data() == entities[:1]
    assert get_data() == entities[:1]
    assert accesses == 2

    src.write_text('2')
    assert get_data() == entities
    assert get_data() == entities
    assert accesses == 3


class UUU(NamedTuple):
    xx: int
    yy: int
class TE2(NamedTuple):
    value: int
    uuu: UUU
    value2: int

# TODO also profile datetimes?
def test_many(tmp_path):
    COUNT = 100000
    logger = get_logger()

    tdir = Path(tmp_path)
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


def test_return_type_inference(tmp_path):
    """
    Tests that return type (BB) is inferred from the type annotation
    """
    tdir = Path(tmp_path)

    @cachew(cache_path=tdir / 'cache')
    def data() -> Iterator[BB]:
        yield BB(xx=1, yy=2)
        yield BB(xx=3, yy=4)

    assert len(list(data())) == 2
    assert len(list(data())) == 2


def test_return_type_mismatch(tmp_path):
    tdir = Path(tmp_path)
    # even though user got invalid type annotation here, they specified correct type, and it's the one that should be used
    @cachew(cache_path=tdir / 'cache2', cls=AA)
    def data2() -> List[BB]:
        return [ # type: ignore
            AA(value=1, b=None, value2=123),
        ]

    # TODO hmm, this is kinda a downside that it always returns
    # could preserve the original return type, but too much trouble for now

    assert list(data2()) == [AA(value=1, b=None, value2=123)]


def test_return_type_none(tmp_path):
    tdir = Path(tmp_path)
    with pytest.raises(CachewException):
        @cachew(cache_path=tdir / 'cache')
        # pylint: disable=unused-variable
        def data():
            return []


def test_callable_cache_path(tmp_path):
    tdir = Path(tmp_path)
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


def test_nested(tmp_path):
    tdir = Path(tmp_path)

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

    @cachew(cache_path=tdir / 'cache', cls=AA)
    def get_data():
        yield from data()

    assert list(get_data()) == [d1, d2]
    assert list(get_data()) == [d1, d2]


class BBv2(NamedTuple):
    xx: int
    yy: int
    zz: float


def test_schema_change(tmp_path):
    """
    Should discard cache on schema change (BB to BBv2) in this example
    """
    tdir = Path(tmp_path)
    b = BB(xx=2, yy=3)

    @cachew(cache_path=tdir / 'cache', cls=BB)
    def get_data():
        return [b]

    assert list(get_data()) == [b]

    # TODO make type part of key?
    b2 = BBv2(xx=3, yy=4, zz=5.0)
    @cachew(cache_path=tdir / 'cache', cls=BBv2)
    def get_data_v2():
        return [b2]

    assert list(get_data_v2()) == [b2]


def test_transaction(tmp_path):
    """
    Should keep old cache and not leave it in some broken state in case of errors
    """
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    tdir = Path(tmp_path)

    class TestError(Exception):
        pass

    @cachew(cache_path=tdir / 'cache', cls=BB, chunk_by=1)
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


# TODO test NTBinder tree instead
# def test_namedtuple_schema():
#     schema = get_namedtuple_schema(Person)
#     assert schema == (
#         ('name'      , str, False),
#         ('secondname', str, False),
#         ('age'       , int, False),
#         ('job'       , Job, True),
#     )


def test_binder():
    b = NTBinder.make(Person)

    # TODO that could be a doctest showing actual database schema
    assert [(c.name, type(c.type)) for c in b.columns] == [
        ('name'        , sqlalchemy.String),
        ('secondname'  , sqlalchemy.String),
        ('age'         , sqlalchemy.Integer),

        ('_job_is_null', sqlalchemy.Boolean),
        ('job_company' , sqlalchemy.String),
        ('job_title'   , sqlalchemy.String),
    ]


class Breaky(NamedTuple):
    job_title: int
    job: Optional[Job]


def test_unique(tmp_path):
    tdir = Path(tmp_path)

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
    @cachew(cache_path=tdir / 'cache')
    def iter_breaky() -> Iterator[Breaky]:
        yield b
        yield b

    assert list(iter_breaky()) == [b, b]
    assert list(iter_breaky()) == [b, b]


def test_stats(tmp_path):
    tdir = Path(tmp_path)

    cache_file = tdir / 'cache'

    # 4 + things are string lengths
    one = (4 + 5) + (4 + 10) + 4 + (4 + 12 + 4 + 8)
    N = 10000

    @cachew(cache_path=cache_file, cls=Person)
    def get_people_data() -> Iterator[Person]:
        yield from make_people_data(count=N)

    list(get_people_data())
    print(f"Cache db size for {N} entries: estimated size {one * N // 1024} Kb, actual size {cache_file.stat().st_size // 1024} Kb;")


def test_dataclass(tmp_path):
    tdir = Path(tmp_path)

    from dataclasses import dataclass
    @dataclass
    class Test:
        field: int

    @cachew(tdir / 'cache')
    def get_dataclasses() -> Iterator[Test]:
        yield from [Test(field=i) for i in range(5)]

    assert list(get_dataclasses()) == [Test(field=i) for i in range(5)]
    assert list(get_dataclasses()) == [Test(field=i) for i in range(5)]


def test_types(tmp_path):
    tdir = Path(tmp_path)

    from dataclasses import dataclass
    @dataclass
    class Test:
        an_int : int
        a_bool : bool
        a_float: float
        a_str  : str
        a_dt   : datetime
        a_date : date

    tz = pytz.timezone('Europe/Berlin')
    obj = Test(
        an_int =1123,
        a_bool =True,
        a_float=3.131,
        a_str  ='abac',
        a_dt   =datetime.now(tz=tz),
        a_date =datetime.now().replace(year=2000).date(),
    )

    assert len(obj.__dict__) == len(PRIMITIVES) # precondition

    @cachew(tdir / 'cache')
    def get() -> Iterator[Test]:
        yield obj

    assert list(get()) == [obj]
    assert list(get()) == [obj]


# TODO if I do perf tests, look at this https://docs.sqlalchemy.org/en/13/_modules/examples/performance/large_resultsets.html
# TODO should be possible to iterate anonymous tuples too? or just sequences of primitive types?

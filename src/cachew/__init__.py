from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

__author__    = "Dima Gerasimov"
__copyright__ = "Dima Gerasimov"
__license__   = "mit"


import functools
import logging
from itertools import chain, islice
import string
from datetime import datetime
import tempfile
from pathlib import Path
from random import Random
import sys
from typing import (Any, Callable, Iterator, List, NamedTuple, Optional, Tuple,
                    Type, Union, TypeVar, Generic, Sequence, Iterable, Set)

import sqlalchemy # type: ignore
from sqlalchemy import Column, Table, event


if sys.version_info[1] < 7:
    from .compat import fromisoformat
else:
    fromisoformat = datetime.fromisoformat


# in case of changes in the way cachew stores data, this should be changed to discard old caches
CACHEW_FORMAT = 1

def get_logger() -> logging.Logger:
    return logging.getLogger('cachew')


T = TypeVar('T')
def ichunks(l: Iterable[T], n: int) -> Iterator[List[T]]:
    it: Iterator[T] = iter(l)
    while True:
        chunk: List[T] = list(islice(it, 0, n))
        if len(chunk) == 0:
            break
        yield chunk


class IsoDateTime(sqlalchemy.TypeDecorator):
    # in theory could use something more effecient? e.g. blob for encoded datetime and tz?
    # but practically, the difference seems to be pretty small, so perhaps fine for now
    impl = sqlalchemy.String

    # TODO optional?
    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[str]:
        if value is None:
            return None
        return value.isoformat()

    def process_result_value(self, value: Optional[str], dialect) -> Optional[datetime]:
        if value is None:
            return None
        return fromisoformat(value)


PRIMITIVES = {
    str     : sqlalchemy.String,
    float   : sqlalchemy.Float,
    int     : sqlalchemy.Integer,
    bool    : sqlalchemy.Boolean,
    datetime: IsoDateTime,
}


Types = Union[
    Type[str],
    Type[int],
    Type[float],
    Type[bool],
    Type[NamedTuple],
]


def is_primitive(cls) -> bool:
    return cls in PRIMITIVES


# https://stackoverflow.com/a/2166841/706389
def is_namedtuple(t):
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


class CachewException(RuntimeError):
    pass


def strip_optional(cls):
    if getattr(cls, '__origin__', None) == Union:
        # handles Optional
        elems = cls.__args__
        elems = [e for e in elems if e != type(None)]
        if len(elems) == 1:
            nonopt = elems[0] # meh
            return (nonopt, True)
        else:
            raise CachewException(f'{cls} is unsupported!')
    return (cls, False)




class NTBinder(NamedTuple):
    name     : Optional[str] # None means toplevel
    type_    : Types
    span     : int # TODO not sure if span should include optional col?
    primitive: bool
    optional : bool
    fields   : Sequence[Any] # mypy can't handle cyclic definition at this point :(

    @staticmethod
    def make(tp, name: Optional[str]=None) -> 'NTBinder':
        tp, optional = strip_optional(tp)
        primitive = is_primitive(tp)
        if primitive:
            assert name is not None # TODO too paranoid?
        fields: Tuple[Any, ...]
        if primitive:
            fields = ()
            span = 1
        if not primitive:
            fields = tuple(NTBinder.make(tp=ann, name=fname) for fname, ann in tp.__annotations__.items())
            span = sum(f.span for f in fields) + (1 if optional else 0)
        return NTBinder(
            name=name,
            type_=tp,
            span=span,
            primitive=primitive,
            optional=optional,
            fields=fields,
        )

    @property
    def columns(self) -> List[Column]:
        return list(self.iter_columns())

    def to_row(self, obj):
        if self.primitive:
            yield obj
        else:
            if self.optional:
                is_none = obj is None
                yield is_none
            else:
                is_none = False; assert obj is not None # TODO hmm, that last assert is not very symmetric...

            if is_none:
                for _ in range(self.span - 1):
                    yield None
            else:
                yield from chain.from_iterable(
                    f.to_row(getattr(obj, f.name))
                    for f in self.fields
                )

    def from_row(self, row_iter):
        if self.primitive:
            return next(row_iter)
        else:
            if self.optional:
                is_none = next(row_iter)
            else:
                is_none = False

            if is_none:
                for _ in range(self.span - 1):
                    x = next(row_iter); assert x is None  # huh. assert is kinda opposite of producing value
                return None
            else:
                return self.type_(*(
                    f.from_row(row_iter)
                    for f in self.fields
                ))


    # TODO not sure if we want to allow optionals on top level?
    def iter_columns(self) -> Iterator[Column]:
        used_names: Set[str] = set()
        def col(name: str, tp) -> Column:
            while name in used_names:
                name = '_' + name
            used_names.add(name)
            return Column(name, tp)


        if self.primitive:
            assert self.name is not None
            yield col(self.name, PRIMITIVES[self.type_])
        else:
            prefix = '' if self.name is None else self.name + '_'
            if self.optional:
                yield col(f'_{prefix}is_null', sqlalchemy.Boolean)
            for f in self.fields:
                for c in f.iter_columns():
                    yield col(f'{prefix}{c.name}', c.type)


    def __str__(self):
        lines = ['  ' * level + str(x.name) + ('?' if x.optional else '') + ' '  + str(x.span) for level, x in self.iterxxx()]
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)

    def iterxxx(self, level=0):
        yield (level, self)
        for f in self.fields:
            yield from f.iterxxx(level=level + 1)


NT = TypeVar('NT')
# sadly, bound=NamedTuple is not working yet in mypy
# https://github.com/python/mypy/issues/685


class DbBinder(Generic[NT]):
    # ugh. Generic has cls as argument and it conflicts..
    def __init__(self, cls_: Type[NT]) -> None:
        self.cls = cls_
        self.nt_binder = NTBinder.make(self.cls)

    def __hash__(self):
        return hash(self.cls)

    def __eq__(self, o):
        return self.cls == o.cls

    @property
    def db_columns(self) -> List[Column]:
        return self.nt_binder.columns

    def to_row(self, obj: NT) -> Tuple[Any, ...]:
        return tuple(self.nt_binder.to_row(obj))

    def from_row(self, row: Iterable[Any]) -> NT:
        riter = iter(row)
        res = self.nt_binder.from_row(riter)
        remaining = list(islice(riter, 0, 1))
        assert len(remaining) == 0
        return res


# TODO better name to represent what it means?
SourceHash = str


# TODO give a better name
class DbWrapper:
    def __init__(self, db_path: Path, cls) -> None:
        from sqlalchemy.interfaces import PoolListener # type: ignore
        # TODO ugh. not much faster...
        class MyListener(PoolListener):
            def connect(self, dbapi_con, con_record):
                pass
                # eh. doesn't seem to help much..
                # dbapi_con.execute('PRAGMA journal_mode=MEMORY')
                # dbapi_con.execute('PRAGMA synchronous=OFF')


        self.engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        # self.db = sqlalchemy.create_engine(f'sqlite:///{db_path}', listeners=[MyListener()])
        self.connection = self.engine.connect() # TODO do I need to tear anything down??

        """
        Erm... this is pretty confusing.
        https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#transaction-isolation-level

        Somehow without this thing sqlalchemy logs BEGIN (implicit) instead of BEGIN TRANSACTION which actually works in sqlite...

        Judging by sqlalchemy/dialects/sqlite/base.py, looks like some sort of python sqlite driver problem??
        """
        @event.listens_for(self.connection, "begin")
        def do_begin(conn):
            conn.execute("BEGIN")


        self.meta = sqlalchemy.MetaData(self.connection)
        self.table_hash = Table('hash', self.meta, Column('value', sqlalchemy.String))
        self.table_hash.create(self.connection, checkfirst=True)

        self.binder = DbBinder(cls)
        self.table_data = Table('table', self.meta, *self.binder.db_columns)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

HashF = Callable[..., SourceHash]


def default_hash(*args, **kwargs) -> SourceHash:
    return str(args + tuple(sorted(kwargs.items()))) # good enough??


# TODO give it as an example in docs
def mtime_hash(path: Path, *args, **kwargs) -> SourceHash:
    mt = path.stat().st_mtime
    return default_hash(f'{path}.{mt}', *args, **kwargs)


Failure = str

def infer_type(func) -> Union[Failure, Type[Any]]:
    """
    >>> from typing import Collection, NamedTuple
    >>> class Person(NamedTuple):
    ...     name: str
    ...     age: int
    >>> def person_provider() -> Collection[Person]:
    ...     return []
    >>> infer_type(person_provider)
    <class 'cachew.Person'>
    """
    rtype = getattr(func, '__annotations__', {}).get('return', None)
    if rtype is None:
        # TODO mm. if
        return f"no return type annotation on {func}"

    def bail(reason):
        return f"can't infer type from {rtype}: " + reason

    # need to get erased type, otherwise subclass check would fail
    if not hasattr(rtype, '__origin__'):
        return bail("expected __origin__")
    if not issubclass(rtype.__origin__, Iterable):
        return bail("not subclassing Iterable")

    args = getattr(rtype, '__args__', None)
    if args is None:
        return bail("has no __args__")
    if len(args) != 1:
        return bail(f"wrong number of __args__: {args}")
    arg = args[0]
    if not is_namedtuple(arg):
        return bail(f"{arg} is not NamedTuple")
    return arg

# https://stackoverflow.com/questions/653368/how-to-create-a-python-decorator-that-can-be-used-either-with-or-without-paramet
def doublewrap(f):
    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)
    return new_dec


PathIsh = Union[Path, str]
PathProvider = Union[PathIsh, Callable[..., PathIsh]]


@doublewrap
def cachew(func=None, db_path: Optional[PathProvider]=None, cls=None, hashf: HashF=default_hash, chunk_by=10000, logger=None): # TODO what's a reasonable default?):
    # [[[cog
    # import cog
    # lines = open('README.org').readlines()
    # l = lines.index('#+BEGIN_SRC python\n')
    # r = lines.index('#+END_SRC\n')
    # src = lines[l + 1: r]
    # cog.outl("'''")
    # for line in src:
    #     cog.out(line)
    # cog.outl("'''")
    # ]]]
    '''
    >>> from typing import Collection, NamedTuple
    >>> from timeit import Timer
    >>> class Person(NamedTuple):
    ...     name: str
    ...     age: int
    >>> @cachew
    ... def person_provider() -> Iterator[Person]:
    ...     for i in range(5):
    ...         import time; time.sleep(1) # simulate slow IO
    ...         yield Person(name=str(i), age=20 + i)
    >>> list(person_provider()) # that should take about 5 seconds on first run
    [Person(name='0', age=20), Person(name='1', age=21), Person(name='2', age=22), Person(name='3', age=23), Person(name='4', age=24)]
    >>> res = Timer(lambda: list(person_provider())).timeit(number=1) # second run is cached, so should take less time
    >>> assert res < 0.1
    >>> print(f"took {res} seconds to query cached items")
    took ... seconds to query cached items
    '''
    # [[[end]]]

    # func is optional just to make pylint happy https://github.com/PyCQA/pylint/issues/259
    assert func is not None

    if logger is None:
        logger = get_logger()

    if db_path is None:
        td = Path(tempfile.gettempdir()) / 'cachew'
        td.mkdir(parents=True, exist_ok=True)
        db_path = td / func.__qualname__ # TODO sanitize?
        logger.info('No db_path specified, using %s as implicit cache', db_path)

    inferred = infer_type(func)
    if isinstance(inferred, Failure):
        msg = f"failed to infer cache type: {inferred}"
        if cls is None:
            raise CachewException(msg)
        else:
            # it's ok, assuming user knows better
            logger.debug(msg)
    else:
        if cls is None:
            logger.debug("using inferred type %s", inferred)
            cls = inferred
        else:
            if cls != inferred:
                logger.warning("inferred type %s mismatches specified type %s", inferred, cls)
                # TODO not sure if should be more serious error...
    assert is_namedtuple(cls)

    def composite_hash(*args, **kwargs) -> SourceHash:
        return f'cachew: {CACHEW_FORMAT}, schema: {cls._field_types}, hash: {hashf(*args, **kwargs)}'

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if callable(db_path): # TODO test this..
            dbp = Path(db_path(*args, **kwargs))
        else:
            dbp = Path(db_path)

        logger.debug('using %s for db cache', dbp)

        if not dbp.parent.exists():
            raise CachewException(f"{dbp.parent} doesn't exist") # otherwise, sqlite error is quite cryptic

        # TODO make sure we have exclusive write lock
        with DbWrapper(dbp, cls) as db:
            binder = db.binder
            conn = db.connection
            values_table = db.table_data

            prev_hashes = conn.execute(db.table_hash.select()).fetchall()
            # TODO .order_by('rowid') ?
            if len(prev_hashes) > 1:
                raise CachewException(f'Multiple hashes! {prev_hashes}')

            prev_hash: Optional[SourceHash]
            if len(prev_hashes) == 0:
                prev_hash = None
            else:
                prev_hash = prev_hashes[0][0] # TODO ugh, returns a tuple...

            logger.debug('old hash: %s', prev_hash)
            h = composite_hash(*args, **kwargs); assert h is not None # just in case
            logger.debug('new hash: %s', h)

            with conn.begin() as transaction:
                if h == prev_hash:
                    # TODO not sure if this needs to be in transaction
                    logger.debug('hash matched: loading from cache')
                    rows = conn.execute(values_table.select())
                    for row in rows:
                        yield binder.from_row(row)
                else:
                    logger.debug('hash mismatch: computing data and writing to db')

                    # drop and create to incorporate schema changes
                    values_table.drop(conn, checkfirst=True)
                    values_table.create(conn)

                    datas = func(*args, **kwargs)

                    for chunk in ichunks(datas, n=chunk_by):
                        bound = [tuple(binder.to_row(c)) for c in chunk]
                        # pylint: disable=no-value-for-parameter
                        conn.execute(values_table.insert().values(bound))
                        yield from chunk

                    # TODO insert and replace instead?

                    # pylint: disable=no-value-for-parameter
                    conn.execute(db.table_hash.delete())
                    # pylint: disable=no-value-for-parameter
                    conn.execute(db.table_hash.insert().values([{'value': h}]))
    return wrapper


# TODO mypy is unhappy about inline namedtuples.. perhaps should open an issue
class TE(NamedTuple):
    dt: datetime
    value: float
    flag: bool


def test_simple(tmp_path):
    import pytz
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
    COUNT = 1000000
    logger = get_logger()

    tdir = Path(tmp_path)
    src = tdir / 'source'
    src.touch()

    @cachew(db_path=lambda path: tdir / (path.name + '.cache'), cls=TE2)
    def _iter_data(path: Path):
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
    tdir = Path(tmp_path)

    @cachew(db_path=tdir / 'cache')
    def data() -> Iterator[BB]:
        yield BB(xx=1, yy=2)
        yield BB(xx=3, yy=4)

    assert len(list(data())) == 2
    assert len(list(data())) == 2


def test_return_type_mismatch(tmp_path):
    tdir = Path(tmp_path)
    # even though user got invalid type annotation here, they specified correct type, and it's the one that should be used
    @cachew(db_path=tdir / 'cache2', cls=AA)
    def data2() -> List[BB]:
        return [ # type: ignore
            AA(value=1, b=None, value2=123),
        ]

    # TODO hmm, this is kinda a downside that it always returns
    # could preserve the original return type, but too much trouble for now

    assert list(data2()) == [AA(value=1, b=None, value2=123)]


def test_return_type_none(tmp_path):
    tdir = Path(tmp_path)
    import pytest # type: ignore
    with pytest.raises(CachewException):
        @cachew(db_path=tdir / 'cache')
        def data():
            return []


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

    @cachew(db_path=tdir / 'cache', cls=AA)
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

    @cachew(db_path=tdir / 'cache', cls=BB)
    def get_data():
        return [b]

    assert list(get_data()) == [b]

    # TODO make type part of key?
    b2 = BBv2(xx=3, yy=4, zz=5.0)
    @cachew(db_path=tdir / 'cache', cls=BBv2)
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

    @cachew(db_path=tdir / 'cache', cls=BB, chunk_by=1)
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
    b = DbBinder(Person)

    cols = b.db_columns

    # TODO that could be a doctest showing actual database schema
    assert [(c.name, type(c.type)) for c in cols] == [
        ('name'        , sqlalchemy.String),
        ('secondname'  , sqlalchemy.String),
        ('age'         , sqlalchemy.Integer),

        ('_job_is_null', sqlalchemy.Boolean),
        ('job_company' , sqlalchemy.String),
        ('job_title'   , sqlalchemy.String),
    ]


def test_unique(tmp_path):
    tdir = Path(tmp_path)

    class Breaky(NamedTuple):
        job_title: int
        job: Optional[Job]

    assert [c.name for c in DbBinder(Breaky).db_columns] == [
        'job_title',
        '_job_is_null',
        'job_company',
        '_job_title',
    ]

    b = Breaky(
        job_title=123,
        job=Job(company='123', title='whatever'),
    )
    @cachew(db_path=tdir / 'cache')
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

    @cachew(db_path=cache_file, cls=Person)
    def get_people_data() -> Iterator[Person]:
        yield from make_people_data(count=N)


    list(get_people_data())
    print(f"Cache db size for {N} entries: estimated size {one * N // 1024} Kb, actual size {cache_file.stat().st_size // 1024} Kb;")



# TODO if I do perf tests, look at this https://docs.sqlalchemy.org/en/13/_modules/examples/performance/large_resultsets.html
# TODO should be possible to iterate anonymous tuples too? or just sequences of primitive types?

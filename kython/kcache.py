import functools
import logging
from itertools import chain
import string
from datetime import datetime
from pathlib import Path
from random import Random
from typing import (Any, Callable, Iterator, List, NamedTuple, Optional, Tuple,
                    Type, Union, TypeVar, Generic, Sequence)

import sqlalchemy # type: ignore
from sqlalchemy import Column, Table, event # type: ignore
from sqlalchemy.sql import text # type: ignore

from kython.klogging import setup_logzero
from kython.ktyping import PathIsh
from kython.py37 import fromisoformat


def get_kcache_logger():
    return logging.getLogger('kcache')


# TODO move to some common thing?
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

_tmap = {
    str: sqlalchemy.String,
    float: sqlalchemy.Float,
    int: sqlalchemy.Integer,
    datetime: IsoDateTime,
}


def is_primitive(cls) -> bool:
    return cls in _tmap


def _map_type(cls):
    r = _tmap.get(cls, None)
    if r is not None:
        return r
    raise RuntimeError(f'Unexpected type {cls}')

# https://stackoverflow.com/a/2166841/706389
@functools.lru_cache()
# cache here gives a 25% speedup
def isnamedtuple(t):
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


# TODO not sure if needs cache?
# TODO use nullable=True from sqlalchemy?
def strip_optional(cls):
    if getattr(cls, '__origin__', None) == Union:
        # handles Optional
        elems = cls.__args__
        elems = [e for e in elems if e != type(None)]
        if len(elems) == 1:
            nonopt = elems[0] # meh
            return (nonopt, True)
        else:
            raise RuntimeError(f'{cls} is unsupported!')
    return (cls, False)


# TODO shit. doesn't really help...
@functools.lru_cache(maxsize=None) # TODO kinda arbitrary..
def get_namedtuple_schema(cls):
    # caching is_namedtuple doesn't seem to give a major speedup here, but whatever..
    def gen():
        # fuck python not allowing multiline expressions..
        for name, ann in cls.__annotations__.items():
            ann, is_opt = strip_optional(ann)
            # caching try_remove_optional is a massive speedup though
            yield name, ann, is_opt
    return tuple(gen())


NT = TypeVar('NT', bound=NamedTuple)



from kython import cproperty
# TODO not sure...
# TODO iterator over fields??

# TODO FIXME should be possible to iterate anonymous tuples too? or just sequences of primitive types?

class ZZZ(NamedTuple):
    name     : Optional[str] # None means toplevel
    type_    : Type[Any] # TODO
    primitive: bool
    optional : bool
    fields   : Sequence[Any] # TODO FIXME recursive type?

    # TODO not sure if span should include optional col?
    @cproperty
    def span(self) -> int:
        if self.primitive:
            return 1
        return sum(f.span for f in self.fields) + (1 if self.optional else 0)

    @staticmethod
    def make(tp, name: Optional[str]=None):
        tp, optional = strip_optional(tp) # TODO eh?
        prim = is_primitive(tp)
        if prim:
            assert name is not None # TODO too paranoid?
        fields = ()
        if not prim:
            fields = tuple(ZZZ.make(tp=ann, name=fname) for fname, ann in tp.__annotations__.items())
        return ZZZ(
            name=name,
            type_=tp,
            primitive=is_primitive(tp),
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


    # TODO FIXME make sure col names are unique
    # TODO not sure if we want to allow optionals on top level?
    def iter_columns(self) -> Iterator[Column]:
        if self.primitive:
            yield Column(self.name, _map_type(self.type_))
        else:
            if self.optional:
                yield Column(f'_{self.name}_is_null', sqlalchemy.Boolean)
            for f in self.fields:
                yield from f.iter_columns()


    def __str__(self):
        lines = ['  ' * level + str(x.name) + ('?' if x.optional else '') + ' '  + str(x.span) for level, x in self.iterxxx()]
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)

    def iterxxx(self, level=0):
        yield (level, self)
        for f in self.fields:
            yield from f.iterxxx(level=level + 1)


# TODO just make it generic?
class Binder(Generic[NT]):
    def __init__(self, clazz: Type[NT]) -> None:
        self.clazz = clazz
        self.zzz = ZZZ.make(self.clazz)

    def __hash__(self):
        return hash(self.clazz)

    def __eq__(self, o):
        return self.clazz == o.clazz

    @property
    def columns(self) -> List[Column]:
        return self.zzz.columns

    def to_row(self, obj: NT) -> Tuple[Any, ...]:
        return tuple(self.zzz.to_row(obj))

    def from_row(self, row) -> NT:
        return self.zzz.from_row(iter(row)) # TODO assert consumed?


# TODO better name to represent what it means?
SourceHash = str


# TODO give a better name
class DbWrapper:
    def __init__(self, db_path: Path, type_) -> None:
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

        self.binder = Binder(clazz=type_)
        self.table_data = Table('table', self.meta, *self.binder.columns)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()


# TODO what if we want dynamic path??

# TODO ugh. there should be a nicer way to wrap that...
# TODO mypy return types
# TODO FIXME pathish thing
PathProvider = Union[PathIsh, Callable[..., PathIsh]]
HashF = Callable[..., SourceHash]


def default_hashf(*args, **kwargs) -> SourceHash:
    return str(args + tuple(sorted(kwargs.items()))) # good enough??


def make_dbcache(db_path: PathProvider, type_, hashf: HashF=default_hashf, chunk_by=10000, logger=None): # TODO what's a reasonable default?
    def chash(*args, **kwargs) -> SourceHash:
        return str(type_._field_types) + hashf(*args, **kwargs)

    if logger is None:
        logger = get_kcache_logger()

    def dec(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if callable(db_path): # TODO test this..
                dbp = Path(db_path(*args, **kwargs))
            else:
                dbp = Path(db_path)

            logger.debug('using %s for db cache', dbp)

            if not dbp.parent.exists():
                raise RuntimeError(f"{dbp.parent} doesn't exist") # otherwise, sqlite error is quite cryptic

            # TODO FIXME make sure we have exclusive write lock
            with DbWrapper(dbp, type_) as db:
                binder = db.binder
                conn = db.connection
                valuest = db.table_data

                prev_hashes = conn.execute(db.table_hash.select()).fetchall()
                # TODO .order_by('rowid') ?
                if len(prev_hashes) > 1:
                    raise RuntimeError(f'Multiple hashes! {prev_hashes}')

                prev_hash: Optional[SourceHash]
                if len(prev_hashes) == 0:
                    prev_hash = None
                else:
                    prev_hash = prev_hashes[0][0] # TODO ugh, returns a tuple...
                logger.debug('previous hash: %s', prev_hash)

                h = chash(*args, **kwargs)
                logger.debug('current hash: %s', h)
                assert h is not None # just in case

                with conn.begin() as transaction:
                    if h == prev_hash:
                        # TODO not sure if this needs to be in transaction
                        logger.debug('hash matched: loading from cache')
                        rows = conn.execute(valuest.select())
                        for row in rows:
                            yield binder.from_row(row)
                    else:
                        logger.debug('hash mismatch: computing data and writing to db')

                        # drop and create to incorporate schema changes
                        valuest.drop(conn, checkfirst=True)
                        valuest.create(conn)

                        datas = func(*args, **kwargs)
                        from kython import ichunks

                        for chunk in ichunks(datas, n=chunk_by):
                            bound = [tuple(binder.to_row(c)) for c in chunk]
                            # logger.debug('inserting...')
                            # from sqlalchemy.sql import text
                            # nulls = ', '.join("(NULL)" for _ in bound)
                            # st = text("""INSERT INTO 'table' VALUES """ + nulls)
                            # engine.execute(st)
                            # shit. so manual operation is quite a bit faster??
                            # but we still want serialization :(
                            # ok, inserting gives noticeable lag
                            # thiere must be some obvious way to speed this up...
                            # pylint: disable=no-value-for-parameter
                            conn.execute(valuest.insert().values(bound))
                            # logger.debug('inserted...')
                            yield from chunk

                        # TODO FIXME insert and replace instead

                        # pylint: disable=no-value-for-parameter
                        conn.execute(db.table_hash.delete())
                        # pylint: disable=no-value-for-parameter
                        conn.execute(db.table_hash.insert().values([{'value': h}]))
        return wrapper

    return dec


# TODO give it as an example in docs
def mtime_hash(path: Path) -> SourceHash:
    # TODO hopefully float are ok here?
    mt = path.stat().st_mtime
    return f'{path}.{mt}'

# TODO mypy is unhappy about inline namedtuples.. perhaps should open an issue
class TE(NamedTuple):
    dt: datetime
    value: float

def test_dbcache(tmp_path):
    from kython.klogging import setup_logzero
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)

    import pytz
    mad = pytz.timezone('Europe/Madrid')
    utc = pytz.utc


    tdir = Path(tmp_path)
    src = tdir / 'source'
    src.write_text('0')

    db_path = tdir / 'db.sqlite'
    dbcache = make_dbcache(db_path, hashf=mtime_hash, type_=TE)

    entities = [
        TE(dt=utc.localize(datetime(year=1991, month=5, day=3, minute=1)), value=123.43242),
        TE(dt=mad.localize(datetime(year=1997, month=7, day=4, second=5)), value=9842.4234),
    ]

    accesses = 0
    @dbcache
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
def test_dbcache_many(tmp_path):
    COUNT = 1000000
    # 100K: about  3.0 seconds
    # 500K: about 15.5 seconds
    #   1M: about 29.4 seconds
    from kython.klogging import setup_logzero
    logger = get_kcache_logger()
    setup_logzero(logger, level=logging.DEBUG)

    tdir = Path(tmp_path)
    src = tdir / 'source'
    src.touch()

    dbcache = make_dbcache(db_path=lambda path: tdir / (path.name + '.cache'), type_=TE2)

    @dbcache
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


def test_dbcache_nested(tmp_path):
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)
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

    dbcache = make_dbcache(db_path=tdir / 'cache', type_=AA)

    @dbcache
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
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)
    tdir = Path(tmp_path)
    b = BB(xx=2, yy=3)

    dbcache = make_dbcache(db_path=tdir / 'cache', type_=BB) # TODO could deduce type automatically from annotations??
    @dbcache
    def get_data():
        return [b]

    assert list(get_data()) == [b]

    # TODO make type part of key?
    b2 = BBv2(xx=3, yy=4, zz=5.0)
    dbcache2 = make_dbcache(db_path=tdir / 'cache', type_=BBv2)
    @dbcache2
    def get_data_v2():
        return [b2]

    assert list(get_data_v2()) == [b2]

def test_transaction(tmp_path):
    """
    Should keep old cache and not leave it in some broken state in case of errors
    """
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    tdir = Path(tmp_path)

    dbcache = make_dbcache(db_path=tdir / 'cache', type_=BB, chunk_by=1)
    @dbcache
    def get_data(version: int):
        for i in range(3):
            yield BB(xx=2, yy=i)
            if version == 2:
                raise RuntimeError

    exp = [BB(xx=2, yy=0), BB(xx=2, yy=1), BB(xx=2, yy=2)]
    assert list(get_data(1)) == exp
    assert list(get_data(1)) == exp

    # TODO test that hash is unchanged?
    import pytest # type: ignore
    with pytest.raises(RuntimeError):
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


def test_namedtuple_schema():
    schema = get_namedtuple_schema(Person)
    assert schema == (
        ('name'      , str, False),
        ('secondname', str, False),
        ('age'       , int, False),
        ('job'       , Job, True),
    )


def test_binder():
    b = Binder(clazz=Person)
    cols = b.columns

    # TODO that could be a doctest showing actual database schema
    assert [(c.name, type(c.type)) for c in cols] == [
        ('name'        , sqlalchemy.String),
        ('secondname'  , sqlalchemy.String),
        ('age'         , sqlalchemy.Integer),

        # TODO FIXME need to prevent name conflicts with origina objects names
        ('_job_is_null', sqlalchemy.Boolean),
        ('company'     , sqlalchemy.String),
        ('title'       , sqlalchemy.String),
    ]



def test_stats(tmp_path):
    tdir = Path(tmp_path)

    cache_file = tdir / 'cache'

    # 4 + things are string lengths
    one = (4 + 5) + (4 + 10) + 4 + (4 + 12 + 4 + 8)
    N = 10000

    dbcache = make_dbcache(db_path=cache_file, type_=Person)
    @dbcache
    def get_people_data() -> Iterator[Person]:
        yield from make_people_data(count=N)


    list(get_people_data())
    print(f"Cache db size for {N} entries: estimate size {one * N // 1024} Kb, actual size {cache_file.stat().st_size // 1024} Kb;")



# TODO if I do perf tests, look at this https://docs.sqlalchemy.org/en/13/_modules/examples/performance/large_resultsets.html


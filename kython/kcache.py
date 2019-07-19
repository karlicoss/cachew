from datetime import datetime
from typing import Optional, Type, NamedTuple, Union, Callable, List
from pathlib import Path
import functools
import logging

import sqlalchemy # type: ignore
from sqlalchemy import Column, Table # type: ignore

from kython.ktyping import PathIsh
from kython.py37 import fromisoformat
from kython.klogging import setup_logzero


def get_kcache_logger():
    return logging.getLogger('kcache')

# TODO FIXME REMOVE THIS
# class UUU(NamedTuple):
#     xx: int
#     yy: int
# class TE2(NamedTuple):
#     value: int
#     uuu: UUU
#     value2: int

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
def try_remove_optional(cls):
    if getattr(cls, '__origin__', None) == Union:
        # handles Optional
        elems = cls.__args__
        elems = [e for e in elems if e != type(None)]
        if len(elems) == 1:
            return elems[0] # meh..
    return cls

# TODO ugh. couldn't get this to work
# from typing import TypeVar
# NT = TypeVar('NT', bound=NamedTuple)

class Binder:
    def __init__(self, clazz) -> None:
        self.clazz = clazz

    @property
    def columns(self) -> List[Column]:
        def helper(cls) -> List[Column]:
            res = []
            for name, ann, is_nt in self._namedtuple_schema(cls):
                # TODO def cache this schema, especially considering try_remove_optional
                # TODO just remove optionals here? sqlite doesn't really respect them anyway IIRC
                # TODO might need optional handling as well...
                # TODO add optional to test
                if is_nt:
                    res.extend(helper(ann)) # TODO FIXME make sure col names are unique
                else:
                    res.append(Column(name, _map_type(ann)))
            return res
        return helper(self.clazz)

    def to_row(self, obj):
        def helper(cls, o):
            for name, ann, is_nt in self._namedtuple_schema(cls):
                v = getattr(o, name)
                if is_nt:
                    if v is None:
                        raise RuntimeError("can't handle None tuples at this point :(")
                    yield from helper(ann, v)
                else:
                    yield v
        # meh..
        return tuple(helper(self.clazz, obj))

    def __hash__(self):
        return hash(self.clazz)

    def __eq__(self, o):
        return self.clazz == o.clazz

    # TODO shit. doesn't really help...
    @functools.lru_cache() # TODO kinda arbitrary..
    def _namedtuple_schema(self, cls):
        # caching is_namedtuple doesn't seem to give a major speedup here, but whatever..
        def gen():
            # fuck python not allowing multiline expressions..
            for name, ann in cls.__annotations__.items():
                ann = try_remove_optional(ann)
                # caching try_remove_optional is a massive speedup though
                yield name, ann, isnamedtuple(ann)
        return tuple(gen())

    # TODO FIXME shit, need to be careful during serializing if the namedtuple itself is None... not even sure how to distinguish if we are flattening :(
    # TODO for now just forbid None in runtime
    # might need extra entry....
    def from_row(self, row):
        # uuu = UUU(row[1], row[2])
        # te2 = TE2(row[0], uuu, row[3])
        # return te2
        # huh! ok, without deserializing it's almost instantaneous..
        pos = 0
        def helper(cls):
            nonlocal pos
            vals = []
            for name, ann, is_nt in self._namedtuple_schema(cls):
                if is_nt:
                    val = helper(ann)
                else:
                    val = row[pos]
                    pos += 1
                vals.append(val)
            return cls(*vals)
        return helper(self.clazz)


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


        self.db = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        # self.db = sqlalchemy.create_engine(f'sqlite:///{db_path}', listeners=[MyListener()])
        self.engine = self.db.connect() # TODO do I need to tear anything down??

        self.meta = sqlalchemy.MetaData(self.engine)
        self.table_hash = Table('hash' , self.meta, Column('value', sqlalchemy.String))
        self.table_hash.create(self.engine, checkfirst=True)

        self.binder = Binder(clazz=type_)
        self.table_data = Table('table', self.meta, *self.binder.columns)


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
            if callable(db_path):
                dbp = Path(db_path(*args, **kwargs))
            else:
                dbp = Path(db_path)

            logger.debug('using %s for db cache', dbp)

            if not dbp.parent.exists():
                raise RuntimeError(f"{dbp.parent} doesn't exist") # otherwise, sqlite error is quite cryptic

            # TODO FIXME make sure we have exclusive write lock
            alala = DbWrapper(dbp, type_)
            binder = alala.binder
            engine = alala.engine

            prev_hashes = engine.execute(alala.table_hash.select()).fetchall()
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

            with engine.begin() as transaction:
                if h == prev_hash:
                    logger.debug('hash matched: loading from cache')
                    rows = engine.execute(alala.table_data.select())
                    for row in rows:
                        yield binder.from_row(row)
                else:
                    logger.debug('hash mismatch: computing data and writing to db')
                    # TODO mm, not so sure how transactional it all actually is... test it?
                    alala.table_data.drop(engine, checkfirst=True)
                    alala.table_data.create(engine)

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
                        engine.execute(alala.table_data.insert().values(bound))
                        # logger.debug('inserted...')
                        yield from chunk

                    # TODO FIXME insert and replace instead

                    # pylint: disable=no-value-for-parameter
                    engine.execute(alala.table_hash.delete())
                    # pylint: disable=no-value-for-parameter
                    engine.execute(alala.table_hash.insert().values([{'value': h}]))
        return wrapper

    # TODO FIXME engine is leaking??
    return dec


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

    d = AA(
        value=1,
        b=BB(xx=2, yy=3),
        value2=4,
    )
    def data():
        yield d

    dbcache = make_dbcache(db_path=tdir / 'cache', type_=AA)

    @dbcache
    def get_data():
        yield from data()

    assert list(get_data()) == [d]
    assert list(get_data()) == [d]


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









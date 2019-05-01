from datetime import datetime
from typing import Optional, Type, NamedTuple, Union, Callable, List
from pathlib import Path
import functools
import logging

import sqlalchemy # type: ignore
from sqlalchemy import Column, Table # type: ignore

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


def _map_type(cls):
    tmap = {
        str: sqlalchemy.String,
        float: sqlalchemy.Float,
        int: sqlalchemy.Integer,
        datetime: IsoDateTime,
    }
    r = tmap.get(cls, None)
    if r is not None:
        return r
    raise RuntimeError(f'Unexpected type {cls}')

# https://stackoverflow.com/a/2166841/706389
def isnamedtuple(t):
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def try_remove_optional(cls):
    if getattr(cls, '__origin__', None) == Union:
        # handles Optional
        elems = cls.__args__
        elems = [e for e in elems if e != type(None)]
        if len(elems) == 1:
            return elems[0] # meh..
    return cls


class Binder:
    def __init__(self, clazz: Type[NamedTuple]) -> None: # TODO covariant?
        self.clazz = clazz

    @property
    def columns(self) -> List[Column]:
        def helper(cls: Type[NamedTuple], prefix='') -> List[Column]:
            res = []
            for name, ann in cls.__annotations__.items():
                ann = try_remove_optional(ann)
                # TODO def cache this schema, especially considering try_remove_optional
                # TODO just remove optionals here? sqlite doesn't really respect them anyway IIRC
                # TODO might need optional handling as well...
                # TODO add optional to test
                if isnamedtuple(ann):
                    res.extend(helper(ann)) # TODO FIXME make sure col names are unique
                else:
                    res.append(Column(name, _map_type(ann)))
            return res
        return helper(self.clazz)

    def to_row(self, obj):
        for k, v in obj._asdict().items():
            if isinstance(v, tuple):
                yield from self.to_row(v)
            else:
                yield v
                # meh..

    def from_row(self, row):
        pos = 0
        def helper(cls):
            nonlocal pos
            dct = {}
            for name, ann in cls.__annotations__.items(): # TODO cache if necessary? benchmark quickly
                ann = try_remove_optional(ann)
                if isnamedtuple(ann):
                    val = helper(ann)
                else:
                    val = row[pos]
                    pos += 1
                dct[name] = val
            return cls(**dct)
        return helper(self.clazz)

# TODO better name to represent what it means?
SourceHash = str


# TODO give a better name
class DbWrapper:
    def __init__(self, db_path: Path, type_) -> None:
        self.db = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        self.engine = self.db.connect() # TODO do I need to tear anything down??
        self.meta = sqlalchemy.MetaData(self.engine)
        self.table_hash = Table('hash' , self.meta, Column('value', sqlalchemy.String))

        self.binder = Binder(clazz=type_)
        self.table_data = Table('table', self.meta, *self.binder.columns)
        self.meta.create_all()


# TODO what if we want dynamic path??

# TODO ugh. there should be a nicer way to wrap that...
# TODO mypy return types
# TODO FIXME pathish thing
PathProvider = Union[PathIsh, Callable[..., PathIsh]]
HashF = Callable[..., SourceHash]


def default_hashf(*args, **kwargs) -> SourceHash:
    return str(args + tuple(sorted(kwargs.items()))) # good enough??


def make_dbcache(db_path: PathProvider, type_, hashf: HashF=default_hashf, chunk_by=10000, logger=None): # TODO what's a reasonable default?
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

            h = hashf(*args, **kwargs)
            logger.debug('current hash: %s', h)
            assert h is not None # just in case

            with engine.begin() as transaction:
                if h == prev_hash:
                    logger.debug('hash match: loading from cache')
                    rows = engine.execute(alala.table_data.select())
                    for row in rows:
                        yield binder.from_row(row)
                else:
                    logger.debug('hash mismatch: retrieving data and writing to db')
                    datas = func(*args, **kwargs)
                    engine.execute(alala.table_data.delete())
                    from kython import ichunks

                    for chunk in ichunks(datas, n=chunk_by):
                        bound = [tuple(binder.to_row(c)) for c in chunk]
                        engine.execute(alala.table_data.insert().values(bound))
                        yield from chunk

                    # TODO FIXME insert and replace instead
                    engine.execute(alala.table_hash.delete())
                    engine.execute(alala.table_hash.insert().values([{'value': h}]))
        return wrapper

    # TODO FIXME engine is leaking??
    return dec


def mtime_hash(path: Path) -> SourceHash:
    # TODO hopefully float are ok here?
    mt = path.stat().st_mtime
    return f'{path}.{mt}'


def test_dbcache(tmp_path):
    from kython.klogging import setup_logzero
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)

    import pytz
    mad = pytz.timezone('Europe/Madrid')
    utc = pytz.utc

    class TE(NamedTuple):
        dt: datetime
        value: float

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


def test_dbcache_many(tmp_path):
    from kython.klogging import setup_logzero
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)
    class TE2(NamedTuple):
        value: int
        value2: int

    tdir = Path(tmp_path)
    src = tdir / 'source'
    src.touch()

    dbcache = make_dbcache(db_path=lambda path: tdir / (path.name + '.cache'), hashf=mtime_hash, type_=TE2)

    @dbcache
    def _iter_data(path: Path):
        for i in range(100000):
            yield TE2(value=i, value2=i)

    def iter_data():
        return _iter_data(src)

    def ilen(it):
        ll = 0
        for _ in it:
            ll += 1
        return ll
    assert ilen(iter_data()) == 100000
    assert ilen(iter_data()) == 100000


def test_dbcache_nested(tmp_path):
    from kython.klogging import setup_logzero
    setup_logzero(get_kcache_logger(), level=logging.DEBUG)
    tdir = Path(tmp_path)

    class B(NamedTuple):
        xx: int
        yy: int

    class A(NamedTuple):
        value: int
        b: Optional[B]
        value2: int

    d = A(
        value=1,
        b=B(xx=2, yy=3),
        value2=4,
    )
    def data():
        yield d

    dbcache=make_dbcache(db_path=tdir / 'cache', type_=A)

    @dbcache
    def get_data():
        yield from data()

    assert list(get_data()) == [d]
    assert list(get_data()) == [d]

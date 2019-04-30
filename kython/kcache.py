from datetime import datetime
from typing import Optional, Type, NamedTuple, Union
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
        datetime: IsoDateTime,
    }
    r = tmap.get(cls, None)
    if r is not None:
        return r


    if getattr(cls, '__origin__', None) == Union:
        elems = cls.__args__
        elems = [e for e in elems if e != type(None)]
        if len(elems) == 1:
            return _map_type(elems[0]) # meh..
    raise RuntimeError(f'Unexpected type {cls}')


def _make_schema(cls: Type[NamedTuple]): # TODO covariant?
    res = []
    for name, ann in cls.__annotations__.items():
        res.append(Column(name, _map_type(ann)))
    return res


# TODO better name to represent what it means?
SourceHash = str


# TODO give a better name
class DbWrapper:
    def __init__(self, db_path: Path, type_) -> None:
        self.db = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        self.engine = self.db.connect() # TODO do I need to tear anything down??
        self.meta = sqlalchemy.MetaData(self.engine)
        self.table_hash = Table('hash' , self.meta, Column('value', sqlalchemy.String))

        schema = _make_schema(type_)
        self.table_data = Table('table', self.meta, *schema)
        self.meta.create_all()


# TODO what if we want dynamic path??

# TODO ugh. there should be a nicer way to wrap that...
def make_dbcache(db_path: PathIsh, hashf, type_):
    logger = get_kcache_logger()
    def dec(func):
        @functools.wraps(func)
        def wrapper(key):
            # TODO FIXME make sure we have exclusive write lock
            alala = DbWrapper(Path(db_path), type_)
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

            h = hashf(key)
            logger.debug('current hash: %s', h)
            assert h is not None # just in case

            with engine.begin() as transaction:
                if h == prev_hash:
                    rows = engine.execute(alala.table_data.select()).fetchall()
                    return [type_(**row) for row in rows]
                else:
                    datas = func(key)
                    if len(datas) > 0:
                        # TODO warn if zero?
                        engine.execute(alala.table_data.delete())
                        engine.execute(alala.table_data.insert().values(datas)) # TODO chunks??

                    # TODO FIXME insert and replace instead
                    engine.execute(alala.table_hash.delete())
                    engine.execute(alala.table_hash.insert().values([{'value': h}]))
                    return datas
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
        return _get_data(src)

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









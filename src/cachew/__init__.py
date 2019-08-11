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
from datetime import datetime, date
import tempfile
from pathlib import Path
import sys
from typing import (Any, Callable, Iterator, List, NamedTuple, Optional, Tuple,
                    Type, Union, TypeVar, Generic, Sequence, Iterable, Set)
import dataclasses

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

    @property
    def python_type(self): return datetime

    def process_literal_param(self, value, dialect): raise NotImplementedError() # make pylint happy

    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[str]:
        if value is None:
            return None
        return value.isoformat()

    def process_result_value(self, value: Optional[str], dialect) -> Optional[datetime]:
        if value is None:
            return None
        return fromisoformat(value)


# a bit hacky, but works...
class IsoDate(IsoDateTime):
    impl = sqlalchemy.String

    @property
    def python_type(self): return date

    def process_literal_param(self, value, dialect): raise NotImplementedError() # make pylint happy

    def process_result_value(self, value: Optional[str], dialect) -> Optional[date]: # type: ignore
        res = super().process_result_value(value, dialect)
        if res is None:
            return None
        return res.date()


PRIMITIVES = {
    str     : sqlalchemy.String,
    float   : sqlalchemy.Float,
    int     : sqlalchemy.Integer,
    bool    : sqlalchemy.Boolean,
    datetime: IsoDateTime,
    date    : IsoDate,
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
def is_dataclassish(t):
    """
    >>> is_dataclassish(int)
    False
    >>> is_dataclassish(tuple)
    False
    >>> from typing import NamedTuple
    >>> class N(NamedTuple):
    ...     field: int
    >>> is_dataclassish(N)
    True
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class D:
    ...     field: str
    >>> is_dataclassish(D)
    True
    """
    if dataclasses.is_dataclass(t):
        return True
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    # pylint: disable=unidiomatic-typecheck
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


# release mode friendly assert
def kassert(x: bool) -> None:
    if x is False:
        raise AssertionError


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
            kassert(name is not None) # TODO too paranoid?
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
                is_none = False; kassert(obj is not None) # TODO hmm, that last assert is not very symmetric...

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
                    x = next(row_iter); kassert(x is None)  # huh. assert is kinda opposite of producing value
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
            if self.name is None: raise AssertionError
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
        kassert(len(remaining) == 0)
        return res


# TODO better name to represent what it means?
SourceHash = str


# TODO give a better name
class DbWrapper:
    def __init__(self, db_path: Path, cls) -> None:
        self.engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
        self.connection = self.engine.connect() # TODO do I need to tear anything down??

        """
        Erm... this is pretty confusing.
        https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#transaction-isolation-level

        Somehow without this thing sqlalchemy logs BEGIN (implicit) instead of BEGIN TRANSACTION which actually works in sqlite...

        Judging by sqlalchemy/dialects/sqlite/base.py, looks like some sort of python sqlite driver problem??
        """
        @event.listens_for(self.connection, "begin")
        # pylint: disable=unused-variable
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

# pylint: disable=too-many-return-statements
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
    if not is_dataclassish(arg):
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
# pylint: disable=too-many-arguments
def cachew(
        func=None,
        db_path: Optional[PathProvider]=None,
        cls=None,
        hashf: HashF=default_hash,
        chunk_by=10000,
        logger=None,
): # TODO what's a reasonable default?):
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
    >>> from typing import NamedTuple, Iterator
    >>> class Link(NamedTuple):
    ...     url : str
    ...     text: str
    ...
    >>> @cachew
    ... def extract_links(archive: str) -> Iterator[Link]:
    ...     for i in range(5):
    ...         import time; time.sleep(1) # simulate slow IO
    ...         yield Link(url=f'http://link{i}.org', text=f'text {i}')
    ...
    >>> list(extract_links(archive='wikipedia_20190830.zip')) # that should take about 5 seconds on first run
    [Link(url='http://link0.org', text='text 0'), Link(url='http://link1.org', text='text 1'), Link(url='http://link2.org', text='text 2'), Link(url='http://link3.org', text='text 3'), Link(url='http://link4.org', text='text 4')]

    >>> from timeit import Timer
    >>> res = Timer(lambda: list(extract_links(archive='wikipedia_20190830.zip'))).timeit(number=1) # second run is cached, so should take less time
    >>> print(f"took {int(res)} seconds to query cached items")
    took 0 seconds to query cached items
    '''
    # [[[end]]]

    # func is optional just to make pylint happy https://github.com/PyCQA/pylint/issues/259
    # kassert(func is not None)

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
    kassert(is_dataclassish(cls))

    return cachew_impl(
        func=func,
        db_path=db_path,
        cls=cls,
        hashf=hashf,
        logger=logger,
        chunk_by=chunk_by,
    )


def cachew_impl(*, func: Callable, db_path: PathProvider, cls: Type, hashf: HashF, logger: logging.Logger, chunk_by: int):
    def composite_hash(*args, **kwargs) -> SourceHash:
        return f'cachew: {CACHEW_FORMAT}, schema: {cls.__annotations__}, hash: {hashf(*args, **kwargs)}'

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # if db_path is None: raise AssertionError  # help mypy

        dbp: Path
        if callable(db_path): # TODO test this..
            dbp = Path(db_path(*args, **kwargs)) # type: ignore
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
            h = composite_hash(*args, **kwargs); kassert(h is not None) # just in case
            logger.debug('new hash: %s', h)

            with conn.begin():
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

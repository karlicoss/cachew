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
import inspect
from datetime import datetime, date
import stat
import tempfile
from pathlib import Path
import time
import sqlite3
import sys
import typing
from typing import (Any, Callable, Iterator, List, NamedTuple, Optional, Tuple,
                    Type, Union, TypeVar, Generic, Sequence, Iterable, Set, cast)
import dataclasses
import warnings


import appdirs  # type: ignore[import]

import sqlalchemy
from sqlalchemy import Column, Table, event, text


from .compat import fix_sqlalchemy_StatementError_str
try:
    fix_sqlalchemy_StatementError_str()
except Exception as e:
    logging.exception(e)


# in case of changes in the way cachew stores data, this should be changed to discard old caches
CACHEW_VERSION: str = __version__


PathIsh = Union[Path, str]

'''
Global settings, you can override them after importing cachew
'''
class settings:
    '''
    Toggle to disable caching
    '''
    ENABLE: bool = True

    DEFAULT_CACHEW_DIR: PathIsh = Path(appdirs.user_cache_dir('cachew'))

    '''
    Set to true if you want to fail early. Otherwise falls back to non-cached version
    '''
    THROW_ON_ERROR: bool = False



def get_logger() -> logging.Logger:
    return logging.getLogger('cachew')



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
        # ok, it's a bit hacky... attempt to preserve pytz infromation
        iso = value.isoformat()
        tz = getattr(value, 'tzinfo', None)
        if tz is None:
            return iso
        try:
            import pytz # type: ignore
        except ImportError:
            self.warn_pytz()
            return iso
        else:
            if isinstance(tz, pytz.BaseTzInfo):
                zone = tz.zone
                # should be present: https://github.com/python/typeshed/blame/968fd6d01d23470e0c8368e7ee7c43f54aaedc0e/stubs/pytz/pytz/tzinfo.pyi#L6
                assert zone is not None, tz
                return iso + ' ' + zone
            else:
                return iso

    def process_result_value(self, value: Optional[str], dialect) -> Optional[datetime]:
        if value is None:
            return None
        spl = value.split(' ')
        dt = datetime.fromisoformat(spl[0])
        if len(spl) <= 1:
            return dt
        zone = spl[1]
        # else attempt to decypher pytz tzinfo
        try:
            import pytz # type: ignore
        except ImportError:
            self.warn_pytz()
            return dt
        else:
            tz = pytz.timezone(zone)
            return dt.astimezone(tz)

    def warn_pytz(self) -> None:
        warnings.warn('install pytz for better timezone support while serializing with cachew')


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

import json
from typing import Dict
class Json(sqlalchemy.TypeDecorator):
    impl = sqlalchemy.String

    @property
    def python_type(self): return Dict

    def process_literal_param(self, value, dialect): raise NotImplementedError() # make pylint happy

    def process_bind_param(self, value: Optional[Dict], dialect) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[datetime]:
        if value is None:
            return None
        return json.loads(value)


jtypes = (int, float, bool, type(None))
class ExceptionAdapter(sqlalchemy.TypeDecorator):
    '''
    Enables support for caching Exceptions. Exception is treated as JSON and serialized.

    It's useful for defensive error handling, in case of cachew in particular for preserving error state.

    I elaborate on it here: [mypy-driven error handling](https://beepb00p.xyz/mypy-error-handling.html#kiss).
    '''
    impl = Json

    @property
    def python_type(self): return Exception

    def process_literal_param(self, value, dialect): raise NotImplementedError()  # make pylint happy

    def process_bind_param(self, value: Optional[Exception], dialect) -> Optional[List[Any]]:  # type: ignore[override]
        if value is None:
            return None
        sargs: List[Any] = []
        for a in value.args:
            if any(isinstance(a, t) for t in jtypes):
                sargs.append(a)
            elif isinstance(a, date):
                sargs.append(a.isoformat())
            else:
                sargs.append(str(a))
        return sargs

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Exception]:
        if value is None:
            return None
        # sadly, can't do much to convert back from the strings? Unless I serialize the type info as well?
        return Exception(*value)


PRIMITIVES = {
    str      : sqlalchemy.String,
    int      : sqlalchemy.Integer,
    float    : sqlalchemy.Float,
    bool     : sqlalchemy.Boolean,
    datetime : IsoDateTime,
    date     : IsoDate,
    dict     : Json,
    list     : Json,
    Exception: ExceptionAdapter,
}


Types = Union[
    Type[str],
    Type[int],
    Type[float],
    Type[bool],
    Type[datetime],
    Type[date],
    Type[dict],
    Type[list],
    Type[Exception],
    Type[NamedTuple],
]

Values = Union[
    str,
    int,
    float,
    bool,
    datetime,
    date,
    dict,
    list,
    Exception,
    NamedTuple,
]
# TODO assert all PRIMITIVES are also in Types/Values?


def is_primitive(cls: Type) -> bool:
    """
    >>> from typing import Dict, Any
    >>> is_primitive(int)
    True
    >>> is_primitive(set)
    False
    >>> is_primitive(dict)
    True
    """
    return cls in PRIMITIVES


# https://stackoverflow.com/a/2166841/706389
def is_dataclassish(t: Type) -> bool:
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


def get_union_args(cls) -> Optional[Tuple[Type]]:
    if getattr(cls, '__origin__', None) != Union:
        return None

    args = cls.__args__
    args = [e for e in args if e != type(None)]
    assert len(args) > 0
    return args


def is_union(cls):
    return get_union_args(cls) is not None


def strip_optional(cls) -> Tuple[Type, bool]:
    """
    >>> from typing import Optional, NamedTuple
    >>> strip_optional(Optional[int])
    (<class 'int'>, True)
    >>> class X(NamedTuple):
    ...     x: int
    >>> strip_optional(X)
    (<class 'cachew.X'>, False)
    """
    is_opt: bool = False

    args = get_union_args(cls)
    if args is not None and len(args) == 1:
        cls = args[0] # meh
        is_opt = True

    return (cls, is_opt)


def strip_generic(tp):
    """
    >>> strip_generic(List[int])
    <class 'list'>
    >>> strip_generic(str)
    <class 'str'>
    """
    GA = getattr(typing, '_GenericAlias') # ugh, can't make both mypy and pylint happy here?
    if isinstance(tp, GA):
        return tp.__origin__
    return tp


NT = TypeVar('NT')
# sadly, bound=NamedTuple is not working yet in mypy
# https://github.com/python/mypy/issues/685
# also needs to support dataclasses?


@dataclasses.dataclass
class NTBinder(Generic[NT]):
    """
    >>> class Job(NamedTuple):
    ...    company: str
    ...    title: Optional[str]
    >>> class Person(NamedTuple):
    ...     name: str
    ...     age: int
    ...     job: Optional[Job]

    NTBinder is a helper class for inteacting with sqlite database.
    Hierarchy is flattened:
    >>> binder = NTBinder.make(Person)
    >>> [(c.name, type(c.type)) for c in binder.columns]
    ... # doctest: +NORMALIZE_WHITESPACE
    [('name',         <class 'sqlalchemy.sql.sqltypes.String'>),
     ('age',          <class 'sqlalchemy.sql.sqltypes.Integer'>),
     ('_job_is_null', <class 'sqlalchemy.sql.sqltypes.Boolean'>),
     ('job_company',  <class 'sqlalchemy.sql.sqltypes.String'>),
     ('job_title',    <class 'sqlalchemy.sql.sqltypes.String'>)]


    >>> person = Person(name='alan', age=40, job=None)

    to_row converts object to a sql-friendly tuple. job=None, so we end up with True in _job_is_null field
    >>> tuple(binder.to_row(person))
    ('alan', 40, True, None, None)

    from_row does reverse conversion
    >>> binder.from_row(('alan', 40, True, None, None))
    Person(name='alan', age=40, job=None)

    >>> binder.from_row(('ann', 25, True, None, None, 'extra'))
    Traceback (most recent call last):
    ...
    cachew.CachewException: unconsumed items in iterator ['extra']
    """
    name     : Optional[str] # None means toplevel
    type_    : Types
    span     : int  # not sure if span should include optional col?
    primitive: bool
    optional : bool
    union    : Optional[Type] # helper, which isn't None if type is Union
    fields   : Sequence[Any] # mypy can't handle cyclic definition at this point :(

    @staticmethod
    def make(tp: Type[NT], name: Optional[str]=None) -> 'NTBinder[NT]':
        tp, optional = strip_optional(tp)
        union: Optional[Type]
        fields: Tuple[Any, ...]
        primitive: bool

        union_args = get_union_args(tp)
        if union_args is not None:
            CachewUnion = NamedTuple('_CachewUnionRepr', [ # type: ignore[misc]
                (x.__name__, Optional[x]) for x in union_args
            ])
            union = CachewUnion
            primitive = False
            fields = (NTBinder.make(tp=CachewUnion, name='_cachew_union_repr'),)
            span = 1
        else:
            union = None
            tp = strip_generic(tp)
            primitive = is_primitive(tp)

            if primitive:
                if name is None:
                    name = '_cachew_primitive' # meh. presumably, top level
            if primitive:
                fields = ()
                span = 1
            else:
                annotations = typing.get_type_hints(tp)
                if annotations == {}:
                    raise CachewException(f"{tp} (field '{name}'): doesn't look like a supported type to cache. See https://github.com/karlicoss/cachew#features for the list of supported types.")
                fields = tuple(NTBinder.make(tp=ann, name=fname) for fname, ann in annotations.items())
                span = sum(f.span for f in fields) + (1 if optional else 0)
        return NTBinder(
            name=name,
            type_=tp,
            span=span,
            primitive=primitive,
            optional=optional,
            union=union,
            fields=fields,
        )

    @property
    def columns(self) -> List[Column]:
        return list(self.iter_columns())

    # TODO not necessarily namedtuple? could be primitive type
    def to_row(self, obj: NT) -> Tuple[Optional[Values], ...]:
        return tuple(self._to_row(obj))

    def from_row(self, row: Iterable[Any]) -> NT:
        riter = iter(row)
        res = self._from_row(riter)
        remaining = list(islice(riter, 0, 1))
        if len(remaining) != 0:
            raise CachewException(f'unconsumed items in iterator {remaining}')
        assert res is not None  # nosec # help mypy; top level will not be None
        return res


    def _to_row(self, obj) -> Iterator[Optional[Values]]:
        if self.primitive:
            yield obj
        elif self.union is not None:
            CachewUnion = self.union
            (uf,) = self.fields
            # TODO assert only one of them matches??
            union = CachewUnion(**{
                f.name: obj if isinstance(obj, f.type_) else None
                for f in uf.fields
            })
            yield from uf._to_row(union)
        else:
            if self.optional:
                is_none = obj is None
                yield is_none
            else:
                is_none = False; assert obj is not None  # TODO hmm, that last assert is not very symmetric...

            if is_none:
                for _ in range(self.span - 1):
                    yield None
            else:
                yield from chain.from_iterable(
                    f._to_row(getattr(obj, f.name))
                    for f in self.fields
                )

    def _from_row(self, row_iter):
        if self.primitive:
            return next(row_iter)
        elif self.union is not None:
            CachewUnion = self.union
            (uf,) = self.fields
            # TODO assert only one of them is not None?
            union_params = [
                r
                for r in uf._from_row(row_iter) if r is not None
            ]
            assert len(union_params) == 1, union_params
            return union_params[0]
        else:
            if self.optional:
                is_none = next(row_iter)
            else:
                is_none = False

            if is_none:
                for _ in range(self.span - 1):
                    x = next(row_iter)
                    assert x is None, x  # huh. assert is kinda opposite of producing value
                return None
            else:
                return self.type_(*(
                    f._from_row(row_iter)
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
        lines = ['  ' * level + str(x.name) + ('?' if x.optional else '') + f' <span {x.span}>' for level, x in self.flatten()]
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)

    def flatten(self, level=0):
        yield (level, self)
        for f in self.fields:
            yield from f.flatten(level=level + 1)


# TODO better name to represent what it means?
SourceHash = str


class DbHelper:
    def __init__(self, db_path: Path, cls: Type) -> None:
        self.engine = sqlalchemy.create_engine(f'sqlite:///{db_path}', connect_args={'timeout': 0})
        # NOTE: timeout is necessary so we don't lose time waiting during recursive calls
        # by default, it's several seconds? you'd see 'test_recursive' test performance degrade

        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            # without wal, concurrent reading/writing is not gonna work

            # ugh. that's odd, how are we supposed to set WAL if the very fact of setting wal might lock the db?
            while True:
                try:
                    dbapi_connection.execute('PRAGMA journal_mode=WAL')
                    break
                except sqlite3.OperationalError as oe:
                    if 'database is locked' not in str(oe):
                        # ugh, pretty annoying that exception doesn't include database path for some reason
                        raise CachewException(f'Error while setting WAL on {db_path}') from oe
                time.sleep(0.1)


        self.connection = self.engine.connect()

        """
        Erm... this is pretty confusing.
        https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#transaction-isolation-level

        Somehow without this thing sqlalchemy logs BEGIN (implicit) instead of BEGIN TRANSACTION which actually works in sqlite...

        Judging by sqlalchemy/dialects/sqlite/base.py, looks like some sort of python sqlite driver problem??

        test_transaction should check this behaviour
        """
        @event.listens_for(self.connection, 'begin')
        # pylint: disable=unused-variable
        def do_begin(conn):
            # NOTE there is also BEGIN CONCURRENT in newer versions of sqlite. could use it later?
            conn.execute(text('BEGIN DEFERRED'))

        self.meta = sqlalchemy.MetaData()
        self.table_hash = Table('hash', self.meta, Column('value', sqlalchemy.String))

        self.binder = NTBinder.make(tp=cls)
        # actual cache
        self.table_cache     = Table('cache'    , self.meta, *self.binder.columns)
        # temporary table, we use it to insert and then (atomically?) rename to the above table at the very end
        self.table_cache_tmp = Table('cache_tmp', self.meta, *self.binder.columns)

    def __enter__(self) -> 'DbHelper':
        return self

    def __exit__(self, *args) -> None:
        self.connection.close()
        self.engine.dispose()


HashFunction = Callable[..., SourceHash]


def default_hash(*args, **kwargs) -> SourceHash:
    # TODO eh, demand hash? it's not safe either... ugh
    # can lead to werid consequences otherwise..
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

    >>> from typing import Sequence
    >>> def int_provider() -> Sequence[int]:
    ...     return (1, 2, 3)
    >>> infer_type(int_provider)
    <class 'int'>

    >> from typing import Iterator, Union
    >>> def union_provider() -> Iterator[Union[str, int]]:
    ...     yield 1
    ...     yield 'aaa'
    >>> infer_type(union_provider)
    typing.Union[str, int]
    """
    rtype = getattr(func, '__annotations__', {}).get('return', None)
    if rtype is None:
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
    if is_primitive(arg):
        return arg
    if is_union(arg):
        return arg # meh?
    if not is_dataclassish(arg):
        return bail(f"{arg} is not NamedTuple/dataclass")
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


PathProvider = Union[PathIsh, Callable[..., PathIsh]]


def cachew_error(e: Exception) -> None:
    if settings.THROW_ON_ERROR:
        raise e
    else:
        logger = get_logger()
        # todo add func name?
        logger.error("cachew: error while setting up cache, falling back to non-cached version")
        logger.exception(e)


use_default_path = cast(Path, object())

@doublewrap
def cachew(
        func=None,
        cache_path: Optional[PathProvider]=use_default_path,
        force_file: bool=False,
        cls=None,
        depends_on: HashFunction=default_hash,
        logger=None,
        chunk_by=100,
        # NOTE: allowed values for chunk_by depend on the system.
        # some systems (to be more specific, sqlite builds), it might be too large and cause issues
        # ideally this would be more defensive/autodetected, maybe with a warning?
        # you can use 'test_many' to experiment
        # - too small values (e.g. 10)  are slower than 100 (presumably, too many sql statements)
        # - too large values (e.g. 10K) are slightly slower as well (not sure why?)
        synthetic_key: Optional[str]=None,
        **kwargs,
):
    r"""
    Database-backed cache decorator. TODO more description?
    # TODO use this doc in readme?

    :param cache_path: if not set, `cachew.settings.DEFAULT_CACHEW_DIR` will be used.
    :param force_file: if set to True, assume `cache_path` is a regular file (instead of a directory)
    :param cls: if not set, cachew will attempt to infer it from return type annotation. See :func:`infer_type` and :func:`cachew.tests.test_cachew.test_return_type_inference`.
    :param depends_on: hash function to determine whether the underlying . Can potentially benefit from the use of side effects (e.g. file modification time). TODO link to test?
    :param logger: custom logger, if not specified will use logger named `cachew`. See :func:`get_logger`.
    :return: iterator over original or cached items

    Usage example:
    >>> from typing import NamedTuple, Iterator
    >>> class Link(NamedTuple):
    ...     url : str
    ...     text: str
    ...
    >>> @cachew
    ... def extract_links(archive_path: str) -> Iterator[Link]:
    ...     for i in range(5):
    ...         # simulate slow IO
    ...         # this function runs for five seconds for the purpose of demonstration, but realistically it might take hours
    ...         import time; time.sleep(1)
    ...         yield Link(url=f'http://link{i}.org', text=f'text {i}')
    ...
    >>> list(extract_links(archive_path='wikipedia_20190830.zip')) # that would take about 5 seconds on first run
    [Link(url='http://link0.org', text='text 0'), Link(url='http://link1.org', text='text 1'), Link(url='http://link2.org', text='text 2'), Link(url='http://link3.org', text='text 3'), Link(url='http://link4.org', text='text 4')]

    >>> from timeit import Timer
    >>> res = Timer(lambda: list(extract_links(archive_path='wikipedia_20190830.zip'))).timeit(number=1)
    ... # second run is cached, so should take less time
    >>> print(f"call took {int(res)} seconds")
    call took 0 seconds

    >>> res = Timer(lambda: list(extract_links(archive_path='wikipedia_20200101.zip'))).timeit(number=1)
    ... # now file has changed, so the cache will be discarded
    >>> print(f"call took {int(res)} seconds")
    call took 5 seconds
    """
    if logger is None:
        logger = get_logger()

    hashf = kwargs.get('hashf', None)
    if hashf is not None:
        warnings.warn("'hashf' is deprecated. Please use 'depends_on' instead")
        depends_on = hashf

    cn = cname(func)
    # todo not very nice that ENABLE check is scattered across two places
    if not settings.ENABLE or cache_path is None:
        logger.info('[%s]: cache explicitly disabled (settings.ENABLE is False or cache_path is None)', cn)
        return func

    if cache_path is use_default_path:
        cache_path = settings.DEFAULT_CACHEW_DIR
        logger.info('[%s]: no cache_path specified, using the default %s', cn, cache_path)

    # TODO fuzz infer_type, should never crash?
    inferred = infer_type(func)
    if isinstance(inferred, Failure):
        msg = f"failed to infer cache type: {inferred}. See https://github.com/karlicoss/cachew#features for the list of supported types."
        if cls is None:
            ex = CachewException(msg)
            cachew_error(ex)
            return func
        else:
            # it's ok, assuming user knows better
            logger.debug(msg)
    else:
        if cls is None:
            logger.debug('[%s] using inferred type %s', cn, inferred)
            cls = inferred
        else:
            if cls != inferred:
                logger.warning("inferred type %s mismatches specified type %s", inferred, cls)
                # TODO not sure if should be more serious error...

    ctx = Context(
        func         =func,
        cache_path   =cache_path,
        force_file   =force_file,
        cls          =cls,
        depends_on   =depends_on,
        logger       =logger,
        chunk_by     =chunk_by,
        synthetic_key=synthetic_key,
    )

    # hack to avoid extra stack frame (see test_recursive, test_deep-recursive)
    @functools.wraps(func)
    def binder(*args, **kwargs):
        kwargs['_cachew_context'] = ctx
        return cachew_wrapper(*args, **kwargs)
    return binder


def cname(func: Callable) -> str:
    # some functions don't have __module__
    mod = getattr(func, '__module__', None) or ''
    return f'{mod}:{func.__qualname__}'


_CACHEW_CACHED       = 'cachew_cached'  # TODO add to docs
_SYNTHETIC_KEY       = 'synthetic_key'
_SYNTHETIC_KEY_VALUE = 'synthetic_key_value'
_DEPENDENCIES        = 'dependencies'


class Context(NamedTuple):
    func         : Callable
    cache_path   : PathProvider
    force_file   : bool
    cls          : Type
    depends_on   : HashFunction
    logger       : logging.Logger
    chunk_by     : int
    synthetic_key: Optional[str]

    def composite_hash(self, *args, **kwargs) -> Dict[str, Any]:
        fsig = inspect.signature(self.func)
        # defaults wouldn't be passed in kwargs, but they can be an implicit dependency (especially inbetween program runs)
        defaults = {
            k: v.default
            for k, v in fsig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        # but only pass default if the user wants it in the hash function?
        hsig = inspect.signature(self.depends_on)
        defaults = {
            k: v
            for k, v in defaults.items()
            if k in hsig.parameters or 'kwargs' in hsig.parameters
        }
        kwargs = {**defaults, **kwargs}
        binder = NTBinder.make(tp=self.cls)
        schema = str(binder.columns) # todo not super nice, but works fine for now
        hash_parts = {
            'cachew'      : CACHEW_VERSION,
            'schema'      : schema,
            _DEPENDENCIES : str(self.depends_on(*args, **kwargs)),
        }
        synthetic_key = self.synthetic_key
        if synthetic_key is not None:
            hash_parts[_SYNTHETIC_KEY      ] = synthetic_key
            hash_parts[_SYNTHETIC_KEY_VALUE] = kwargs[synthetic_key]
            # FIXME assert it's in kwargs in the first place?
            # FIXME support positional args too? maybe extract the name from signature somehow? dunno
            # need to test it
        return hash_parts


def cachew_wrapper(
        *args,
        _cachew_context: Context,
        **kwargs,
):
    C = _cachew_context
    func          = C.func
    cache_path    = C.cache_path
    force_file    = C.force_file
    cls           = C.cls
    depends_on    = C.depends_on
    logger        = C.logger
    chunk_by      = C.chunk_by
    synthetic_key = C.synthetic_key

    cn = cname(func)
    if not settings.ENABLE:
        logger.info('[%s]: cache explicitly disabled (settings.ENABLE is False)', cn)
        yield from func(*args, **kwargs)
        return

    early_exit = False

    # WARNING: annoyingly huge try/catch ahead...
    # but it lets us save a function call, hence a stack frame
    # see test_recursive and test_deep_recursive
    try:
        dbp: Path
        if callable(cache_path):
            pp = cache_path(*args, **kwargs) # type: ignore
            if pp is None:
                logger.info('[%s]: cache explicitly disabled (cache_path is None)', cn)
                yield from func(*args, **kwargs)
                return
            else:
                dbp = Path(pp)
        else:
            dbp = Path(cache_path)

        dbp.parent.mkdir(parents=True, exist_ok=True)

        # need to be atomic here
        try:
            # note: stat follows symlinks (which is what we want)
            st = dbp.stat()
        except FileNotFoundError:
            # doesn't exist. then it's controlled by force_file
            if force_file:
                dbp = dbp
            else:
                dbp.mkdir(parents=True, exist_ok=True)
                dbp = dbp / cn
        else:
            # already exists, so just use cname if it's a dir
            if stat.S_ISDIR(st.st_mode):
                dbp = dbp / cn

        logger.debug('using %s for db cache', dbp)

        new_hash_d = C.composite_hash(*args, **kwargs)
        new_hash = json.dumps(new_hash_d)
        logger.debug('new hash: %s', new_hash)

        with DbHelper(dbp, cls) as db, \
             db.connection.begin():
            # NOTE: deferred transaction
            conn = db.connection
            binder = db.binder
            table_cache     = db.table_cache
            table_cache_tmp = db.table_cache_tmp

            # first, try to do as much as possible read-only, benefiting from deferred transaction
            old_hashes: Sequence
            try:
                # not sure if there is a better way...
                cursor = conn.execute(db.table_hash.select())
            except sqlalchemy.exc.OperationalError as e:
                # meh. not sure if this is a good way to handle this..
                if 'no such table: hash' in str(e):
                    old_hashes = []
                else:
                    raise e
            else:
                old_hashes = cursor.fetchall()


            assert len(old_hashes) <= 1, old_hashes  # shouldn't happen
            old_hash: Optional[SourceHash]
            if len(old_hashes) == 0:
                old_hash = None
            else:
                old_hash = old_hashes[0][0]  # returns a tuple...

            logger.debug('old hash: %s', old_hash)


            def cached_items():
                rows = conn.execute(table_cache.select())
                for row in rows:
                    yield binder.from_row(row)

            if new_hash == old_hash:
                logger.debug('hash matched: loading from cache')
                yield from cached_items()
                return

            logger.debug('hash mismatch: computing data and writing to db')

            if synthetic_key is not None:
                # attempt to use existing cache if possible, as a 'prefix'

                old_hash_d: Dict[str, Any] = {}
                if old_hash is not None:
                    try:
                        old_hash_d = json.loads(old_hash)
                    except json.JSONDecodeError:
                        # possible if we used old cachew version (<=0.8.1), hash wasn't json
                        pass

                hash_diffs = {
                    k: new_hash_d.get(k) == old_hash_d.get(k)
                    for k in (*new_hash_d.keys(), *old_hash_d.keys())
                    # the only 'allowed' differences for hash, otherwise need to recompute (e.g. if schema changed)
                    if k not in {_SYNTHETIC_KEY_VALUE, _DEPENDENCIES}
                }
                cache_compatible = all(hash_diffs.values())
                if cache_compatible:
                    def missing_keys(cached: List[str], wanted: List[str]) -> Optional[List[str]]:
                        # FIXME assert both cached and wanted are sorted? since we rely on it
                        # if not, then the user could use some custom key for caching (e.g. normalise filenames etc)
                        # although in this case passing it into the function wouldn't make sense?

                        if len(cached) == 0:
                            # no point trying to reuse anything, cache should be empty?
                            return None
                        if len(wanted) == 0:
                            # similar, no way to reuse cache
                            return None
                        if cached[0] != wanted[0]:
                            # there is no common prefix, so no way to reuse cache really
                            return None
                        last_cached = cached[-1]
                        # ok, now actually figure out which items are missing
                        for i, k in enumerate(wanted):
                            if k > last_cached:
                                # ok, rest of items are missing
                                return wanted[i:]
                        # otherwise too many things are cached, and we seem to wante less
                        return None

                    new_values: List[str] = new_hash_d[_SYNTHETIC_KEY_VALUE]
                    old_values: List[str] = old_hash_d[_SYNTHETIC_KEY_VALUE]
                    missing = missing_keys(cached=old_values, wanted=new_values)
                    if missing is not None:
                        # can reuse cache
                        kwargs[_CACHEW_CACHED] = cached_items()
                        kwargs[synthetic_key] = missing


            # NOTE on recursive calls
            # somewhat magically, they should work as expected with no extra database inserts?
            # the top level call 'wins' the write transaction and once it's gathered all data, will write it
            # the 'intermediate' level calls fail to get it and will pass data through
            # the cached 'bottom' level is read only and will be yielded without a write transaction
            try:
                # first 'write' statement will upgrade transaction to write transaction which might fail due to concurrency
                # see https://www.sqlite.org/lang_transaction.html
                # NOTE: because of 'checkfirst=True', only the last .create will guarantee the transaction upgrade to write transaction
                db.table_hash.create(conn, checkfirst=True)

                # 'table' used to be old 'cache' table name, so we just delete it regardless
                # otherwise it might overinfalte the cache db with stale values
                conn.execute(text(f'DROP TABLE IF EXISTS `table`'))

                # NOTE: we have to use .drop and then .create (e.g. instead of some sort of replace)
                # since it's possible to have schema changes inbetween calls
                # checkfirst=True because it might be the first time we're using cache
                table_cache_tmp.drop(conn, checkfirst=True)
                table_cache_tmp.create(conn)
            except sqlalchemy.exc.OperationalError as e:
                if e.code == 'e3q8' and 'database is locked' in str(e):
                    # someone else must be have won the write lock
                    # not much we can do here
                    # NOTE: important to close early, otherwise we might hold onto too many file descriptors during yielding
                    # see test_deep_recursive
                    # (normally connection is closed in DbHelper.__exit__)
                    conn.close()
                    yield from func(*args, **kwargs)
                    return
                else:
                    raise e
            # at this point we're guaranteed to have an exclusive write transaction

            datas = func(*args, **kwargs)
            column_names = [c.name for c in table_cache_tmp.columns]
            insert_into_table_cache_tmp = table_cache_tmp.insert()

            chunk: List[Any] = []
            def flush() -> None:
                nonlocal chunk
                if len(chunk) > 0:
                    # TODO hmm, it really doesn't work unless you zip into a dict first
                    # maybe should return dicts from binder instead then?
                    chunk_dict = [
                        dict(zip(column_names, row))
                        for row in chunk
                    ]
                    conn.execute(insert_into_table_cache_tmp, chunk_dict)
                    chunk = []

            for d in datas:
                try:
                    yield d
                except GeneratorExit:
                    early_exit = True
                    return
                  
                chunk.append(binder.to_row(d))
                if len(chunk) >= chunk_by:
                    flush()
            flush()

            # delete hash first, so if we are interrupted somewhere, it mismatches next time and everything is recomputed
            # pylint: disable=no-value-for-parameter
            conn.execute(db.table_hash.delete())

            # checkfirst is necessary since it might not have existed in the first place
            # e.g. first time we use cache
            table_cache.drop(conn, checkfirst=True)

            # meh https://docs.sqlalchemy.org/en/14/faq/metadata_schema.html#does-sqlalchemy-support-alter-table-create-view-create-trigger-schema-upgrade-functionality
            # also seems like sqlalchemy doesn't have any primitives to escape table names.. sigh
            conn.execute(text(f"ALTER TABLE `{table_cache_tmp.name}` RENAME TO `{table_cache.name}`"))

            # pylint: disable=no-value-for-parameter
            conn.execute(db.table_hash.insert().values([{'value': new_hash}]))
    except Exception as e:
        # sigh... see test_early_exit_shutdown...
        if early_exit and 'Cannot operate on a closed database' in str(e):
            return

        # todo hmm, kinda annoying that it tries calling the function twice?
        # but gonna require some sophisticated cooperation with the cached wrapper otherwise
        cachew_error(e)
        yield from func(*args, **kwargs)


__all__ = ['cachew', 'CachewException', 'SourceHash', 'HashFunction', 'get_logger', 'NTBinder']

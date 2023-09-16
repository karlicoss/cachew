from dataclasses import dataclass
import functools
import importlib.metadata
import inspect
import json
import logging
from pathlib import Path
import sqlite3
import stat
import sys
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    Sequence,
    cast,
    get_args,
    get_type_hints,
    get_origin,
    overload,
    TYPE_CHECKING,
)
import warnings

try:
    # orjson might not be available on some architectures, so let's make it defensive just in case
    from orjson import loads as orjson_loads, dumps as orjson_dumps  # pylint: disable=no-name-in-module
except:
    warnings.warn("orjson couldn't be imported. It's _highly_ recommended for better caching performance")
    def orjson_dumps(*args, **kwargs):  # type: ignore[misc]
        # sqlite needs a blob
        return json.dumps(*args, **kwargs).encode('utf8')

    orjson_loads = json.loads

import appdirs
import sqlalchemy
from sqlalchemy import Column, Table, event, text
from sqlalchemy.dialects import sqlite

from .logging_helper import makeLogger
from .marshall.cachew import CachewMarshall, build_schema
from .utils import (
    CachewException,
    TypeNotSupported,
)


# in case of changes in the way cachew stores data, this should be changed to discard old caches
CACHEW_VERSION: str = importlib.metadata.version(__name__)


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
    return makeLogger(__name__)


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

        # actual cache
        self.table_cache     = Table('cache'    , self.meta, Column('data', sqlalchemy.BLOB))
        # temporary table, we use it to insert and then (atomically?) rename to the above table at the very end
        self.table_cache_tmp = Table('cache_tmp', self.meta, Column('data', sqlalchemy.BLOB))

    def __enter__(self) -> 'DbHelper':
        return self

    def __exit__(self, *args) -> None:
        self.connection.close()
        self.engine.dispose()


R = TypeVar('R')
# ugh. python < 3.10 doesn't have ParamSpec and it seems tricky to backport it in compatible manner
if sys.version_info[:2] >= (3, 10) or TYPE_CHECKING:
    if sys.version_info[:2] >= (3, 10):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec
    P = ParamSpec('P')
    CC = Callable[P, R]  # need to give it a name, if inlined into bound=, mypy runs in a bug
    PathProvider = Union[PathIsh, Callable[P, PathIsh]]
    HashFunction = Callable[P, SourceHash]
else:
    # just use some dummy types so runtime is happy
    P = TypeVar('P')
    CC = Any
    PathProvider = Union[P, Any]
    HashFunction = Union[P, Any]

F = TypeVar('F', bound=CC)


def default_hash(*args, **kwargs) -> SourceHash:
    # TODO eh, demand hash? it's not safe either... ugh
    # can lead to werid consequences otherwise..
    return str(args + tuple(sorted(kwargs.items()))) # good enough??


# TODO give it as an example in docs
def mtime_hash(path: Path, *args, **kwargs) -> SourceHash:
    mt = path.stat().st_mtime
    return default_hash(f'{path}.{mt}', *args, **kwargs)


Failure = str
Kind = Literal['single', 'multiple']
Inferred = Tuple[Kind, Type[Any]]


def infer_return_type(func) -> Union[Failure, Inferred]:
    """
    >>> def const() -> int:
    ...     return 123
    >>> infer_return_type(const)
    ('single', <class 'int'>)

    >>> from typing import Optional
    >>> def first_character(s: str) -> Optional[str]:
    ...     return None if len(s) == 0 else s[0]
    >>> kind, opt = infer_return_type(first_character)
    >>> # in 3.8, Optional[str] is printed as Union[str, None], so need to hack around this
    >>> (kind, opt is Optional[str])
    ('single', True)

    # tuple is an iterable.. but presumably should be treated as a single value
    >>> from typing import Tuple
    >>> def a_tuple() -> Tuple[int, str]:
    ...     return (123, 'hi')
    >>> infer_return_type(a_tuple)
    ('single', typing.Tuple[int, str])

    >>> from typing import Collection, NamedTuple
    >>> class Person(NamedTuple):
    ...     name: str
    ...     age: int
    >>> def person_provider() -> Collection[Person]:
    ...     return []
    >>> infer_return_type(person_provider)
    ('multiple', <class 'cachew.Person'>)

    >>> def single_person() -> Person:
    ...     return Person(name="what", age=-1)
    >>> infer_return_type(single_person)
    ('single', <class 'cachew.Person'>)

    >>> from typing import Sequence
    >>> def int_provider() -> Sequence[int]:
    ...     return (1, 2, 3)
    >>> infer_return_type(int_provider)
    ('multiple', <class 'int'>)

    >>> from typing import Iterator, Union
    >>> def union_provider() -> Iterator[Union[str, int]]:
    ...     yield 1
    ...     yield 'aaa'
    >>> infer_return_type(union_provider)
    ('multiple', typing.Union[str, int])

    # a bit of an edge case
    >>> from typing import Tuple
    >>> def empty_tuple() -> Iterator[Tuple[()]]:
    ...     yield ()
    >>> infer_return_type(empty_tuple)
    ('multiple', typing.Tuple[()])

    ... # doctest: +ELLIPSIS

    >>> def untyped():
    ...     return 123
    >>> infer_return_type(untyped)
    'no return type annotation...'

    >>> from typing import List
    >>> class Custom:
    ...     pass
    >>> def unsupported() -> Custom:
    ...     return Custom()
    >>> infer_return_type(unsupported)
    "can't infer type from <class 'cachew.Custom'>: can't cache <class 'cachew.Custom'>"

    >>> def unsupported_list() -> List[Custom]:
    ...     return [Custom()]
    >>> infer_return_type(unsupported_list)
    "can't infer type from typing.List[cachew.Custom]: can't cache <class 'cachew.Custom'>"
    """
    hints = get_type_hints(func)
    rtype = hints.get('return', None)
    if rtype is None:
        return f"no return type annotation on {func}"

    def bail(reason: str) -> str:
        return f"can't infer type from {rtype}: " + reason

    # first we wanna check if the top level type is some sort of iterable that makes sense ot cache
    # e.g. List/Sequence/Iterator etc
    origin = get_origin(rtype)
    return_multiple = False
    if origin is not None and origin is not tuple:
        # TODO need to check it handles namedtuple correctly..
        try:
            return_multiple = issubclass(origin, Iterable)
        except TypeError:
            # that would happen if origin is not a 'proper' type, e.g. is a Union or something
            # seems like exception is the easiest way to check
            pass

    if return_multiple:
        # then the actual type to cache will be the argument of the top level one
        args = get_args(rtype)
        if args is None:
            return bail("has no __args__")

        if len(args) != 1:
            return bail(f"wrong number of __args__: {args}")

        (cached_type,) = args
    else:
        cached_type = rtype

    try:
        build_schema(Type=cached_type)
    except TypeNotSupported as ex:
        return bail(f"can't cache {ex.type_}")

    return ('multiple' if return_multiple else 'single', cached_type)


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


def cachew_error(e: Exception) -> None:
    if settings.THROW_ON_ERROR:
        # TODO would be nice to throw from the original code line -- maybe mess with the stack here?
        raise e
    else:
        logger = get_logger()
        # todo add func name?
        logger.error("cachew: error while setting up cache, falling back to non-cached version")
        logger.exception(e)


use_default_path = cast(Path, object())


# using cachew_impl here just to use different signatures during type checking (see below)
@doublewrap
def cachew_impl(
        func=None,
        cache_path: Optional[PathProvider[P]] = use_default_path,
        force_file: bool = False,
        cls: Optional[Type] = None,
        depends_on: HashFunction[P] = default_hash,
        logger: Optional[logging.Logger] = None,
        chunk_by: int = 100,
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
    :param cls: if not set, cachew will attempt to infer it from return type annotation. See :func:`infer_return_type` and :func:`cachew.tests.test_cachew.test_return_type_inference`.
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
        module_name = getattr(func, '__module__', None)
        if module_name is None:
            # rely on default cachew logger
            logger = get_logger()
        else:
            # if logger for the function's module already exists, reuse it
            if module_name in logging.Logger.manager.loggerDict:
                logger = logging.getLogger(module_name)
            else:
                logger = get_logger()


    hashf = kwargs.get('hashf', None)
    if hashf is not None:
        warnings.warn("'hashf' is deprecated. Please use 'depends_on' instead")
        depends_on = hashf

    cn = callable_name(func)
    # todo not very nice that ENABLE check is scattered across two places
    if not settings.ENABLE or cache_path is None:
        logger.debug('[%s]: cache explicitly disabled (settings.ENABLE is False or cache_path is None)', cn)
        return func

    if cache_path is use_default_path:
        cache_path = settings.DEFAULT_CACHEW_DIR
        logger.debug('[%s]: no cache_path specified, using the default %s', cn, cache_path)

    # TODO fuzz infer_return_type, should never crash?
    inference_res = infer_return_type(func)
    if isinstance(inference_res, Failure):
        msg = f"failed to infer cache type: {inference_res}. See https://github.com/karlicoss/cachew#features for the list of supported types."
        if cls is None:
            ex = CachewException(msg)
            cachew_error(ex)
            return func
        else:
            # it's ok, assuming user knows better
            logger.debug(msg)
    else:
        (kind, inferred) = inference_res
        assert kind == 'multiple'  # TODO implement later
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
        cls_         =cls,
        depends_on   =depends_on,
        logger       =logger,
        chunk_by     =chunk_by,
        synthetic_key=synthetic_key,
    )

    # hack to avoid extra stack frame (see test_recursive*)
    @functools.wraps(func)
    def binder(*args, **kwargs):
        kwargs['_cachew_context'] = ctx
        return cachew_wrapper(*args, **kwargs)
    return binder


if TYPE_CHECKING:
    # we need two versions due to @doublewrap
    # this is when we just annotate as @cachew without any args
    @overload  # type: ignore[no-overload-impl]
    def cachew(fun: F) -> F:
        ...

    # NOTE: we won't really be able to make sure the args of cache_path are the same as args of the wrapped function
    # because when cachew() is called, we don't know anything about the wrapped function yet
    # but at least it works for checking that cachew_path and depdns_on have the same args :shrug:
    @overload
    def cachew(
            cache_path: Optional[PathProvider[P]] = ...,
            *,
            force_file: bool = ...,
            cls: Optional[Type] = ...,
            depends_on: HashFunction[P] = ...,
            logger: Optional[logging.Logger] = ...,
            chunk_by: int = ...,
            synthetic_key: Optional[str] = ...,
    ) -> Callable[[F], F]:
        ...
else:
    cachew = cachew_impl


def callable_name(func: Callable) -> str:
    # some functions don't have __module__
    mod = getattr(func, '__module__', None) or ''
    return f'{mod}:{func.__qualname__}'


_CACHEW_CACHED       = 'cachew_cached'  # TODO add to docs
_SYNTHETIC_KEY       = 'synthetic_key'
_SYNTHETIC_KEY_VALUE = 'synthetic_key_value'
_DEPENDENCIES        = 'dependencies'


@dataclass
class Context(Generic[P]):
    func         : Callable
    cache_path   : PathProvider[P]
    force_file   : bool
    cls_         : Type
    depends_on   : HashFunction[P]
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
        schema = str(self.cls_)
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
        _cachew_context: Context[P],
        **kwargs,
):
    C = _cachew_context
    func          = C.func
    cache_path    = C.cache_path
    force_file    = C.force_file
    cls           = C.cls_
    depends_on    = C.depends_on
    logger        = C.logger
    chunk_by      = C.chunk_by
    synthetic_key = C.synthetic_key

    cn = callable_name(func)
    if not settings.ENABLE:
        logger.debug('[%s]: cache explicitly disabled (settings.ENABLE is False)', cn)
        yield from func(*args, **kwargs)
        return

    early_exit = False

    # WARNING: annoyingly huge try/catch ahead...
    # but it lets us save a function call, hence a stack frame
    # see test_recursive*
    try:
        dbp: Path
        if callable(cache_path):
            pp = cache_path(*args, **kwargs)
            if pp is None:
                logger.debug('[%s]: cache explicitly disabled (cache_path is None)', cn)
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
            # already exists, so just use callable name if it's a dir
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
            marshall = CachewMarshall(Type_=cls)
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

                # by default, sqlalchemy wraps all results into Row object
                # this can cause quite a lot of overhead if you're reading many rows
                # it seems that in principle, sqlalchemy supports just returning bare underlying tuple from the dbapi
                # but from browsing the code it doesn't seem like this functionality exposed
                # if you're looking for cues, see
                # - ._source_supports_scalars
                # - ._generate_rows
                # - ._row_getter
                # by using this raw iterator we speed up reading the cache quite a bit
                # asked here https://github.com/sqlalchemy/sqlalchemy/discussions/10350
                raw_row_iterator = getattr(rows, '_raw_row_iterator', None)
                if raw_row_iterator is None:
                    warnings.warn(
                        "CursorResult._raw_row_iterator method isn't found. This could lead to degraded cache reading performance."
                    )
                    row_iterator = rows
                else:
                    row_iterator = raw_row_iterator()

                for (blob,) in row_iterator:
                    j = orjson_loads(blob)
                    obj = marshall.load(j)
                    yield obj

            if new_hash == old_hash:
                logger.debug('hash matched: loading from cache')
                total = list(conn.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(table_cache)))[0][0]
                logger.info(f'{cn}: loading {total} objects from cachew (sqlite {dbp})')
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
                conn.execute(text('DROP TABLE IF EXISTS `table`'))

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
                    # see test_recursive_deep
                    # (normally connection is closed in DbHelper.__exit__)
                    conn.close()
                    yield from func(*args, **kwargs)
                    return
                else:
                    raise e
            # at this point we're guaranteed to have an exclusive write transaction

            datas = func(*args, **kwargs)
            # uhh. this gives a huge speedup for inserting
            # since we don't have to create intermediate dictionaries
            insert_into_table_cache_tmp_raw = str(table_cache_tmp.insert().compile(dialect=sqlite.dialect(paramstyle='qmark')))
            # I also tried setting paramstyle='qmark' in create_engine, but it seems to be ignored :(
            # idk what benefit sqlalchemy gives at this point, seems to just complicate things

            chunk: List[Any] = []

            def flush() -> None:
                nonlocal chunk
                if len(chunk) > 0:
                    conn.exec_driver_sql(insert_into_table_cache_tmp_raw, [(c,) for c in chunk])
                    chunk = []

            total_objects = 0
            for obj in datas:
                try:
                    total_objects += 1
                    yield obj
                except GeneratorExit:
                    early_exit = True
                    return

                dct = marshall.dump(obj)
                blob = orjson_dumps(dct)
                chunk.append(blob)
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
            logger.info(f'{cn}: wrote   {total_objects} objects to   cachew (sqlite {dbp})')
    except Exception as e:
        # sigh... see test_early_exit_shutdown...
        if early_exit and 'Cannot operate on a closed database' in str(e):
            return

        # todo hmm, kinda annoying that it tries calling the function twice?
        # but gonna require some sophisticated cooperation with the cached wrapper otherwise
        cachew_error(e)
        yield from func(*args, **kwargs)


from .legacy import NTBinder

__all__ = [
    'cachew',
    'CachewException',
    'SourceHash',
    'HashFunction',
    'get_logger',
    'NTBinder',  # NOTE: we need to keep this here for now because it's used in promnesia
]

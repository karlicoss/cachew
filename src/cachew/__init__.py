import fnmatch
import functools
import importlib.metadata
import inspect
import json
import logging
import os
import stat
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

try:
    # orjson might not be available on some architectures, so let's make it defensive just in case
    from orjson import dumps as orjson_dumps
    from orjson import loads as orjson_loads
except:
    warnings.warn("orjson couldn't be imported. It's _highly_ recommended for better caching performance", stacklevel=2)

    def orjson_dumps(*args, **kwargs):  # type: ignore[misc]
        # sqlite needs a blob
        return json.dumps(*args, **kwargs).encode('utf8')

    orjson_loads = json.loads  # ty: ignore[invalid-assignment]

import platformdirs

from .backend.common import AbstractBackend
from .backend.file import FileBackend
from .backend.sqlite import SqliteBackend
from .common import SourceHash
from .logging_helper import make_logger
from .marshall.cachew import CachewMarshall, build_schema
from .utils import (
    CachewException,
    TypeNotSupported,
)

# in case of changes in the way cachew stores data, this should be changed to discard old caches
CACHEW_VERSION: str = importlib.metadata.version(__name__)

PathIsh = Path | str

Backend = Literal['sqlite', 'file']


class settings:
    '''
    Global settings, you can override them after importing cachew
    '''

    '''
    Toggle to disable caching
    '''
    ENABLE: bool = True

    DEFAULT_CACHEW_DIR: PathIsh = Path(platformdirs.user_cache_dir('cachew'))

    '''
    Set to true if you want to fail early. Otherwise falls back to non-cached version
    '''
    THROW_ON_ERROR: bool = False

    DEFAULT_BACKEND: Backend = 'sqlite'


def get_logger() -> logging.Logger:
    return make_logger(__name__)


BACKENDS: dict[Backend, type[AbstractBackend]] = {
    'file': FileBackend,
    'sqlite': SqliteBackend,
}


R = TypeVar('R')
P = ParamSpec('P')
CC = Callable[P, R]  # need to give it a name, if inlined into bound=, mypy runs in a bug
PathProvider = PathIsh | Callable[P, PathIsh]
HashFunction = Callable[P, SourceHash]

F = TypeVar('F', bound=CC)


def default_hash(*args, **kwargs) -> SourceHash:
    # TODO eh, demand hash? it's not safe either... ugh
    # can lead to werid consequences otherwise..
    return str(args + tuple(sorted(kwargs.items())))  # good enough??


# TODO give it as an example in docs
def mtime_hash(path: Path, *args, **kwargs) -> SourceHash:
    mt = path.stat().st_mtime
    return default_hash(f'{path}.{mt}', *args, **kwargs)


Failure = str
Kind = Literal['single', 'multiple']
Inferred = tuple[Kind, type[Any]]


def infer_return_type(func) -> Failure | Inferred:
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

    >>> def single_str() -> str:
    ...     return 'hello'
    >>> infer_return_type(single_str)
    ('single', <class 'str'>)

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
    try:
        hints = get_type_hints(func)
    except Exception as ne:
        # get_type_hints might fail if types are forward defined or missing
        # see test_future_annotation for an example
        return str(ne)
    rtype = hints.get('return', None)
    if rtype is None:
        return f"no return type annotation on {func}"

    def bail(reason: str) -> str:
        return f"can't infer type from {rtype}: " + reason

    # first we wanna check if the top level type is some sort of iterable that makes sense ot cache
    # e.g. List/Sequence/Iterator etc
    return_multiple = _returns_multiple(rtype)

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


def _returns_multiple(rtype) -> bool:
    origin = get_origin(rtype)
    if origin is None:
        return False
    if origin is tuple:
        # usually tuples are more like single values rather than a sequence? (+ this works for namedtuple)
        return False
    try:
        return issubclass(origin, Iterable)
    except TypeError:
        # that would happen if origin is not a 'proper' type, e.g. is a Union or something
        # seems like exception is the easiest way to check
        return False


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


def cachew_error(e: Exception, *, logger: logging.Logger) -> None:
    if settings.THROW_ON_ERROR:
        # TODO would be nice to throw from the original code line -- maybe mess with the stack here?
        raise e
    logger.error("error while setting up cache, falling back to non-cached version")
    logger.exception(e)


use_default_path = cast(Path, object())


# using cachew_impl here just to use different signatures during type checking (see below)
@doublewrap
def cachew_impl(
    func=None,  # TODO should probably type it after switch to python 3.10/proper paramspec
    cache_path: PathProvider[P] | None = use_default_path,
    *,
    force_file: bool = False,
    cls: type | tuple[Kind, type] | None = None,
    depends_on: HashFunction[P] = default_hash,
    logger: logging.Logger | None = None,
    chunk_by: int = 100,
    # NOTE: allowed values for chunk_by depend on the system.
    # some systems (to be more specific, sqlite builds), it might be too large and cause issues
    # ideally this would be more defensive/autodetected, maybe with a warning?
    # you can use 'test_many' to experiment
    # - too small values (e.g. 10)  are slower than 100 (presumably, too many sql statements)
    # - too large values (e.g. 10K) are slightly slower as well (not sure why?)
    synthetic_key: str | None = None,
    backend: Backend | None = None,
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
        if module_name is not None and module_name in logging.Logger.manager.loggerDict:
            # if logger for the function's module already exists, reuse it
            logger = logging.getLogger(module_name)
        else:
            # rely on default cachew logger
            logger = get_logger()

    class AddFuncName(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = self.extra
            assert extra is not None
            func_name = extra['func_name']
            return f'[{func_name}] {msg}', kwargs

    assert func is not None
    func_name = callable_name(func)
    adapter = AddFuncName(logger, {'func_name': func_name})
    logger = cast(logging.Logger, adapter)

    hashf = kwargs.get('hashf')
    if hashf is not None:
        warnings.warn("'hashf' is deprecated. Please use 'depends_on' instead", stacklevel=2)
        depends_on = hashf

    # todo not very nice that ENABLE check is scattered across two places
    if not settings.ENABLE or cache_path is None:
        logger.debug('cache explicitly disabled (settings.ENABLE is False or cache_path is None)')
        return func

    if cache_path is use_default_path:
        cache_path = settings.DEFAULT_CACHEW_DIR
        logger.debug(f'no cache_path specified, using the default {cache_path}')

    use_kind: Kind | None = None
    use_cls: type | None = None
    if cls is not None:
        # defensive here since typing. objects passed as cls might fail on isinstance
        try:
            is_tuple = isinstance(cls, tuple)
        except:
            is_tuple = False
        if is_tuple:
            use_kind, use_cls = cls  # type: ignore[misc]
        else:
            use_kind = 'multiple'
            use_cls = cls  # type: ignore[assignment]

    # TODO fuzz infer_return_type, should never crash?
    inference_res = infer_return_type(func)
    if isinstance(inference_res, Failure):
        msg = f"failed to infer cache type: {inference_res}. See https://github.com/karlicoss/cachew#features for the list of supported types."
        if use_cls is None:
            ex = CachewException(msg)
            cachew_error(ex, logger=logger)
            return func
        else:
            # it's ok, assuming user knows better
            logger.debug(msg)
            assert use_kind is not None
    else:
        (inferred_kind, inferred_cls) = inference_res
        if use_cls is None:
            logger.debug(f'using inferred type {inferred_kind} {inferred_cls}')
            (use_kind, use_cls) = (inferred_kind, inferred_cls)
        else:
            assert use_kind is not None
            if (use_kind, use_cls) != inference_res:
                logger.warning(f"inferred type {inference_res} mismatches explicitly specified type {(use_kind, use_cls)}")
                # TODO not sure if should be more serious error...

    if use_kind == 'single':
        # pretend it's an iterable, this is just simpler for cachew_wrapper
        @functools.wraps(func)
        def _func(*args, **kwargs):
            return [func(*args, **kwargs)]

    else:
        _func = func

    assert use_cls is not None

    ctx = Context(
        func         =_func,
        cache_path   =cache_path,
        force_file   =force_file,
        cls_         =use_cls,
        depends_on   =depends_on,
        logger       =logger,
        chunk_by     =chunk_by,
        synthetic_key=synthetic_key,
        backend      =backend,
    )  # fmt: skip

    # hack to avoid extra stack frame (see test_recursive*)
    @functools.wraps(func)
    def binder(*args, **kwargs):
        kwargs['_cachew_context'] = ctx
        res = cachew_wrapper(*args, **kwargs)  # ty: ignore[missing-argument]

        if use_kind == 'single':
            lres = list(res)
            assert len(lres) == 1, lres  # shouldn't happen
            return lres[0]
        return res

    return binder


if TYPE_CHECKING:
    # we need two versions due to @doublewrap
    # this is when we just annotate as @cachew without any args
    @overload
    def cachew(fun: F) -> F: ...

    # NOTE: we won't really be able to make sure the args of cache_path are the same as args of the wrapped function
    # because when cachew() is called, we don't know anything about the wrapped function yet
    # but at least it works for checking that cachew_path and depdns_on have the same args :shrug:
    @overload
    def cachew(
        cache_path: PathProvider[P] | None = ...,
        *,
        force_file: bool = ...,
        cls: type | tuple[Kind, type] | None = ...,
        depends_on: HashFunction[P] = ...,
        logger: logging.Logger | None = ...,
        chunk_by: int = ...,
        synthetic_key: str | None = ...,
        backend: Backend | None = ...,
    ) -> Callable[[F], F]: ...

    def cachew(*args, **kwargs):  # make ty happy
        raise NotImplementedError
else:
    cachew = cachew_impl


def callable_name(func: Callable) -> str:
    # some functions don't have __module__
    mod = getattr(func, '__module__', None) or ''
    return f'{mod}:{getattr(func, "__qualname__")}'


def callable_module_name(func: Callable) -> str | None:
    return getattr(func, '__module__', None)


# could cache this, but might be worth not to, so the user can change it on the fly?
def _parse_disabled_modules(logger: logging.Logger | None = None) -> list[str]:
    # e.g. CACHEW_DISABLE=my.browser:my.reddit
    if 'CACHEW_DISABLE' not in os.environ:
        return []
    disabled = os.environ['CACHEW_DISABLE']
    if disabled.strip() == '':
        return []
    if ',' in disabled and logger:
        logger.warning(
            'CACHEW_DISABLE contains a comma, but this expects a $PATH-like, colon-separated list; '
            f'try something like CACHEW_DISABLE={disabled.replace(",", ":")}'
        )
    # remove any empty strings incase did something like CACHEW_DISABLE=my.module:$CACHEW_DISABLE
    return [p for p in disabled.split(':') if p.strip() != '']


def _matches_disabled_module(module_name: str, pattern: str) -> bool:
    '''
    >>> _matches_disabled_module('my.browser', 'my.browser')
    True
    >>> _matches_disabled_module('my.browser', 'my.*')
    True
    >>> _matches_disabled_module('my.browser', 'my')
    True
    >>> _matches_disabled_module('my.browser', 'my.browse*')
    True
    >>> _matches_disabled_module('my.browser.export', 'my.browser')
    True
    >>> _matches_disabled_module('mysomething.else', '*')  # CACHEW_DISABLE='*' disables everything
    True
    >>> _matches_disabled_module('my.browser', 'my.br?????')  # fnmatch supports unix-like patterns
    True
    >>> _matches_disabled_module('my.browser', 'my.browse')
    False
    >>> _matches_disabled_module('mysomething.else', 'my')  # since not at '.' boundary, doesn't match
    False
    >>> _matches_disabled_module('mysomething.else', '')
    False
    >>> _matches_disabled_module('my.browser', 'my.browser.export')
    False
    '''

    if module_name == pattern:
        return True

    module_parts = module_name.split('.')
    pattern_parts = pattern.split('.')

    # e.g. if pattern is 'module.submod.inner_module' and module is just 'module.submod'
    # theres no possible way for it to match
    if len(module_parts) < len(pattern_parts):
        return False

    for mp, pp in zip(module_parts, pattern_parts, strict=False):
        if fnmatch.fnmatch(mp, pp):
            continue
        return False
    return True


def _module_is_disabled(module_name: str, logger: logging.Logger) -> bool:
    disabled_modules = _parse_disabled_modules(logger)
    for pat in disabled_modules:
        if _matches_disabled_module(module_name, pat):
            logger.debug(f"caching disabled for {module_name} (matched '{pat}' from 'CACHEW_DISABLE={os.environ['CACHEW_DISABLE']})'")
            return True
    return False


# fmt: off
_CACHEW_CACHED       = 'cachew_cached'  # TODO add to docs
_SYNTHETIC_KEY       = 'synthetic_key'
_SYNTHETIC_KEY_VALUE = 'synthetic_key_value'
_DEPENDENCIES        = 'dependencies'
# fmt: on


@dataclass
class Context(Generic[P]):
    # fmt: off
    func         : Callable
    cache_path   : PathProvider[P]
    force_file   : bool
    cls_         : type
    depends_on   : HashFunction[P]
    logger       : logging.Logger
    chunk_by     : int
    synthetic_key: str | None
    backend      : Backend | None

    def composite_hash(self, *args, **kwargs) -> dict[str, Any]:
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
    # fmt: on


def cachew_wrapper(
    *args,
    _cachew_context: Context[P],
    **kwargs,
):
    C = _cachew_context
    # fmt: off
    func          = C.func
    cache_path    = C.cache_path
    force_file    = C.force_file
    cls           = C.cls_
    logger        = C.logger
    chunk_by      = C.chunk_by
    synthetic_key = C.synthetic_key
    backend_name  = C.backend
    # fmt: on

    used_backend = backend_name or settings.DEFAULT_BACKEND

    func_name = callable_name(func)
    if not settings.ENABLE:
        logger.debug('cache explicitly disabled (settings.ENABLE is False)')
        yield from func(*args, **kwargs)
        return

    mod_name = callable_module_name(func)
    if mod_name is not None and _module_is_disabled(mod_name, logger):
        yield from func(*args, **kwargs)
        return

    def get_db_path() -> Path | None:
        db_path: Path
        if callable(cache_path):
            pp = cache_path(*args, **kwargs)
            if pp is None:
                logger.debug('cache explicitly disabled (cache_path is None)')
                # early return, in this case we just yield the original items from the function
                return None
            else:
                db_path = Path(pp)
        else:
            db_path = Path(cache_path)

        db_path.parent.mkdir(parents=True, exist_ok=True)

        # need to be atomic here, hence calling stat() once and then just using the results
        try:
            # note: stat follows symlinks (which is what we want)
            st = db_path.stat()
        except FileNotFoundError:
            # doesn't exist. then it's controlled by force_file
            if force_file:
                # just use db_path as is
                pass
            else:
                db_path.mkdir(parents=True, exist_ok=True)
                db_path = db_path / func_name
        else:
            # already exists, so just use callable name if it's a dir
            if stat.S_ISDIR(st.st_mode):
                db_path = db_path / func_name

        logger.debug(f'using {used_backend}:{db_path} for cache')
        return db_path

    def try_use_synthetic_key() -> None:
        if synthetic_key is None:
            return
        # attempt to use existing cache if possible, as a 'prefix'

        old_hash_d: dict[str, Any] = {}
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
        if not cache_compatible:
            return

        def missing_keys(cached: list[str], wanted: list[str]) -> list[str] | None:
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

        new_values: list[str] = new_hash_d[_SYNTHETIC_KEY_VALUE]
        old_values: list[str] = old_hash_d[_SYNTHETIC_KEY_VALUE]
        missing = missing_keys(cached=old_values, wanted=new_values)
        if missing is not None:
            # can reuse cache
            kwargs[_CACHEW_CACHED] = cached_items()
            kwargs[synthetic_key] = missing

    early_exit = False

    def written_to_cache():
        nonlocal early_exit

        datas = func(*args, **kwargs)

        if isinstance(backend, FileBackend):
            # FIXME uhhh.. this is a bit crap
            # but in sqlite mode we don't want to publish new hash before we write new items
            # maybe should use tmp table for hashes as well?
            backend.write_new_hash(new_hash)
        else:
            # happens later for sqlite
            pass

        flush_blobs = backend.flush_blobs

        chunk: list[Any] = []

        def flush() -> None:
            nonlocal chunk
            if len(chunk) > 0:
                flush_blobs(chunk=chunk)
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

        backend.finalize(new_hash)
        logger.info(f'wrote   {total_objects} objects to   cachew ({used_backend}:{db_path})')

    def cached_items():
        total_cached = backend.cached_blobs_total()
        total_cached_s = '' if total_cached is None else f'{total_cached} '
        logger.info(f'loading {total_cached_s}objects from cachew ({used_backend}:{db_path})')

        for blob in backend.cached_blobs():
            j = orjson_loads(blob)
            obj = marshall.load(j)
            yield obj

    # NOTE: annoyingly huge try/catch ahead...
    # but it lets us save a function call, hence a stack frame
    # see test_recursive*
    try:
        db_path = get_db_path()
        if db_path is None:
            yield from func(*args, **kwargs)
            return

        BackendCls = BACKENDS[used_backend]

        new_hash_d = C.composite_hash(*args, **kwargs)
        new_hash: SourceHash = json.dumps(new_hash_d)
        logger.debug(f'new hash: {new_hash}')

        marshall: CachewMarshall[Any] = CachewMarshall(Type_=cls)

        with BackendCls(cache_path=db_path, logger=logger) as backend:
            old_hash = backend.get_old_hash()
            logger.debug(f'old hash: {old_hash}')

            if new_hash == old_hash:
                logger.debug('hash matched: loading from cache')
                yield from cached_items()
                return

            logger.debug('hash mismatch: computing data and writing to db')

            try_use_synthetic_key()

            got_write = backend.get_exclusive_write()
            if not got_write:
                # NOTE: this is the bit we really have to watch out for and not put in a helper function
                # otherwise it's causing an extra stack frame on every call
                # the rest (reading from cachew or writing to cachew) happens once per function call? so not a huge deal
                yield from func(*args, **kwargs)
                return

            # at this point we're guaranteed to have an exclusive write transaction
            yield from written_to_cache()
    except Exception as e:
        # sigh... see test_early_exit_shutdown...
        if early_exit and 'Cannot operate on a closed database' in str(e):
            return

        # todo hmm, kinda annoying that it tries calling the function twice?
        # but gonna require some sophisticated cooperation with the cached wrapper otherwise
        cachew_error(e, logger=logger)
        yield from func(*args, **kwargs)


__all__ = [
    'CachewException',
    'HashFunction',
    'SourceHash',
    'cachew',
    'get_logger',
]

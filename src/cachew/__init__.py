import functools
import importlib.metadata
import inspect
import json
import logging
import stat
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Literal,
    Protocol,
    cast,
    overload,
)

try:
    # orjson might not be available on some architectures, so let's make it defensive just in case
    from orjson import dumps as orjson_dumps
    from orjson import loads as orjson_loads
except:
    warnings.warn("orjson couldn't be imported. It's _highly_ recommended for better caching performance", stacklevel=2)

    def orjson_dumps(*args: Any, **kwargs: Any) -> bytes:  # type: ignore[misc]
        # sqlite needs a blob
        return json.dumps(*args, **kwargs).encode('utf8')

    orjson_loads = json.loads  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

import platformdirs

from ._disable import module_is_disabled
from ._infer import Failure, Kind, infer_return_type
from .backend.common import AbstractBackend
from .backend.file import FileBackend
from .backend.sqlite import SqliteBackend
from .common import CachewException, SourceHash
from .logging_helper import make_logger
from .marshall.cachew import CachewMarshall

# in case of changes in the way cachew stores data, this should be changed to discard old caches
CACHEW_VERSION: str = importlib.metadata.version(__name__)

Backend = Literal['sqlite', 'file']


class settings:
    '''
    Global settings, you can override them after importing cachew
    '''

    '''
    Toggle to disable caching
    '''
    ENABLE: bool = True

    DEFAULT_CACHEW_DIR: Path | str = Path(platformdirs.user_cache_dir('cachew'))

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

_DEFAULT_CHUNK_BY = 100


type PathProvider[**P] = Path | str | Callable[P, Path | str | None]
type HashFunction[**P] = Callable[P, SourceHash]


def default_hash(*args, **kwargs) -> SourceHash:
    # TODO eh, demand hash? it's not safe either... ugh
    # can lead to werid consequences otherwise..
    return str(args + tuple(sorted(kwargs.items())))  # good enough??


# TODO give it as an example in docs
def mtime_hash(path: Path, *args, **kwargs) -> SourceHash:
    mt = path.stat().st_mtime
    return default_hash(f'{path}.{mt}', *args, **kwargs)


class PreservingDecorator(Protocol):
    def __call__[F: Callable[..., Any]](self, func: F, /) -> F: ...


type ExplicitCacheType[ItemT] = type[ItemT] | tuple[Kind, type[ItemT]]
# NOTE: just ItemT basically means ('multiple', ItemT)


def cachew_error(e: Exception, *, logger: logging.Logger) -> None:
    if settings.THROW_ON_ERROR:
        # TODO would be nice to throw from the original code line -- maybe mess with the stack here?
        raise e
    logger.error("error while setting up cache, falling back to non-cached version")
    logger.exception(e)


use_default_path = cast(Path, object())


# ReturnT is the decorated function's public return type.
# ItemT is the type cachew serializes: the element type for iterable returns, or ReturnT for single-value returns.
def cachew_impl[**P, ReturnT, ItemT](
    func: Callable[P, ReturnT],
    cache_path: PathProvider[P] | None = use_default_path,
    *,
    force_file: bool = False,
    cls: ExplicitCacheType[ItemT] | None = None,
    depends_on: HashFunction[P] = default_hash,
    logger: logging.Logger | None = None,
    chunk_by: int = _DEFAULT_CHUNK_BY,
    # NOTE: allowed values for chunk_by depend on the system.
    # some systems (to be more specific, sqlite builds), it might be too large and cause issues
    # ideally this would be more defensive/autodetected, maybe with a warning?
    # you can use 'test_many' to experiment
    # - too small values (e.g. 10)  are slower than 100 (presumably, too many sql statements)
    # - too large values (e.g. 10K) are slightly slower as well (not sure why?)
    synthetic_key: str | None = None,
    backend: Backend | None = None,
    **kwargs: Any,
) -> Callable[P, ReturnT]:
    r"""
    Database-backed cache decorator. TODO more description?
    # TODO use this doc in readme?

    :param cache_path: if not set, `cachew.settings.DEFAULT_CACHEW_DIR` will be used. If set to `None`, or if a callable returns `None`, caching is disabled for that call.
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
        def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
            extra = self.extra
            assert extra is not None
            func_name = extra['func_name']
            return f'[{func_name}] {msg}', kwargs

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

    use_kind: Kind
    use_cls: type[ItemT] | None = None
    if cls is not None:
        # defensive here since typing. objects passed as cls might fail on isinstance
        try:
            is_tuple = isinstance(cls, tuple)
        except:
            is_tuple = False
        if is_tuple:
            use_kind, use_cls = cls  # type: ignore[misc]  # ty: ignore[not-iterable]
        else:
            use_kind = 'multiple'
            use_cls = cls  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

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
    else:
        (inferred_kind, inferred_cls) = inference_res
        if use_cls is None:
            logger.debug(f'using inferred type {inferred_kind} {inferred_cls}')
            (use_kind, use_cls) = (inferred_kind, inferred_cls)
        else:
            if (use_kind, use_cls) != inference_res:
                logger.warning(
                    f"inferred type {inference_res} mismatches explicitly specified type {(use_kind, use_cls)}"
                )
                # TODO not sure if should be more serious error...

    _func: Callable[P, Iterable[ItemT]]
    if use_kind == 'single':
        # pretend it's an iterable, this is just simpler for cachew_wrapper
        @functools.wraps(func)
        def _func_single(*args: P.args, **kwargs: P.kwargs) -> list[ItemT]:
            # Runtime invariant: in single mode, ReturnT is ItemT.
            single_func = cast(Callable[P, ItemT], func)
            return [single_func(*args, **kwargs)]

        _func = _func_single

    else:
        # Runtime invariant: in multiple mode, ReturnT is Iterable[ItemT].
        # This comes from cls/return-type inference, which type checkers can't connect back to the generic parameters.
        _func = cast(Callable[P, Iterable[ItemT]], func)

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
        backend      =backend or settings.DEFAULT_BACKEND,
    )  # fmt: skip

    # hack to avoid extra stack frame (see test_recursive*)
    @functools.wraps(func)
    def binder(*args: P.args, **kwargs: P.kwargs) -> ReturnT:
        res = cachew_wrapper(ctx, *args, **kwargs)

        if use_kind == 'single':
            lres = list(res)
            assert len(lres) == 1, lres  # shouldn't happen
            return cast(ReturnT, lres[0])
        else:
            return cast(ReturnT, res)

    return binder


@overload
def cachew[F: Callable[..., Any]](fun: F, /) -> F: ...


# NOTE: cache_path and depends_on are tied to each other, but not to the wrapped function.
# Runtime supports looser helpers, such as depends_on accepting only the arguments it cares about.
@overload
def cachew[**P, ItemT](
    cache_path: PathProvider[P] | None = ...,
    *,
    force_file: bool = ...,
    cls: ExplicitCacheType[ItemT] | None = ...,
    depends_on: HashFunction[P] = ...,
    logger: logging.Logger | None = ...,
    chunk_by: int = ...,
    synthetic_key: str | None = ...,
    backend: Backend | None = ...,
) -> PreservingDecorator: ...


def cachew(func_or_cache_path: Any = use_default_path, /, **kwargs: Any) -> Any:
    if callable(func_or_cache_path) and len(kwargs) == 0:
        return cachew_impl(func_or_cache_path)

    if 'cache_path' in kwargs:
        if func_or_cache_path is not use_default_path:
            raise TypeError("cachew() got multiple values for argument 'cache_path'")
        cache_path = kwargs.pop('cache_path')
    else:
        cache_path = func_or_cache_path

    def decorator[F: Callable[..., Any]](func: F, /) -> F:
        return cast(F, cachew_impl(func, cache_path=cache_path, **kwargs))

    return decorator


cachew.__doc__ = cachew_impl.__doc__


def callable_name(func: Callable[..., Any]) -> str:
    # some functions don't have __module__
    mod = getattr(func, '__module__', None) or ''
    return f'{mod}:{getattr(func, "__qualname__")}'


def callable_module_name(func: Callable[..., Any]) -> str | None:
    return getattr(func, '__module__', None)


# fmt: off
_CACHEW_CACHED       = 'cachew_cached'  # TODO add to docs
_SYNTHETIC_KEY       = 'synthetic_key'
_SYNTHETIC_KEY_VALUE = 'synthetic_key_value'
_DEPENDENCIES        = 'dependencies'
# fmt: on


@dataclass
class Context[**P, ItemT]:
    # fmt: off
    func         : Callable[P, Iterable[ItemT]]
    cache_path   : PathProvider[P]
    force_file   : bool
    cls_         : type[ItemT]
    depends_on   : HashFunction[P]
    logger       : logging.Logger
    chunk_by     : int
    synthetic_key: str | None
    backend      : Backend
    # fmt: on

    def resolve_cache_path(self, /, *args: P.args, **kwargs: P.kwargs) -> Path | None:
        resolved_path: Path
        if isinstance(self.cache_path, (Path, str)):
            resolved_path = Path(self.cache_path)
        else:
            pp = self.cache_path(*args, **kwargs)
            if pp is None:
                self.logger.debug('cache explicitly disabled (cache_path is None)')
                return None
            resolved_path = Path(pp)

        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # Need to be atomic here, hence calling stat() once and then just using the results.
        try:
            # stat follows symlinks, which is what we want.
            st = resolved_path.stat()
        except FileNotFoundError:
            if not self.force_file:
                resolved_path.mkdir(parents=True, exist_ok=True)
                resolved_path = resolved_path / callable_name(self.func)
        else:
            if stat.S_ISDIR(st.st_mode):
                resolved_path = resolved_path / callable_name(self.func)

        self.logger.debug(f'using {self.backend}:{resolved_path} for cache')
        return resolved_path

    def composite_hash(self, *args, **kwargs) -> dict[str, Any]:
        fsig = inspect.signature(self.func)
        # defaults wouldn't be passed in kwargs, but they can be an implicit dependency (especially inbetween program runs)
        defaults = {
            k: v.default
            for k, v in fsig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }  # fmt: skip
        # but only pass default if the user wants it in the hash function?
        hsig = inspect.signature(self.depends_on)
        defaults = {
            k: v
            for k, v in defaults.items()
            if k in hsig.parameters or 'kwargs' in hsig.parameters
        }  # fmt: skip
        kwargs = {**defaults, **kwargs}
        schema = str(self.cls_)
        hash_parts = {
            'cachew'      : CACHEW_VERSION,
            'schema'      : schema,
            _DEPENDENCIES : str(self.depends_on(*args, **kwargs)),
        }  # fmt: skip
        synthetic_key = self.synthetic_key
        if synthetic_key is not None:
            hash_parts[_SYNTHETIC_KEY      ] = synthetic_key  # fmt: skip
            hash_parts[_SYNTHETIC_KEY_VALUE] = kwargs[synthetic_key]
            # FIXME assert it's in kwargs in the first place?
            # FIXME support positional args too? maybe extract the name from signature somehow? dunno
            # need to test it
        return hash_parts


def cachew_wrapper[**P, ItemT](
    _cachew_context: Context[P, ItemT],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Iterator[ItemT]:
    C = _cachew_context
    # fmt: off
    func          = C.func
    cls           = C.cls_
    logger        = C.logger
    chunk_by      = C.chunk_by
    synthetic_key = C.synthetic_key
    # fmt: on

    used_backend = C.backend

    if not settings.ENABLE:
        logger.debug('cache explicitly disabled (settings.ENABLE is False)')
        yield from func(*args, **kwargs)
        return

    mod_name = callable_module_name(func)
    if mod_name is not None and module_is_disabled(mod_name, logger):
        yield from func(*args, **kwargs)
        return

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
            kwargs[_CACHEW_CACHED] = cached_items()  # ty: ignore[invalid-assignment]
            kwargs[synthetic_key] = missing  # ty: ignore[invalid-assignment]

    early_exit = False

    def written_to_cache() -> Iterator[ItemT]:
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

        chunk: list[bytes] = []

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
        logger.info(f'wrote   {total_objects} objects to   cachew ({used_backend}:{resolved_cache_path})')

    def cached_items() -> Iterator[ItemT]:
        total_cached = backend.cached_blobs_total()
        total_cached_s = '' if total_cached is None else f'{total_cached} '
        logger.info(f'loading {total_cached_s}objects from cachew ({used_backend}:{resolved_cache_path})')

        for blob in backend.cached_blobs():
            j = orjson_loads(blob)
            obj = marshall.load(j)
            yield obj

    # NOTE: annoyingly huge try/catch ahead...
    # but it lets us save a function call, hence a stack frame
    # see test_recursive*
    try:
        resolved_cache_path = C.resolve_cache_path(*args, **kwargs)
        if resolved_cache_path is None:
            yield from func(*args, **kwargs)
            return

        BackendCls = BACKENDS[used_backend]

        new_hash_d = C.composite_hash(*args, **kwargs)
        new_hash: SourceHash = json.dumps(new_hash_d)
        logger.debug(f'new hash: {new_hash}')

        # NOTE: marshall is captured by written_to_db
        marshall: CachewMarshall[ItemT] = CachewMarshall(Type_=cls)

        with BackendCls(cache_path=resolved_cache_path, logger=logger) as backend:
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

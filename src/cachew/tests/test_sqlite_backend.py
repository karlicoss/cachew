import logging
import sqlite3
from collections.abc import Generator, Iterator, Sequence
from contextlib import ExitStack, closing
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
from pathlib import Path
from subprocess import check_call
from typing import Protocol, cast

import pytest

from .. import BACKENDS, Backend, cachew, get_logger, settings
from ..backend import sqlite
from ..backend.common import AbstractBackend
from ..common import SourceHash

_PROCESS_TIMEOUT_SECONDS = 10.0
# The child timeout is longer so the parent remains responsible for diagnosing and cleaning up a stalled child.
_CHILD_RETRY_TIMEOUT_SECONDS = 30.0


class _ProcessEvent(Protocol):
    """The subset of a multiprocessing Event used by spawned workers."""

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class _RetryWaiter:
    """Turn a production retry sleep into observable, parent-controlled synchronization."""

    def __init__(
        self,
        *,
        retry_entered: _ProcessEvent,
        allow_retry: _ProcessEvent,
    ) -> None:
        self.retry_entered = retry_entered
        self.allow_retry = allow_retry

    def sleep(self, _seconds: float) -> None:
        # Reaching this method proves that the backend caught a lock error and entered its retry branch.
        self.retry_entered.set()
        # Keep the constructor inside that retry branch until the parent has observed both workers.
        assert self.allow_retry.wait(timeout=_CHILD_RETRY_TIMEOUT_SECONDS)


def _backend(*, cache_path: Path, backend: Backend) -> AbstractBackend:
    return BACKENDS[backend](cache_path=cache_path, logger=get_logger())


def _call_cachew_after_wal_retry(
    *,
    cache_path: Path,
    backend: Backend,
    retry_entered: _ProcessEvent,
    allow_retry: _ProcessEvent,
) -> None:
    """Run a strict Cachew call after exposing and pausing its first WAL retry."""

    # Replace the backend module's binding without changing time.sleep globally in the child.
    setattr(
        sqlite,
        'time',
        _RetryWaiter(retry_entered=retry_entered, allow_retry=allow_retry),
    )
    settings.THROW_ON_ERROR = True

    @cachew(cache_path=cache_path, force_file=True, backend=backend)
    def items(version: int) -> Iterator[int]:
        yield version * 10
        yield version * 10 + 1

    assert list(items(version=1)) == [10, 11]


def _publish(
    *,
    cache_path: Path,
    backend: Backend,
    source_hash: SourceHash,
    blobs: Sequence[bytes],
) -> None:
    with _backend(cache_path=cache_path, backend=backend) as writer:
        assert writer.get_exclusive_write()
        writer.flush_blobs(chunk=blobs)
        writer.finalize(source_hash)


def _assert_cache(
    *,
    cache_path: Path,
    backend: Backend,
    source_hash: SourceHash,
    blobs: Sequence[bytes],
) -> None:
    with _backend(cache_path=cache_path, backend=backend) as reader:
        assert reader.get_old_hash() == source_hash
        assert reader.cached_blobs_total() == len(blobs)
        assert list(reader.cached_blobs()) == list(blobs)


def test_old_cache_v0_6_3(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, 'THROW_ON_ERROR', True)

    sql = '''
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE hash (
	value VARCHAR
);
INSERT INTO hash VALUES('cachew: 1, schema: {''_'': <class ''int''>}, hash: ()');
CREATE TABLE IF NOT EXISTS "table" (
	_cachew_primitive INTEGER
);
INSERT INTO "table" VALUES(1);
INSERT INTO "table" VALUES(2);
INSERT INTO "table" VALUES(3);
COMMIT;
    '''
    db = tmp_path / 'cache.sqlite'
    check_call(['sqlite3', db, sql])

    @cachew(db, backend='sqlite')
    def fun() -> Iterator[int]:
        yield from [1, 2, 3]

    # This checks that Cachew can replace the legacy schema without crashing.
    # See test_version_change for the full version-invalidation behavior.
    assert list(fun()) == [1, 2, 3]


def test_concurrent_first_cachew_calls_retry_initial_wal_transition(
    *,
    tmp_path: Path,
) -> None:
    """
    Concurrent first calls may use the same new cache path, so a transient lock while enabling WAL must delay initialization rather than fail either decorated call.
    Hold an empty DELETE-mode database exclusively, prove that both calls enter their backend's real retry branch, then release them and verify that they return normally and publish a reusable cache.
    """
    cache_path = tmp_path / 'cache.sqlite'
    assert cache_path.exists() is False

    # 'spawn' gives each worker a fresh interpreter instead of inheriting the parent's live SQLite connection and exclusive lock.
    context = get_context('spawn')
    # Each retry event identifies one worker reaching its retry branch, while allow_retry releases both workers together.
    allow_retry = context.Event()
    retry_events = [context.Event(), context.Event()]
    backends: list[Backend] = ['sqlite', 'sqlite']
    processes = [
        context.Process(
            target=_call_cachew_after_wal_retry,
            kwargs={
                'cache_path': cache_path,
                'backend': backend,
                'retry_entered': retry_entered,
                'allow_retry': allow_retry,
            },
            # Explicit reaping is still required, but daemon mode is a final safeguard against a stuck retry loop blocking interpreter shutdown.
            daemon=True,
        )
        for backend, retry_entered in zip(backends, retry_events, strict=True)
    ]
    # Only successfully started processes need joining or termination.
    started_processes: list[BaseProcess] = []
    # A forced process failed to exit during the initial graceful join and required terminate().
    forced_processes: list[str] = []
    # A surviving process remained alive even after the terminate/kill cleanup sequence.
    surviving_processes: list[str] = []
    # Exit codes prove that both planned workers completed successfully instead of merely disappearing.
    exit_codes: list[int | None] = []

    def reap_processes() -> None:
        """Join every started worker, escalate bounded cleanup when needed, and retain failure diagnostics."""

        # Give released workers a bounded opportunity to complete normally.
        for process in started_processes:
            process.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        # Any timeout is a test failure, but terminate the worker so pytest cannot hang during shutdown.
        for process in started_processes:
            if process.is_alive():
                forced_processes.append(process.name)
                process.terminate()
        for process in started_processes:
            process.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        # Escalate after the graceful termination period.
        for process in started_processes:
            process.kill()
        for process in started_processes:
            process.join(timeout=_PROCESS_TIMEOUT_SECONDS)

        surviving_processes.extend(process.name for process in started_processes if process.is_alive())
        exit_codes.extend(process.exitcode for process in processes)
        for process in processes:
            if process.is_alive() is False:
                process.close()

    with ExitStack() as cleanup:
        # ExitStack is LIFO, so it rolls back and closes the blocker, releases both retry loops, and reaps the workers last.
        cleanup.callback(reap_processes)
        cleanup.callback(allow_retry.set)
        # 'blocker' because its exclusive transaction deliberately prevents both constructors from changing journal mode.
        blocker = sqlite3.connect(
            cache_path,
            timeout=0.0,
            autocommit=cast(bool, sqlite3.LEGACY_TRANSACTION_CONTROL),
        )
        cleanup.callback(blocker.close)
        cleanup.callback(blocker.rollback)

        with closing(blocker.execute('PRAGMA journal_mode')) as cursor:
            assert cursor.fetchone() == ('delete',)
        # An exclusive lock forces both constructors through their actual WAL retry branches.
        blocker.execute('BEGIN EXCLUSIVE').close()

        for backend, process, retry_entered in zip(
            backends,
            processes,
            retry_events,
            strict=True,
        ):
            process.start()
            started_processes.append(process)
            # Observe this worker's retry before starting the next process so pair order remains meaningful.
            assert retry_entered.wait(timeout=_PROCESS_TIMEOUT_SECONDS), (
                backend,
                process.exitcode,
            )

    assert forced_processes == []
    assert surviving_processes == []
    assert exit_codes == [0, 0]

    # Check the workers' persisted mode before those reads perform WAL setup and could mask a worker failure.
    with closing(sqlite3.connect(cache_path)) as connection:
        with closing(connection.execute('PRAGMA journal_mode')) as cursor:
            assert cursor.fetchone() == ('wal',)

    def assert_cache_hit(*, reader_backend: Backend) -> None:
        source_calls = 0

        @cachew(cache_path=cache_path, force_file=True, backend=reader_backend)
        def read_cache(version: int) -> Iterator[int]:  # noqa: ARG001
            nonlocal source_calls
            source_calls += 1
            yield -1

        # A public cache hit proves that one contending first call published a complete, reusable result.
        assert list(read_cache(version=1)) == [10, 11]
        assert source_calls == 0

    for reader_backend in backends:
        assert_cache_hit(reader_backend=reader_backend)


def test_sqlite_backend_publication_is_atomic_for_existing_reader(
    *,
    tmp_path: Path,
) -> None:
    backend: Backend = 'sqlite'
    cache_path = tmp_path / 'cache.sqlite'
    old_hash = 'old-hash'
    old_blobs = [b'old-1', b'old-2']
    new_hash = 'new-hash'
    new_blobs = [b'new-1', b'new-2', b'new-3']
    _publish(
        cache_path=cache_path,
        backend=backend,
        source_hash=old_hash,
        blobs=old_blobs,
    )

    with (
        _backend(cache_path=cache_path, backend=backend) as old_reader,
        _backend(cache_path=cache_path, backend=backend) as concurrent_reader,
    ):
        # This SELECT pins old_reader to the snapshot from before publication starts.
        assert old_reader.get_old_hash() == old_hash

        with _backend(cache_path=cache_path, backend=backend) as writer:
            assert writer.get_old_hash() == old_hash
            assert writer.get_exclusive_write()
            writer.flush_blobs(chunk=new_blobs)
            writer.finalize(new_hash)

            # A separate reader must not observe the finalized tables until the writer commits.
            assert concurrent_reader.get_old_hash() == old_hash
            assert concurrent_reader.cached_blobs_total() == len(old_blobs)
            assert list(concurrent_reader.cached_blobs()) == old_blobs

        # After the writer commits, the existing reader must still see the previously committed hash and blobs.
        assert old_reader.get_old_hash() == old_hash
        assert old_reader.cached_blobs_total() == len(old_blobs)
        assert list(old_reader.cached_blobs()) == old_blobs

    _assert_cache(
        cache_path=cache_path,
        backend=backend,
        source_hash=new_hash,
        blobs=new_blobs,
    )


def test_sqlite_backend_finalize_is_rolled_back(
    *,
    tmp_path: Path,
) -> None:
    backend: Backend = 'sqlite'

    class ForcedRollback(Exception):
        pass

    cache_path = tmp_path / 'cache.sqlite'
    old_hash = 'old-hash'
    old_blobs = [b'old-1', b'old-2']
    new_hash = 'new-hash'
    new_blobs = [b'new-1', b'new-2', b'new-3']
    _publish(
        cache_path=cache_path,
        backend=backend,
        source_hash=old_hash,
        blobs=old_blobs,
    )

    def finalize_then_rollback() -> None:
        with _backend(cache_path=cache_path, backend=backend) as writer:
            assert writer.get_old_hash() == old_hash
            assert writer.get_exclusive_write()
            writer.flush_blobs(chunk=new_blobs)
            writer.finalize(new_hash)

            # Prove finalize completed inside the transaction before forcing its rollback.
            assert writer.get_old_hash() == new_hash
            assert writer.cached_blobs_total() == len(new_blobs)
            assert list(writer.cached_blobs()) == new_blobs
            raise ForcedRollback

    with pytest.raises(ForcedRollback):
        finalize_then_rollback()

    _assert_cache(
        cache_path=cache_path,
        backend=backend,
        source_hash=old_hash,
        blobs=old_blobs,
    )


def test_sqlite_partial_write_close_releases_lock(
    *,
    tmp_path: Path,
) -> None:
    backend: Backend = 'sqlite'

    class ProbeRollback(Exception):
        pass

    cache_path = tmp_path / 'cache.sqlite'
    source_calls: list[int] = []

    # One-item chunks ensure resuming for the second item flushes the first item before suspension.
    @cachew(cache_path=cache_path, force_file=True, backend=backend, chunk_by=1)
    def items(version: int) -> Iterator[int]:
        source_calls.append(version)
        yield version * 10
        yield version * 10 + 1
        yield version * 10 + 2

    old_items = [10, 11, 12]
    new_items = [20, 21, 22]
    assert list(items(version=1)) == old_items
    assert source_calls == [1]
    with _backend(cache_path=cache_path, backend=backend) as reader:
        old_hash = reader.get_old_hash()
        assert old_hash is not None

    with _backend(cache_path=cache_path, backend=backend) as blocked_writer:
        partial_write = cast(Generator[int, None, None], items(version=2))
        try:
            assert next(partial_write) == new_items[0]
            # Resuming after the first yield flushes its one-item chunk before yielding the second item.
            assert next(partial_write) == new_items[1]
            assert source_calls == [1, 2]

            assert blocked_writer.get_old_hash() == old_hash
            assert blocked_writer.get_exclusive_write() is False
        finally:
            partial_write.close()

    def acquire_write_then_rollback() -> None:
        with _backend(cache_path=cache_path, backend=backend) as available_writer:
            assert available_writer.get_old_hash() == old_hash
            assert available_writer.get_exclusive_write() is True
            raise ProbeRollback

    with pytest.raises(ProbeRollback):
        acquire_write_then_rollback()

    # Closing the interrupted v2 refresh must leave the existing v1 cache readable without recomputation.
    assert list(items(version=1)) == old_items
    assert source_calls == [1, 2]

    # A complete v2 retry must publish once and make the following call a cache hit.
    assert list(items(version=2)) == new_items
    assert source_calls == [1, 2, 2]
    assert list(items(version=2)) == new_items
    assert source_calls == [1, 2, 2]


@pytest.mark.parametrize('throw_on_error', [False, True], ids=['fallback', 'strict'])
def test_cachew_handles_non_lock_wal_failure(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    throw_on_error: bool,
) -> None:
    """
    A genuine SQLite setup failure must follow Cachew's configured error policy instead of being retried forever as transient contention.
    Strict mode must raise the original error without running the source, while defensive mode must run the source uncached; either path must close the failed connection and allow later calls to populate and reuse the cache.
    """

    monkeypatch.setattr(settings, 'THROW_ON_ERROR', throw_on_error)
    real_connect = sqlite3.connect
    wal_error = sqlite3.OperationalError('forced WAL failure')
    # The explicit SQLite code sends this error through the non-retry branch regardless of its message.
    setattr(wal_error, 'sqlite_errorcode', sqlite3.SQLITE_IOERR)

    # A narrow fake records the constructor's behavior without replacing methods on the C connection type.
    class FailingConnection:
        def __init__(self) -> None:
            self.execute_calls = 0
            self.close_calls = 0

        def execute(self, statement: str) -> sqlite3.Cursor:
            assert statement == 'PRAGMA journal_mode=WAL'
            # A retry would mean the non-lock IO error was misclassified and would otherwise loop forever.
            assert self.execute_calls == 0
            self.execute_calls += 1
            raise wal_error

        def close(self) -> None:
            self.close_calls += 1

    connection = FailingConnection()

    def connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        return cast(sqlite3.Connection, connection)

    monkeypatch.setattr(sqlite.sqlite3, 'connect', connect)
    cache_path = tmp_path / 'cache.sqlite'
    source_calls = 0

    @cachew(cache_path=cache_path, force_file=True, backend='sqlite')
    def items() -> Iterator[int]:
        nonlocal source_calls
        source_calls += 1
        yield 1

    if throw_on_error:
        with pytest.raises(sqlite3.OperationalError) as raised:
            list(items())
        assert raised.value is wal_error
        assert source_calls == 0
    else:
        assert list(items()) == [1]
        assert source_calls == 1

    # Raw SQLite preserves the original error and uses a note only to attach the cache path.
    assert getattr(wal_error, '__notes__') == [f'while setting WAL mode on cache {cache_path}']
    assert connection.execute_calls == 1
    assert connection.close_calls == 1

    # Restore real SQLite, then prove that the next call can write the cache and the following call can read it without running the source.
    monkeypatch.setattr(sqlite.sqlite3, 'connect', real_connect)
    expected_source_calls = source_calls + 1
    assert list(items()) == [1]
    assert source_calls == expected_source_calls
    assert list(items()) == [1]
    assert source_calls == expected_source_calls


@pytest.mark.parametrize('throw_on_error', [False, True], ids=['fallback', 'strict'])
def test_cachew_recovers_after_sqlite_transaction_entry_failure(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    throw_on_error: bool,
) -> None:
    """
    A failed cache transaction must raise before running the source in strict mode or run it once through defensive fallback.
    Closing the unusable connection must also let the next call populate the cache and the following call reuse it.
    """

    monkeypatch.setattr(settings, 'THROW_ON_ERROR', throw_on_error)
    fail_next_entry = True
    failed_backends: list[sqlite.SqliteBackend] = []
    failed_connections: list[sqlite3.Connection] = []

    class FailFirstTransactionEntry(sqlite.SqliteBackend):
        def __init__(self, cache_path: Path, *, logger: logging.Logger) -> None:
            nonlocal fail_next_entry
            super().__init__(cache_path, logger=logger)
            if fail_next_entry:
                fail_next_entry = False
                connection = self.connection
                assert connection is not None
                failed_backends.append(self)
                failed_connections.append(connection)
                # This transaction makes the BEGIN in the inherited __enter__ fail naturally.
                # The inherited BEGIN will fail, and any leaked connection would keep this write lock and break recovery below.
                connection.execute('BEGIN IMMEDIATE').close()

    monkeypatch.setitem(BACKENDS, 'sqlite', FailFirstTransactionEntry)
    cache_path = tmp_path / 'cache.sqlite'
    source_calls = 0

    @cachew(cache_path=cache_path, force_file=True, backend='sqlite')
    def items() -> Iterator[int]:
        nonlocal source_calls
        source_calls += 1
        yield 1

    if throw_on_error:
        with pytest.raises(sqlite3.OperationalError, match='cannot start a transaction within a transaction'):
            list(items())
        assert source_calls == 0
    else:
        assert list(items()) == [1]
        assert source_calls == 1

    [failed_backend] = failed_backends
    [failed_connection] = failed_connections
    assert failed_backend.connection is None
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        failed_connection.execute('SELECT 1')

    expected_source_calls = source_calls + 1
    assert list(items()) == [1]
    assert source_calls == expected_source_calls
    assert list(items()) == [1]
    assert source_calls == expected_source_calls


@pytest.mark.parametrize(
    ('error_code', 'expected'),
    [
        pytest.param(sqlite3.SQLITE_BUSY             , True , id='busy'),
        pytest.param(sqlite3.SQLITE_LOCKED           , True , id='locked'),
        # SQLite stores extended error details above the low-byte primary result code.
        pytest.param(sqlite3.SQLITE_BUSY   | (1 << 8), True , id='extended-busy'),
        pytest.param(sqlite3.SQLITE_LOCKED | (1 << 8), True , id='extended-locked'),
        pytest.param(sqlite3.SQLITE_ERROR            , False, id='error'),
        pytest.param(sqlite3.SQLITE_IOERR  | (1 << 8), False, id='extended-io-error'),
    ],
)  # fmt: skip
def test_sqlite_lock_error_classification(
    *,
    error_code: int,
    expected: bool,
) -> None:
    """
    This deliberately remains a unit test because manufacturing every synthetic extended result code through Cachew would obscure the classification rule.
    Cachew treats only SQLite lock contention as recoverable: WAL setup retries it, and write contention runs the source without updating the cache.
    Unrelated failures must propagate to Cachew's normal strict-mode or defensive-fallback handling instead of being mistaken for locks.
    Extended SQLite result codes retain that classification through their low-byte primary code.
    """

    error = sqlite3.OperationalError('forced error')
    setattr(error, 'sqlite_errorcode', error_code)

    assert sqlite._is_lock_error(error) is expected

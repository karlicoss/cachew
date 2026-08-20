import logging
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from types import TracebackType
from typing import Self, cast, override

from ..common import SourceHash
from .common import AbstractBackend

_WAL_LOCK_RETRY_INTERVAL_SECONDS = 0.1


def _is_lock_error(error: sqlite3.OperationalError) -> bool:
    primary_code = error.sqlite_errorcode & 0xFF
    return primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


class SqliteBackend(AbstractBackend):
    def __init__(self, cache_path: Path, *, logger: logging.Logger) -> None:
        self.logger = logger
        self.cache_path = cache_path
        connection = sqlite3.connect(
            cache_path,
            # Keep timeout at zero so recursive and concurrent paths fail fast instead of waiting on the lock for seconds.
            # `test_recursive_deep` covers recursive lock loss, and `test_sqlite_locked_write_falls_back_to_uncached_and_recovers` covers locked write recovery.
            timeout=0.0,
            # Pin legacy mode because Python's future `autocommit=False` default would open a transaction before WAL setup and `__enter__`.
            # `test_transaction[sqlite]` covers the required explicit commit and rollback behavior.
            #   see https://docs.python.org/3/library/sqlite3.html#transaction-control-via-the-autocommit-attribute
            autocommit=cast(bool, sqlite3.LEGACY_TRANSACTION_CONTROL),
        )
        try:
            self._set_wal(connection)
        except BaseException:
            connection.close()
            raise
        self.connection: sqlite3.Connection | None = connection
        self._new_hash: SourceHash | None = None
        self._max_blobs_per_insert = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        self._insert_sql_text_by_size: dict[int, str] = {}

    def _set_wal(self, connection: sqlite3.Connection) -> None:
        # Retry transient lock contention indefinitely to preserve the existing Cachew behavior.
        # TODO consider a bounded policy so a persistently locked cache cannot hang indefinitely.
        while True:
            try:
                cursor = connection.execute('PRAGMA journal_mode=WAL')
            except sqlite3.OperationalError as error:
                if not _is_lock_error(error):
                    error.add_note(f'while setting WAL mode on cache {self.cache_path}')
                    raise
                time.sleep(_WAL_LOCK_RETRY_INTERVAL_SECONDS)
                continue

            with closing(cursor):
                row = cursor.fetchone()
            assert row == ('wal',), (self.cache_path, row)
            return

    def _require_connection(self) -> sqlite3.Connection:
        conn = self.connection
        assert conn is not None
        return conn

    def _insert_sql_text(self, *, batch_size: int) -> str:
        """Return cached SQL text for inserting one batch as a multi-row statement.

        The cache avoids rebuilding placeholder text for repeated full-sized chunks.
        The sqlite3 connection separately caches the compiled statement for identical SQL text.
        """
        sql = self._insert_sql_text_by_size.get(batch_size)
        if sql is None:
            placeholders = ', '.join('(?)' for _ in range(batch_size))
            sql = f'INSERT INTO cache_tmp (data) VALUES {placeholders}'
            self._insert_sql_text_by_size[batch_size] = sql
        return sql

    @override
    def __enter__(self) -> Self:
        conn = self._require_connection()
        try:
            conn.execute('BEGIN DEFERRED').close()
        except BaseException:
            conn.close()
            self.connection = None
            raise
        return self

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        conn = self.connection
        self.connection = None
        if conn is None:
            return
        try:
            if exc_type is None:
                conn.commit()
            else:
                conn.rollback()
        finally:
            conn.close()

    @override
    def get_old_hash(self) -> SourceHash | None:
        conn = self._require_connection()
        try:
            cursor = conn.execute('SELECT value FROM hash')
        except sqlite3.OperationalError as e:
            if 'no such table: hash' in str(e):
                return None
            raise

        with closing(cursor):
            rows = cursor.fetchall()

        assert len(rows) <= 1, rows
        if len(rows) == 0:
            return None
        return rows[0][0]

    @override
    def cached_blobs_total(self) -> int | None:
        conn = self._require_connection()
        with closing(conn.execute('SELECT COUNT(*) FROM cache')) as cursor:
            row = cursor.fetchone()
        assert row is not None
        (total,) = row
        return total

    @override
    def cached_blobs(self) -> Iterator[bytes]:
        conn = self._require_connection()
        with closing(conn.execute('SELECT data FROM cache')) as cursor:
            for (blob,) in cursor:
                yield blob

    @override
    def start_write(self, *, new_hash: SourceHash) -> bool:
        conn = self._require_connection()
        try:
            # One of these schema statements upgrades BEGIN DEFERRED to a write transaction.
            conn.execute('CREATE TABLE IF NOT EXISTS hash (value TEXT)').close()
            conn.execute('DROP TABLE IF EXISTS `table`').close()
            conn.execute('DROP TABLE IF EXISTS cache_tmp').close()
            conn.execute('CREATE TABLE cache_tmp (data BLOB)').close()
        except sqlite3.OperationalError as e:
            if _is_lock_error(e):
                conn.close()
                self.connection = None
                return False
            e.add_note(f'while acquiring a write transaction on cache {self.cache_path}')
            raise
        self._new_hash = new_hash
        return True

    @override
    def flush_blobs(self, chunk: Sequence[bytes]) -> None:
        conn = self._require_connection()
        for offset in range(0, len(chunk), self._max_blobs_per_insert):
            batch = chunk[offset : offset + self._max_blobs_per_insert]
            batch_size = len(batch)
            sql = self._insert_sql_text(batch_size=batch_size)

            # executemany() steps and resets a single-row statement once per blob.
            # One multi-row statement amortizes that overhead while keeping the same one-row-per-item cache format.
            conn.execute(sql, batch).close()

    @override
    def finalize(self) -> None:
        new_hash = self._new_hash
        assert new_hash is not None
        conn = self._require_connection()
        conn.execute('DELETE FROM hash').close()
        conn.execute('DROP TABLE IF EXISTS cache').close()
        conn.execute('ALTER TABLE cache_tmp RENAME TO cache').close()
        conn.execute('INSERT INTO hash (value) VALUES (?)', (new_hash,)).close()
        self._new_hash = None

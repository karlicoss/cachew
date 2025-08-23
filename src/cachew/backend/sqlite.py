import logging
import sqlite3
import time
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import sqlalchemy
import sqlalchemy.exc
from sqlalchemy import Column, Table, event, text
from sqlalchemy.dialects import sqlite

from ..common import SourceHash
from .common import AbstractBackend


class SqliteBackend(AbstractBackend):
    def __init__(self, cache_path: Path, *, logger: logging.Logger) -> None:
        self.logger = logger
        self.engine = sqlalchemy.create_engine(f'sqlite:///{cache_path}', connect_args={'timeout': 0})
        # NOTE: timeout is necessary so we don't lose time waiting during recursive calls
        # by default, it's several seconds? you'd see 'test_recursive' test performance degrade

        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ARG001
            # without wal, concurrent reading/writing is not gonna work

            # ugh. that's odd, how are we supposed to set WAL if the very fact of setting wal might lock the db?
            while True:
                try:
                    dbapi_connection.execute('PRAGMA journal_mode=WAL')
                    break
                except sqlite3.OperationalError as oe:
                    if 'database is locked' not in str(oe):
                        # ugh, pretty annoying that exception doesn't include database path for some reason
                        raise RuntimeError(f'Error while setting WAL on {cache_path}') from oe
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

        # fmt: off
        # actual cache
        self.table_cache     = Table('cache'    , self.meta, Column('data', sqlalchemy.BLOB))
        # temporary table, we use it to insert and then (atomically?) rename to the above table at the very end
        self.table_cache_tmp = Table('cache_tmp', self.meta, Column('data', sqlalchemy.BLOB))
        # fmt: on

    def __enter__(self) -> 'SqliteBackend':
        # NOTE: deferred transaction
        self.transaction = self.connection.begin()
        # FIXME this is a bit crap.. is there a nicer way to use another ctx manager here?
        self.transaction.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.transaction.__exit__(*args)
        self.connection.close()
        self.engine.dispose()

    def get_old_hash(self) -> SourceHash | None:
        # first, try to do as much as possible read-only, benefiting from deferred transaction
        old_hashes: Sequence
        try:
            # not sure if there is a better way...
            cursor = self.connection.execute(self.table_hash.select())
        except sqlalchemy.exc.OperationalError as e:
            # meh. not sure if this is a good way to handle this..
            if 'no such table: hash' in str(e):
                old_hashes = []
            else:
                raise e
        else:
            old_hashes = cursor.fetchall()

        assert len(old_hashes) <= 1, old_hashes  # shouldn't happen

        old_hash: SourceHash | None
        if len(old_hashes) == 0:
            old_hash = None
        else:
            old_hash = old_hashes[0][0]  # returns a tuple...
        return old_hash

    def cached_blobs_total(self) -> int | None:
        [(total,)] = self.connection.execute(sqlalchemy.select(sqlalchemy.func.count()).select_from(self.table_cache))
        return total

    def cached_blobs(self) -> Iterator[bytes]:
        rows = self.connection.execute(self.table_cache.select())
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
                "CursorResult._raw_row_iterator method isn't found. This could lead to degraded cache reading performance.", stacklevel=2
            )
            row_iterator = rows
        else:
            row_iterator = raw_row_iterator()

        for (blob,) in row_iterator:
            yield blob

    def get_exclusive_write(self) -> bool:
        # NOTE on recursive calls
        # somewhat magically, they should work as expected with no extra database inserts?
        # the top level call 'wins' the write transaction and once it's gathered all data, will write it
        # the 'intermediate' level calls fail to get it and will pass data through
        # the cached 'bottom' level is read only and will be yielded without a write transaction
        try:
            # first 'write' statement will upgrade transaction to write transaction which might fail due to concurrency
            # see https://www.sqlite.org/lang_transaction.html
            # NOTE: because of 'checkfirst=True', only the last .create will guarantee the transaction upgrade to write transaction
            self.table_hash.create(self.connection, checkfirst=True)

            # 'table' used to be old 'cache' table name, so we just delete it regardless
            # otherwise it might overinfalte the cache db with stale values
            self.connection.execute(text('DROP TABLE IF EXISTS `table`'))

            # NOTE: we have to use .drop and then .create (e.g. instead of some sort of replace)
            # since it's possible to have schema changes inbetween calls
            # checkfirst=True because it might be the first time we're using cache
            self.table_cache_tmp.drop(self.connection, checkfirst=True)
            self.table_cache_tmp.create(self.connection)
        except sqlalchemy.exc.OperationalError as e:
            if e.code == 'e3q8' and 'database is locked' in str(e):
                # someone else must be have won the write lock
                # not much we can do here
                # NOTE: important to close early, otherwise we might hold onto too many file descriptors during yielding
                # see test_recursive_deep
                # (normally connection is closed in SqliteBackend.__exit__)
                self.connection.close()
                # in this case all the callee can do is just to call the actual function
                return False
            else:
                raise e
        return True

    def flush_blobs(self, chunk: Sequence[bytes]) -> None:
        # uhh. this gives a huge speedup for inserting
        # since we don't have to create intermediate dictionaries
        # TODO move this to __init__?
        insert_into_table_cache_tmp_raw = str(self.table_cache_tmp.insert().compile(dialect=sqlite.dialect(paramstyle='qmark')))
        # I also tried setting paramstyle='qmark' in create_engine, but it seems to be ignored :(
        # idk what benefit sqlalchemy gives at this point, seems to just complicate things
        self.connection.exec_driver_sql(insert_into_table_cache_tmp_raw, [(c,) for c in chunk])

    def finalize(self, new_hash: SourceHash) -> None:
        # delete hash first, so if we are interrupted somewhere, it mismatches next time and everything is recomputed
        # pylint: disable=no-value-for-parameter
        self.connection.execute(self.table_hash.delete())

        # checkfirst is necessary since it might not have existed in the first place
        # e.g. first time we use cache
        self.table_cache.drop(self.connection, checkfirst=True)

        # meh https://docs.sqlalchemy.org/en/14/faq/metadata_schema.html#does-sqlalchemy-support-alter-table-create-view-create-trigger-schema-upgrade-functionality
        # also seems like sqlalchemy doesn't have any primitives to escape table names.. sigh
        self.connection.execute(text(f"ALTER TABLE `{self.table_cache_tmp.name}` RENAME TO `{self.table_cache.name}`"))

        # pylint: disable=no-value-for-parameter
        self.connection.execute(self.table_hash.insert().values([{'value': new_hash}]))

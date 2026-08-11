from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import cast

import pytest

from .. import BACKENDS, Backend, cachew, get_logger
from ..backend.common import AbstractBackend
from ..common import SourceHash

_SQLITE_BACKEND_PAIRS: list[tuple[Backend, Backend]] = [
    ('sqlite', 'sqlite'),
    ('sqlite', 'sqlite_raw'),
    ('sqlite_raw', 'sqlite'),
    ('sqlite_raw', 'sqlite_raw'),
]


def _backend(*, cache_path: Path, backend: Backend) -> AbstractBackend:
    return BACKENDS[backend](cache_path=cache_path, logger=get_logger())


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


@pytest.mark.parametrize(('writer_backend', 'reader_backend'), _SQLITE_BACKEND_PAIRS)
def test_sqlite_backend_cache_compatibility(
    *,
    tmp_path: Path,
    writer_backend: Backend,
    reader_backend: Backend,
) -> None:
    cache_path = tmp_path / 'cache.sqlite'
    writer_calls = 0
    reader_calls = 0

    @cachew(cache_path=cache_path, force_file=True, backend=writer_backend)
    def write_cache(version: int) -> Iterator[int]:  # noqa: ARG001
        nonlocal writer_calls
        writer_calls += 1
        yield 1
        yield 2

    @cachew(cache_path=cache_path, force_file=True, backend=reader_backend)
    def read_cache(version: int) -> Iterator[int]:  # noqa: ARG001
        nonlocal reader_calls
        reader_calls += 1
        yield -1

    assert list(write_cache(version=1)) == [1, 2]
    assert writer_calls == 1

    assert list(read_cache(version=1)) == [1, 2]
    assert reader_calls == 0


@pytest.mark.parametrize(('reader_backend', 'writer_backend'), _SQLITE_BACKEND_PAIRS)
def test_sqlite_backend_publication_is_atomic_for_existing_reader(
    *,
    tmp_path: Path,
    reader_backend: Backend,
    writer_backend: Backend,
) -> None:
    cache_path = tmp_path / 'cache.sqlite'
    old_hash = 'old-hash'
    old_blobs = [b'old-1', b'old-2']
    new_hash = 'new-hash'
    new_blobs = [b'new-1', b'new-2', b'new-3']
    _publish(
        cache_path=cache_path,
        backend=writer_backend,
        source_hash=old_hash,
        blobs=old_blobs,
    )

    with (
        _backend(cache_path=cache_path, backend=reader_backend) as old_reader,
        _backend(cache_path=cache_path, backend=reader_backend) as concurrent_reader,
    ):
        # This SELECT pins old_reader to the snapshot from before publication starts.
        assert old_reader.get_old_hash() == old_hash

        with _backend(cache_path=cache_path, backend=writer_backend) as writer:
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
        backend=reader_backend,
        source_hash=new_hash,
        blobs=new_blobs,
    )


@pytest.mark.parametrize('backend', ['sqlite', 'sqlite_raw'])
def test_sqlite_backend_finalize_is_rolled_back(
    *,
    tmp_path: Path,
    backend: Backend,
) -> None:
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


@pytest.mark.parametrize('backend', ['sqlite', 'sqlite_raw'])
def test_sqlite_partial_write_close_releases_lock(
    *,
    tmp_path: Path,
    backend: Backend,
) -> None:
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

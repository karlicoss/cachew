from collections.abc import Iterator
from pathlib import Path

import pytest

from .. import Backend, cachew


@pytest.mark.parametrize(
    ('writer_backend', 'reader_backend'),
    [
        ('sqlite', 'sqlite_raw'),
        ('sqlite_raw', 'sqlite'),
    ],
)
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

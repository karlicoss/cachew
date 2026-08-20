from collections.abc import Iterator
from pathlib import Path

import pytest

from .. import BACKENDS, cachew, settings
from ..backend.file import FileBackend
from ..common import CacheWriteError, SourceHash


@pytest.mark.parametrize('throw_on_error', [False, True], ids=['fallback', 'strict'])
def test_file_cache_header_write_error_obeys_cachew_policy_and_recovers(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, throw_on_error: bool
) -> None:
    """
    A file-header failure happens before Cachew consumes the source, so defensive mode may run it uncached.
    """
    monkeypatch.setattr(settings, 'THROW_ON_ERROR', throw_on_error)

    class FailingHeaderBackend(FileBackend):
        def write_new_hash(self, new_hash: SourceHash) -> None:
            del new_hash
            raise RuntimeError('failed to write cache header')

    monkeypatch.setitem(BACKENDS, 'file', FailingHeaderBackend)

    source_calls = 0
    expected = [1, 2]

    @cachew(cache_path=tmp_path / 'cache', force_file=True, backend='file')
    def fun() -> Iterator[int]:
        nonlocal source_calls
        source_calls += 1
        yield from expected

    if throw_on_error:
        with pytest.raises(CacheWriteError) as exc:
            list(fun())
        assert str(exc.value.__cause__) == 'failed to write cache header'
        assert source_calls == 0
    else:
        assert list(fun()) == expected
        assert source_calls == 1

    calls_before_recovery = source_calls
    monkeypatch.setitem(BACKENDS, 'file', FileBackend)

    # The failed temporary cache must be cleaned up so the next call can publish and reuse a complete cache.
    assert list(fun()) == expected
    assert source_calls == calls_before_recovery + 1
    assert list(fun()) == expected
    assert source_calls == calls_before_recovery + 1

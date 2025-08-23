import logging
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..common import SourceHash


class AbstractBackend:
    @abstractmethod
    def __init__(self, cache_path: Path, *, logger: logging.Logger) -> None:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, *args) -> None:
        raise NotImplementedError

    def get_old_hash(self) -> SourceHash | None:
        raise NotImplementedError

    def cached_blobs_total(self) -> int | None:
        raise NotImplementedError

    def cached_blobs(self) -> Iterator[bytes]:
        raise NotImplementedError

    def get_exclusive_write(self) -> bool:
        '''
        Returns whether it actually managed to get it
        '''
        raise NotImplementedError

    def write_new_hash(self, new_hash: SourceHash) -> None:
        raise NotImplementedError

    def flush_blobs(self, chunk: Sequence[bytes]) -> None:
        raise NotImplementedError

    def finalize(self, new_hash: SourceHash) -> None:
        raise NotImplementedError

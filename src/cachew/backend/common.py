import logging
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path

from ..common import SourceHash


class AbstractBackend(AbstractContextManager):
    @abstractmethod
    def __init__(self, cache_path: Path, *, logger: logging.Logger) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_old_hash(self) -> SourceHash | None:
        raise NotImplementedError

    @abstractmethod
    def cached_blobs_total(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def cached_blobs(self) -> Iterator[bytes]:
        raise NotImplementedError

    @abstractmethod
    def get_exclusive_write(self) -> bool:
        """
        Returns whether it actually managed to get it.
        """
        raise NotImplementedError

    @abstractmethod
    def write_new_hash(self, new_hash: SourceHash) -> None:
        raise NotImplementedError

    @abstractmethod
    def flush_blobs(self, chunk: Sequence[bytes]) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self, new_hash: SourceHash) -> None:
        """
        Atomically commit changes and make them visible to readers.
        """
        raise NotImplementedError

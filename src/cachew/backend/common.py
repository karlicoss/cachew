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
    def start_write(self, *, new_hash: SourceHash) -> bool:
        """
        Prepare an atomic cache replacement and return whether the backend acquired its exclusive write slot.
        """
        raise NotImplementedError

    @abstractmethod
    def flush_blobs(self, chunk: Sequence[bytes]) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        """
        Complete the atomic replacement.
        Transactional backends commit and expose it on successful context exit.
        """
        raise NotImplementedError

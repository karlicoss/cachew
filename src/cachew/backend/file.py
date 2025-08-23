import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import (
    BinaryIO,
)

from ..common import SourceHash
from .common import AbstractBackend


class FileBackend(AbstractBackend):
    jsonl: Path
    jsonl_tmp: Path
    jsonl_fr: BinaryIO | None
    jsonl_tmp_fw: BinaryIO | None

    def __init__(self, cache_path: Path, *, logger: logging.Logger) -> None:
        self.logger = logger
        self.jsonl = cache_path
        self.jsonl_tmp = Path(str(self.jsonl) + '.tmp')

        self.jsonl_fr = None
        self.jsonl_tmp_fw = None

    def __enter__(self) -> 'FileBackend':
        try:
            self.jsonl_fr = self.jsonl.open('rb')
        except FileNotFoundError:
            self.jsonl_fr = None
        return self

    def __exit__(self, *args) -> None:
        if self.jsonl_tmp_fw is not None:
            # might still exist in case of early exit
            self.jsonl_tmp.unlink(missing_ok=True)

            # NOTE: need to unlink first
            # otherwise possible that someone else might open the file before we unlink it
            self.jsonl_tmp_fw.close()

        if self.jsonl_fr is not None:
            self.jsonl_fr.close()

    def get_old_hash(self) -> SourceHash | None:
        if self.jsonl_fr is None:
            return None
        hash_line = self.jsonl_fr.readline().rstrip(b'\n')
        return hash_line.decode('utf8')

    def cached_blobs_total(self) -> int | None:
        # not really sure how to support that for a plaintext file?
        # could wc -l but it might be costly..
        return None

    def cached_blobs(self) -> Iterator[bytes]:
        assert self.jsonl_fr is not None  # should be guaranteed by get_old_hash
        yield from self.jsonl_fr  # yields line by line

    def get_exclusive_write(self) -> bool:
        # NOTE: opening in x (exclusive write) mode just in case, so it throws if file exists
        try:
            self.jsonl_tmp_fw = self.jsonl_tmp.open('xb')
        except FileExistsError:
            self.jsonl_tmp_fw = None
            return False
        else:
            return True

    def write_new_hash(self, new_hash: SourceHash) -> None:
        assert self.jsonl_tmp_fw is not None
        self.jsonl_tmp_fw.write(new_hash.encode('utf8') + b'\n')

    def flush_blobs(self, chunk: Sequence[bytes]) -> None:
        fw = self.jsonl_tmp_fw
        assert fw is not None
        for blob in chunk:
            fw.write(blob)
            fw.write(b'\n')

    def finalize(self, new_hash: SourceHash) -> None:  # noqa: ARG002
        # TODO defensive??
        self.jsonl_tmp.rename(self.jsonl)

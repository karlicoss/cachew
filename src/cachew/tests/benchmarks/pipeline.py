import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Literal, assert_never, cast, get_args

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from .common import (
    BENCHMARK_COUNT,
    CASE_SPECS,
    CaseSpec,
    Impl,
    Impls,
    attach_case_metadata,
    benchmark_pedantic,
    make_case,
    skip_unsupported_impl,
)

COUNT_ID = (
    f'{BENCHMARK_COUNT // 1000}k' if BENCHMARK_COUNT >= 1000 and BENCHMARK_COUNT % 1000 == 0 else str(BENCHMARK_COUNT)
)
COUNT_PARAM = pytest.mark.parametrize('count', [BENCHMARK_COUNT], ids=[COUNT_ID])
SPEC_PARAM = pytest.mark.parametrize('spec', CASE_SPECS, ids=lambda spec: spec.id)
IMPL_PARAM = pytest.mark.parametrize('impl', Impls)
type Storage = Literal['sqlite', 'file']
STORAGES = cast(Sequence[Storage], get_args(Storage.__value__))
STORAGE_PARAM = pytest.mark.parametrize('storage', STORAGES)
DISABLE_GC = pytest.mark.benchmark(disable_gc=True)


def _sample(values: Sequence[object], *, sample_size: int = 100) -> list[object]:
    return list(values[:sample_size]) + list(values[-sample_size:])


def _group_name(request: pytest.FixtureRequest, spec: CaseSpec) -> str:
    original_name = request.node.originalname
    assert original_name is not None
    assert original_name.startswith('test_')
    stage = original_name[len('test_') :]
    return f'{spec.id}-{stage}'


def _sqlite_db_path(tmp_path: Path, *, spec: CaseSpec, impl: Impl, count: int) -> Path:
    return tmp_path / f'{spec.id}-{impl}-{count}.sqlite'


def _file_path(tmp_path: Path, *, spec: CaseSpec, impl: Impl, count: int) -> Path:
    return tmp_path / f'{spec.id}-{impl}-{count}.jsonl'


def _storage_path(tmp_path: Path, *, spec: CaseSpec, impl: Impl, count: int, storage: Storage) -> Path:
    if storage == 'sqlite':
        return _sqlite_db_path(tmp_path, spec=spec, impl=impl, count=count)
    if storage == 'file':
        return _file_path(tmp_path, spec=spec, impl=impl, count=count)
    assert_never(storage)


def _sqlite_dump(db: Path, blobs: list[bytes]) -> None:
    db.unlink(missing_ok=True)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute('CREATE TABLE data (value BLOB)')
        conn.executemany('INSERT INTO data (value) VALUES (?)', [(blob,) for blob in blobs])


def _sqlite_load(db: Path) -> list[bytes]:
    with closing(sqlite3.connect(db)) as conn, conn:
        return [value for (value,) in conn.execute('SELECT value FROM data')]


def _file_dump(path: Path, blobs: list[bytes]) -> None:
    path.unlink(missing_ok=True)
    # NOTE: tried an os.writev() implementation here, but it was a slowdown in
    # practice because the extra Python-side bookkeeping outweighed any syscall
    # reduction for this newline-delimited benchmark helper.
    with path.open('wb') as fw:
        write = fw.write
        for blob in blobs:
            write(blob)
            write(b'\n')


def _file_load(path: Path) -> list[bytes]:
    with path.open('rb') as fr:
        return [line[:-1] for line in fr]


def _storage_dump(path: Path, blobs: list[bytes], *, storage: Storage) -> None:
    if storage == 'sqlite':
        _sqlite_dump(path, blobs)
        return
    if storage == 'file':
        _file_dump(path, blobs)
        return
    assert_never(storage)


def _storage_load(path: Path, *, storage: Storage) -> list[bytes]:
    if storage == 'sqlite':
        return _sqlite_load(path)
    if storage == 'file':
        return _file_load(path)
    assert_never(storage)


def _supports_delimited_file_storage(impl: Impl) -> bool:
    return impl not in {'pickle', 'msgspec-msgpack'}


def skip_unsupported_storage(*, storage: Storage, impl: Impl) -> None:
    if storage == 'file' and not _supports_delimited_file_storage(impl):
        pytest.skip(
            'raw file storage uses newline-delimited framing and is unsupported for binary blob implementations'
        )


def attach_storage_metadata(benchmark: BenchmarkFixture, *, storage: Storage) -> None:
    benchmark.extra_info['storage'] = storage


@COUNT_PARAM
@SPEC_PARAM
@DISABLE_GC
def test_01_build(benchmark: BenchmarkFixture, request: pytest.FixtureRequest, count: int, spec: CaseSpec) -> None:
    benchmark.group = _group_name(request, spec)
    benchmark.extra_info['count'] = count
    benchmark.extra_info['shape'] = spec.id
    benchmark.extra_info['impl'] = 'baseline'
    benchmark.extra_info['operation'] = 'build'
    benchmark.extra_info['serialized_total_bytes'] = 0
    benchmark.extra_info['serialized_avg_bytes'] = 0.0

    result = benchmark_pedantic(benchmark, spec.build_objects, count)

    assert len(result) == count


@COUNT_PARAM
@SPEC_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_02_encode(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='encode', spec=spec, case=case)

    result = benchmark_pedantic(benchmark, case.serialize_all)

    assert _sample(result) == _sample(case.payloads)


@COUNT_PARAM
@SPEC_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_03_blob_dump(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='blob-dump', spec=spec, case=case)

    result = benchmark_pedantic(benchmark, case.dump_blobs_all)

    assert _sample(result) == _sample(case.blobs)


@COUNT_PARAM
@SPEC_PARAM
@STORAGE_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_04_storage_dump(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    count: int,
    spec: CaseSpec,
    impl: Impl,
    storage: Storage,
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    skip_unsupported_storage(storage=storage, impl=impl)
    case = make_case(spec, count=count, impl=impl)
    path = _storage_path(tmp_path, spec=spec, impl=impl, count=count, storage=storage)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='storage-dump', spec=spec, case=case)
    attach_storage_metadata(benchmark, storage=storage)

    benchmark_pedantic(benchmark, _storage_dump, path, case.blobs, storage=storage)

    assert path.exists()


@COUNT_PARAM
@SPEC_PARAM
@STORAGE_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_05_dump_e2e(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    count: int,
    spec: CaseSpec,
    impl: Impl,
    storage: Storage,
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    skip_unsupported_storage(storage=storage, impl=impl)
    case = make_case(spec, count=count, impl=impl)
    path = _storage_path(tmp_path, spec=spec, impl=impl, count=count, storage=storage)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='dump-e2e', spec=spec, case=case)
    attach_storage_metadata(benchmark, storage=storage)

    def dump_e2e() -> None:
        _storage_dump(path, case.dump_blobs_all(), storage=storage)

    benchmark_pedantic(benchmark, dump_e2e)

    assert _sample(_storage_load(path, storage=storage)) == _sample(case.blobs)


@COUNT_PARAM
@SPEC_PARAM
@STORAGE_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_06_storage_load(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    count: int,
    spec: CaseSpec,
    impl: Impl,
    storage: Storage,
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    skip_unsupported_storage(storage=storage, impl=impl)
    case = make_case(spec, count=count, impl=impl)
    path = _storage_path(tmp_path, spec=spec, impl=impl, count=count, storage=storage)
    _storage_dump(path, case.blobs, storage=storage)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='storage-load', spec=spec, case=case)
    attach_storage_metadata(benchmark, storage=storage)

    result = benchmark_pedantic(benchmark, _storage_load, path, storage=storage)

    assert _sample(result) == _sample(case.blobs)


@COUNT_PARAM
@SPEC_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_07_blob_load(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='blob-load', spec=spec, case=case)

    result = benchmark_pedantic(benchmark, case.load_payloads_all)

    assert [case.payload_to_blob(payload) for payload in _sample(result)] == _sample(case.blobs)


@COUNT_PARAM
@SPEC_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_08_decode(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='decode', spec=spec, case=case)

    result = benchmark_pedantic(benchmark, case.deserialize_all)

    spec.validate_deserialized(impl, result, case.objects)


@COUNT_PARAM
@SPEC_PARAM
@STORAGE_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_09_load_e2e(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    count: int,
    spec: CaseSpec,
    impl: Impl,
    storage: Storage,
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    skip_unsupported_storage(storage=storage, impl=impl)
    case = make_case(spec, count=count, impl=impl)
    path = _storage_path(tmp_path, spec=spec, impl=impl, count=count, storage=storage)
    _storage_dump(path, case.blobs, storage=storage)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='load-e2e', spec=spec, case=case)
    attach_storage_metadata(benchmark, storage=storage)

    def load_e2e() -> list[object]:
        blobs = _storage_load(path, storage=storage)
        return [case.decode(case.blob_to_payload(blob)) for blob in blobs]

    result = benchmark_pedantic(benchmark, load_e2e)

    spec.validate_deserialized(impl, result, case.objects)


# TODO add a separate benchmark module for the real SqliteBackend path.
# TODO add a true cachew end-to-end cache miss/cache hit benchmark.

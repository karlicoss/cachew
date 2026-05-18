import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

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


def _sqlite_dump(db: Path, blobs: list[bytes]) -> None:
    db.unlink(missing_ok=True)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute('CREATE TABLE data (value BLOB)')
        conn.executemany('INSERT INTO data (value) VALUES (?)', [(blob,) for blob in blobs])


def _sqlite_load(db: Path) -> list[bytes]:
    with closing(sqlite3.connect(db)) as conn, conn:
        return [value for (value,) in conn.execute('SELECT value FROM data')]


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
@IMPL_PARAM
@DISABLE_GC
def test_04_sqlite_dump(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, tmp_path: Path, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    db = _sqlite_db_path(tmp_path, spec=spec, impl=impl, count=count)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='sqlite-dump', spec=spec, case=case)

    benchmark_pedantic(benchmark, _sqlite_dump, db, case.blobs)

    assert db.exists()


@COUNT_PARAM
@SPEC_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_05_dump_e2e(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, tmp_path: Path, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    db = _sqlite_db_path(tmp_path, spec=spec, impl=impl, count=count)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='dump-e2e', spec=spec, case=case)

    def dump_e2e() -> None:
        _sqlite_dump(db, case.dump_blobs_all())

    benchmark_pedantic(benchmark, dump_e2e)

    assert _sample(_sqlite_load(db)) == _sample(case.blobs)


@COUNT_PARAM
@SPEC_PARAM
@IMPL_PARAM
@DISABLE_GC
def test_06_sqlite_load(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, tmp_path: Path, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    db = _sqlite_db_path(tmp_path, spec=spec, impl=impl, count=count)
    _sqlite_dump(db, case.blobs)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='sqlite-load', spec=spec, case=case)

    result = benchmark_pedantic(benchmark, _sqlite_load, db)

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
@IMPL_PARAM
@DISABLE_GC
def test_09_load_e2e(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, tmp_path: Path, count: int, spec: CaseSpec, impl: Impl
) -> None:
    benchmark.group = _group_name(request, spec)
    skip_unsupported_impl(spec, impl)
    case = make_case(spec, count=count, impl=impl)
    db = _sqlite_db_path(tmp_path, spec=spec, impl=impl, count=count)
    _sqlite_dump(db, case.blobs)
    attach_case_metadata(benchmark, count=count, impl=impl, operation='load-e2e', spec=spec, case=case)

    def load_e2e() -> list[object]:
        blobs = _sqlite_load(db)
        return [case.decode(case.blob_to_payload(blob)) for blob in blobs]

    result = benchmark_pedantic(benchmark, load_e2e)

    spec.validate_deserialized(impl, result, case.objects)


# TODO add a separate benchmark module for the real SqliteBackend path.
# TODO add a true cachew end-to-end cache miss/cache hit benchmark.

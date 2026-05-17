from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, assert_never

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ...marshall.cachew import CachewMarshall
from ...marshall.common import Json

# OK, this doesn't work since function level @pytest.mark.benchmark overrides it
# pytestmark = pytest.mark.benchmark(disable_gc=True)
# TODO use functools_partial or something?

BENCHMARK_COUNT = 100_000
BENCHMARK_ROUNDS = 50

Impl = Literal['cachew', 'cattrs', 'legacy']
Impls: list[Impl] = ['cachew', 'cattrs', 'legacy']


@dataclass
class UUU:
    xx: int
    yy: int


@dataclass
class TE2:
    value: int
    uuu: UUU
    value2: int


@dataclass
class MarshallCase:
    objects: list[TE2]
    jsons: list[Json]
    to_json: Callable[[Any], Json]
    from_json: Callable[[Json], Any]

    def serialize_all(self) -> list[Json]:
        return [self.to_json(obj) for obj in self.objects]

    def deserialize_all(self) -> list[Any]:
        return [self.from_json(json_) for json_ in self.jsons]


def _sample(values: list[Any], *, sample_size: int = 100) -> list[Any]:
    # just to quickly sanity check/validate results
    return values[:sample_size] + values[-sample_size:]


def make_marshaller_impl(Type, *, impl: Impl) -> tuple[Callable[[Any], Json], Callable[[Json], Any]]:
    if impl == 'cachew':
        marshall: CachewMarshall[Any] = CachewMarshall(Type_=Type)
        return marshall.dump, marshall.load
    elif impl == 'cattrs':
        from cattrs import Converter

        converter = Converter()

        # NOTE: using dispatched functions directly seems a bit faster than
        # going through Converter.un/structure each time.
        # fmt: off
        unstruct_func = converter._unstructure_func.dispatch(Type)  # type: ignore[call-arg, misc]  # ty: ignore[missing-argument]
        struct_func   = converter._structure_func  .dispatch(Type)  # type: ignore[call-arg, misc]  # ty: ignore[missing-argument]
        # fmt: on

        return unstruct_func, lambda x: struct_func(x, Type)
    elif impl == 'legacy':
        from ...legacy import NTBinder

        binder = NTBinder.make(Type)
        return binder.to_row, binder.from_row  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
    else:
        assert_never(impl)


def make_nested_dataclass_case(*, count: int, impl: Impl) -> MarshallCase:
    to_json, from_json = make_marshaller_impl(TE2, impl=impl)
    objects = [TE2(value=i, uuu=UUU(xx=i, yy=i), value2=i) for i in range(count)]
    jsons = [to_json(obj) for obj in objects]
    return MarshallCase(objects=objects, jsons=jsons, to_json=to_json, from_json=from_json)


@pytest.mark.parametrize('count', [BENCHMARK_COUNT], ids=['100k'])
@pytest.mark.parametrize('impl', Impls)
@pytest.mark.benchmark(disable_gc=True, group='marshall-serialize')
def test_marshall_nested_dataclass_serialize(benchmark: BenchmarkFixture, count: int, impl: Impl) -> None:
    case = make_nested_dataclass_case(count=count, impl=impl)
    benchmark.extra_info['count'] = count
    benchmark.extra_info['impl'] = impl
    benchmark.extra_info['operation'] = 'serialize'

    # These are large batch benchmarks, so keep the run shape explicit and skip
    # pytest-benchmark's calibration logic for now.
    # TODO hmm, seems like default benchmark might have a lot of noise due to calibration, perhaps because it's not quite a microbenchmark..
    # result = benchmark(case.serialize_all)
    result = benchmark.pedantic(case.serialize_all, rounds=BENCHMARK_ROUNDS, warmup_rounds=2, iterations=1)

    assert _sample(result) == _sample(case.jsons)


@pytest.mark.parametrize('count', [BENCHMARK_COUNT], ids=['100k'])
@pytest.mark.parametrize('impl', Impls)
@pytest.mark.benchmark(disable_gc=True, group='marshall-deserialize')
def test_marshall_nested_dataclass_deserialize(benchmark: BenchmarkFixture, count: int, impl: Impl) -> None:
    case = make_nested_dataclass_case(count=count, impl=impl)
    benchmark.extra_info['count'] = count
    benchmark.extra_info['impl'] = impl
    benchmark.extra_info['operation'] = 'deserialize'

    # These are large batch benchmarks, so keep the run shape explicit and skip
    # pytest-benchmark's calibration logic for now.
    # result = benchmark(case.deserialize_all)
    result = benchmark.pedantic(case.deserialize_all, rounds=BENCHMARK_ROUNDS, warmup_rounds=2, iterations=1)

    assert _sample(result) == _sample(case.objects)

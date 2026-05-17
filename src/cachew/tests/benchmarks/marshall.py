from dataclasses import dataclass
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ...marshall.cachew import CachewMarshall
from ...marshall.common import Json

# OK, this doesn't work since function level @pytest.mark.benchmark overrides it
# pytestmark = pytest.mark.benchmark(disable_gc=True)
# TODO use functools_partial or something?

BENCHMARK_COUNT = 100_000
BENCHMARK_ROUNDS = 50


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
    marshall: CachewMarshall[Any]
    objects: list[TE2]
    jsons: list[Json]

    def serialize_all(self) -> list[Json]:
        return [self.marshall.dump(obj) for obj in self.objects]

    def deserialize_all(self) -> list[Any]:
        return [self.marshall.load(json_) for json_ in self.jsons]


def _sample(values: list[Any], *, sample_size: int = 100) -> list[Any]:
    # just to quickly sanity check/validate results
    return values[:sample_size] + values[-sample_size:]


def make_nested_dataclass_case(*, count: int) -> MarshallCase:
    marshall: CachewMarshall[Any] = CachewMarshall(Type_=TE2)
    objects = [TE2(value=i, uuu=UUU(xx=i, yy=i), value2=i) for i in range(count)]
    jsons = [marshall.dump(obj) for obj in objects]
    return MarshallCase(marshall=marshall, objects=objects, jsons=jsons)


@pytest.mark.parametrize('count', [BENCHMARK_COUNT], ids=['100k'])
@pytest.mark.benchmark(disable_gc=True, group='marshall-serialize')
def test_marshall_nested_dataclass_serialize(benchmark: BenchmarkFixture, count: int) -> None:
    case = make_nested_dataclass_case(count=count)
    benchmark.extra_info['count'] = count
    benchmark.extra_info['operation'] = 'serialize'

    # These are large batch benchmarks, so keep the run shape explicit and skip
    # pytest-benchmark's calibration logic for now.
    # TODO hmm, seems like default benchmark might have a lot of noise due to calibration, perhaps because it's not quite a microbenchmark..
    # result = benchmark(case.serialize_all)
    result = benchmark.pedantic(case.serialize_all, rounds=BENCHMARK_ROUNDS, warmup_rounds=2, iterations=1)

    assert _sample(result) == _sample(case.jsons)


@pytest.mark.parametrize('count', [BENCHMARK_COUNT], ids=['100k'])
@pytest.mark.benchmark(disable_gc=True, group='marshall-deserialize')
def test_marshall_nested_dataclass_deserialize(benchmark: BenchmarkFixture, count: int) -> None:
    case = make_nested_dataclass_case(count=count)
    benchmark.extra_info['count'] = count
    benchmark.extra_info['operation'] = 'deserialize'

    # These are large batch benchmarks, so keep the run shape explicit and skip
    # pytest-benchmark's calibration logic for now.
    # result = benchmark(case.deserialize_all)
    result = benchmark.pedantic(case.deserialize_all, rounds=BENCHMARK_ROUNDS, warmup_rounds=2, iterations=1)

    assert _sample(result) == _sample(case.objects)

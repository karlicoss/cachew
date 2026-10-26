from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

import pytest
import pytest_benchmark.table as benchmark_table

from ... import settings


@pytest.fixture(autouse=True)
def strict_cache_errors() -> Iterator[None]:
    # for benchmark we really don't want to fall back onto the original function
    original = settings.THROW_ON_ERROR
    settings.THROW_ON_ERROR = True
    yield
    settings.THROW_ON_ERROR = original


@pytest.hookimpl
def pytest_benchmark_update_machine_info(machine_info: dict[str, Any]) -> None:
    # a bit annoying, hostname can be volatile, e.g. when running under docker
    # (and this also keeps it out of committed benchmark results)
    machine_info["node"] = "redacted"


type Bench = dict[str, Any]
type BenchGroup = list[Bench]
type GroupStats = list[tuple[str | None, BenchGroup]]


def _constant_group_value(benchmarks: BenchGroup, prop: str) -> Any | None:
    values = {bench[prop] for bench in benchmarks}
    if len(values) != 1:
        return None
    return next(iter(values))


@pytest.hookimpl(wrapper=True)
def pytest_benchmark_group_stats(config: Any) -> Generator[None, GroupStats, GroupStats]:
    # NOTE: benchmarks here are copies only used for the results table (json results are already saved), so safe to modify
    groups = yield
    shown_columns = set(config.option.benchmark_columns or [])
    items: GroupStats = []
    for group, grouped_benchmarks in groups:
        # pytest-benchmark doesn't support custom columns, so show serialized size next to the name instead
        width = max(len(bench['name']) for bench in grouped_benchmarks)
        for bench in grouped_benchmarks:
            avg_bytes = bench['extra_info'].get('serialized_avg_bytes')
            if avg_bytes is not None:
                bench['name'] = f'{bench["name"]:<{width}} {avg_bytes:>6.1f} B/obj'

        summary: list[str] = []
        for prop, label in (('rounds', 'Rounds'), ('iterations', 'Iterations')):
            if prop in shown_columns:
                continue
            value = _constant_group_value(grouped_benchmarks, prop)
            if value is not None:
                summary.append(f'{label}: {value}')

        if summary:
            group = ' | '.join(summary if group is None else [group, *summary])
        items.append((group, grouped_benchmarks))
    return items


def pytest_configure() -> None:
    # default is 4 decimal places, way too noisy for these timings
    benchmark_table.NUMBER_FMT = '{0:,.1f}'  # ty: ignore[invalid-assignment]
    benchmark_table.ALIGNED_NUMBER_FMT = '{0:>{1},.1f}{2:<{3}}'  # ty: ignore[invalid-assignment]

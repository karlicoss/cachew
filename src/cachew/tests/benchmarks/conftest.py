from __future__ import annotations

from typing import Any

import pytest_benchmark.table as benchmark_table
from pytest_benchmark.table import TableResults
from pytest_benchmark.utils import DEFAULT_COLUMNS as _UPSTREAM_DEFAULT_COLUMNS


def pytest_benchmark_update_machine_info(config, machine_info: dict[str, Any]) -> None:  # noqa: ARG001
    # a bit annoying, hostname can be volatile, e.g. when running under docker
    machine_info["node"] = "redacted"


type Bench = dict[str, Any]
type BenchGroup = list[Bench]
type GroupStats = list[tuple[str | None, BenchGroup]]


_EXTRA_COLUMNS = {
    'serialized_avg_bytes'  : 'Avg Bytes',
    'serialized_total_bytes': 'Total Bytes',
}  # fmt: skip
_NUMBER_FMT = '{0:,.1f}'
_ALIGNED_NUMBER_FMT = '{0:>{1},.1f}{2:<{3}}'
_DEFAULT_COLUMNS = [column for column in _UPSTREAM_DEFAULT_COLUMNS if column not in {'ops', 'rounds', 'iterations'}]
_ORIGINAL_STAT_PROPS = benchmark_table.STAT_PROPS
_ORIGINAL_COMPUTE_SCALE = TableResults.compute_scale


def _promote_extra_columns(bench: Bench) -> Bench:
    promoted = dict(bench)
    extra_info = promoted.get('extra_info') or {}
    for column in _EXTRA_COLUMNS:
        if column in extra_info:
            promoted[column] = extra_info[column]
    return promoted


def _constant_group_value(benchmarks: BenchGroup, prop: str) -> Any | None:
    values = {bench[prop] for bench in benchmarks}
    if len(values) != 1:
        return None
    return next(iter(values))


def _compute_scale(
    self: TableResults,
    benchmarks: BenchGroup,
    best: dict[str, Any],
    worst: dict[str, Any],
) -> tuple[str, float, float, dict[str, str]]:
    unit, adjustment, ops_adjustment, labels = _ORIGINAL_COMPUTE_SCALE(self, benchmarks, best, worst)
    return unit, adjustment, ops_adjustment, {**labels, **_EXTRA_COLUMNS}


def pytest_benchmark_group_stats(config: Any, benchmarks: BenchGroup, group_by: str) -> GroupStats | None:
    grouped: dict[Any, BenchGroup] = {}
    shown_columns = set(config.option.benchmark_columns or [])

    if group_by == 'group':
        for bench in benchmarks:
            grouped.setdefault(bench['group'], []).append(_promote_extra_columns(bench))
        items: GroupStats = []
        for group, grouped_benchmarks in sorted(grouped.items(), key=lambda pair: pair[0] or ''):
            summary: list[str] = []
            for prop, label in (('rounds', 'Rounds'), ('iterations', 'Iterations')):
                if prop in shown_columns:
                    continue
                value = _constant_group_value(grouped_benchmarks, prop)
                if value is not None:
                    summary.append(f'{label}: {value}')

            rendered_group: str | None = group
            if summary:
                rendered_group = ' | '.join(part for part in (group, *summary) if part)
            items.append((rendered_group, grouped_benchmarks))

        return items

    return None


def pytest_configure(config) -> None:
    configured_columns = config.getoption('benchmark_columns')
    if configured_columns is None:
        columns = list(_DEFAULT_COLUMNS)
    else:
        columns = list(configured_columns)
    for column in _EXTRA_COLUMNS:
        if column not in columns:
            columns.append(column)
    config.option.benchmark_columns = columns
    bs = getattr(config, '_benchmarksession', None)
    if bs is not None:
        bs.columns = columns

    benchmark_table.NUMBER_FMT = _NUMBER_FMT  # ty: ignore[invalid-assignment]
    benchmark_table.ALIGNED_NUMBER_FMT = _ALIGNED_NUMBER_FMT  # ty: ignore[invalid-assignment]
    benchmark_table.STAT_PROPS = tuple(dict.fromkeys((*_ORIGINAL_STAT_PROPS, *_EXTRA_COLUMNS)))  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    TableResults.compute_scale = _compute_scale  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal, assert_never, cast, get_args
from zoneinfo import ZoneInfo

import orjson
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ...marshall.cachew import CachewMarshall, SDatetime
from ..utils import running_on_ci

# CI runs the benchmark suite as a smoke test, so keep the dataset small there
# while preserving the larger default for manual/local benchmark runs.
BENCHMARK_COUNT = 100 if running_on_ci else 100_000
# Default benchmark calibration introduced a fair bit of noise for these heavy
# batch workloads, so the benchmark modules consistently use pedantic mode.
BENCHMARK_PEDANTIC = {
    'rounds': 50,
    'warmup_rounds': 5,
    'iterations': 1,
}

type Impl = Literal[
    'cachew',
    'cattrs',
    'legacy',
    'pickle',
    'msgspec-json',
    'msgspec-msgpack',
]  # fmt: skip
Impls = cast(Sequence[Impl], get_args(Impl.__value__))
type Marshalled = Any
type ValidateDeserialized = Callable[[Impl, list[Any], list[Any]], None]


@dataclass(frozen=True)
class CaseSpec:
    id: str
    Type: Any
    build_objects: Callable[[int], list[Any]]
    validate_deserialized: ValidateDeserialized
    unsupported_impls: frozenset[Impl] = frozenset()
    unsupported_reason: str | None = None


@dataclass
class MarshallCase:
    objects: list[Any]
    payloads: list[Marshalled]
    blobs: list[bytes]
    encode: Callable[[Any], Marshalled]
    decode: Callable[[Marshalled], Any]
    payload_to_blob: Callable[[Marshalled], bytes]
    blob_to_payload: Callable[[bytes], Marshalled]

    def serialize_all(self) -> list[Marshalled]:
        return [self.encode(obj) for obj in self.objects]

    def deserialize_all(self) -> list[Any]:
        return [self.decode(payload) for payload in self.payloads]

    def dump_blobs_all(self) -> list[bytes]:
        return [self.payload_to_blob(payload) for payload in self.payloads]

    def load_payloads_all(self) -> list[Marshalled]:
        return [self.blob_to_payload(blob) for blob in self.blobs]

    def serialized_size_bytes(self) -> int:
        return sum(len(blob) for blob in self.blobs)


def _sample(values: list[Any], *, sample_size: int = 100) -> list[Any]:
    # just to quickly sanity check/validate results
    return values[:sample_size] + values[-sample_size:]


def _size_info_bytes(*, operation: str, case: MarshallCase) -> int:
    if operation in {'blob-dump', 'blob-load', 'storage-dump', 'storage-load', 'dump-e2e', 'load-e2e'}:
        return case.serialized_size_bytes()
    return 0


def _attach_size_info(benchmark: BenchmarkFixture, *, total_bytes: int, count: int) -> None:
    benchmark.extra_info['serialized_total_bytes'] = total_bytes
    benchmark.extra_info['serialized_avg_bytes'] = 0.0 if count == 0 else total_bytes / count


def attach_case_metadata(
    benchmark: BenchmarkFixture,
    *,
    count: int,
    impl: str,
    operation: str,
    spec: CaseSpec,
    case: MarshallCase,
) -> None:
    benchmark.extra_info['count'] = count
    benchmark.extra_info['shape'] = spec.id
    benchmark.extra_info['impl'] = impl
    benchmark.extra_info['operation'] = operation
    _attach_size_info(benchmark, total_bytes=_size_info_bytes(operation=operation, case=case), count=len(case.blobs))


def benchmark_pedantic[**P, R](
    benchmark: BenchmarkFixture,
    func: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    return benchmark.pedantic(func, args=args, kwargs=kwargs, **BENCHMARK_PEDANTIC)


_SDATETIME = SDatetime(type=datetime)


def _identity_bytes(blob: bytes) -> bytes:
    return blob


def _is_msgspec_impl(impl: Impl) -> bool:
    return impl in {'msgspec-json', 'msgspec-msgpack'}


def make_marshaller_impl(
    Type,
    *,
    impl: Impl,
) -> tuple[
    Callable[[Any], Marshalled],
    Callable[[Marshalled], Any],
    Callable[[Marshalled], bytes],
    Callable[[bytes], Marshalled],
]:
    if impl == 'cachew':
        marshall: CachewMarshall[Any] = CachewMarshall(Type_=Type)
        return marshall.dump, marshall.load, orjson.dumps, orjson.loads
    elif impl == 'cattrs':
        from cattrs import Converter

        converter = Converter()
        converter.register_unstructure_hook(datetime, _SDATETIME.dump)
        converter.register_structure_hook(datetime, lambda dct, _type: _SDATETIME.load(dct))

        # NOTE: using dispatched functions directly seems a bit faster than
        # going through Converter.un/structure each time.
        # fmt: off
        unstruct_func = converter._unstructure_func.dispatch(Type)  # type: ignore[call-arg, misc]  # ty: ignore[missing-argument]
        struct_func   = converter._structure_func  .dispatch(Type)  # type: ignore[call-arg, misc]  # ty: ignore[missing-argument]
        # fmt: on

        # def union_structure_hook_factory(_):
        #     def union_hook(data, type_):
        #         args = get_args(type_)

        #         if data is None:  # we don't try to coerce None into anything
        #             return None

        #         for t in args:
        #             try:
        #                 res = converter.structure(data, t)
        #             except Exception:
        #                 continue
        #             else:
        #                 return res
        #         raise ValueError(f"Could not cast {data} to {type_}")

        #     return union_hook

        # borrowed from https://github.com/python-attrs/cattrs/issues/423
        # uhh, this doesn't really work straightaway...
        # likely need to combine what cattr does with configure_tagged_union
        # converter.register_structure_hook_factory(is_union, union_structure_hook_factory)
        # configure_tagged_union(
        #     union=Type,
        #     converter=converter,
        # )

        return unstruct_func, lambda x: struct_func(x, Type), orjson.dumps, orjson.loads
    elif impl == 'legacy':
        from ...legacy import NTBinder

        # NOTE: legacy binder emits a tuple which can be inserted directly into
        # the database. So blob/sqlite stages are not directly comparable to the
        # JSON-like implementations, where you first marshal and then encode.
        # That also gives legacy a bit of an advantage for custom types, since
        # those would otherwise normally be handled by SQLAlchemy instead.
        binder = NTBinder.make(Type)
        return binder.to_row, binder.from_row, orjson.dumps, orjson.loads
    elif impl == 'pickle':
        # Keep the protocol explicit so cross-version benchmark results are
        # comparable even if pickle defaults change in the future.
        return partial(pickle.dumps, protocol=5), pickle.loads, _identity_bytes, _identity_bytes
    elif impl == 'msgspec-json':
        import msgspec

        json_encoder = msgspec.json.Encoder()
        json_decoder = msgspec.json.Decoder(type=Type)
        return json_encoder.encode, json_decoder.decode, _identity_bytes, _identity_bytes
    elif impl == 'msgspec-msgpack':
        import msgspec

        msgpack_encoder = msgspec.msgpack.Encoder()
        msgpack_decoder = msgspec.msgpack.Decoder(type=Type)
        return msgpack_encoder.encode, msgpack_decoder.decode, _identity_bytes, _identity_bytes
    else:
        assert_never(impl)


def _validate_sample_equal(_impl: Impl, result: list[Any], expected: list[Any]) -> None:
    assert _sample(result) == _sample(expected)


def _validate_datetimes(impl: Impl, result: list[Any], expected: list[Any]) -> None:
    if impl == 'legacy':
        # Legacy relies on SQLAlchemy's datetime adapter in the real sqlite path.
        # In these raw blob/json pipeline stages, orjson turns top-level datetime
        # payloads into strings, and NTBinder.from_row() does not convert them
        # back. Keep legacy in the benchmark for throughput numbers, but skip the
        # stronger round-trip assertion for this benchmark-only transport path.
        return
    assert _sample(result) == _sample(expected)
    if not _is_msgspec_impl(impl):
        # msgspec reconstructs fixed-offset tzinfo from RFC3339 rather than the
        # original named timezone object, so this stronger check doesn't apply.
        for r, e in zip(_sample(result), _sample(expected), strict=True):
            assert r.tzinfo == e.tzinfo


def skip_unsupported_impl(spec: CaseSpec, impl: Impl) -> None:
    if impl in spec.unsupported_impls:
        reason = spec.unsupported_reason or f'{impl} is unsupported for {spec.id}'
        pytest.skip(reason)


def make_case(spec: CaseSpec, *, count: int, impl: Impl) -> MarshallCase:
    objects = spec.build_objects(count)
    encode, decode, payload_to_blob, blob_to_payload = make_marshaller_impl(spec.Type, impl=impl)
    payloads = [encode(obj) for obj in objects]
    blobs = [payload_to_blob(payload) for payload in payloads]
    return MarshallCase(
        objects=objects,
        payloads=payloads,
        blobs=blobs,
        encode=encode,
        decode=decode,
        payload_to_blob=payload_to_blob,
        blob_to_payload=blob_to_payload,
    )


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
class Name:
    first: str
    last: str


@dataclass
class NameAlt:
    full_name: str
    label: str


def make_nested_dataclass_objects(count: int) -> list[TE2]:
    return [TE2(value=i, uuu=UUU(xx=i, yy=i), value2=i) for i in range(count)]


def make_datetime_objects(count: int) -> list[datetime]:
    import pytz

    tzs = [
        UTC,
        pytz.timezone('Europe/Berlin'),
        pytz.timezone('America/New_York'),
        ZoneInfo('America/Los_Angeles'),
        ZoneInfo('Asia/Shanghai'),
    ]
    # Using UTC datetimes to iterate; avoids pytz's replace/localize bs.
    start = datetime.fromisoformat('1990-01-01T12:00:00+00:00')
    end = datetime.fromisoformat('2030-01-01T12:00:00+00:00')
    step = (end - start) / count
    objects = []
    for i in range(count):
        dt = start + step * i
        tz = tzs[i % len(tzs)]
        objects.append(dt.astimezone(tz))
    return objects


def make_union_dataclass_objects(count: int) -> list[Name | NameAlt]:
    # Important that we test union of two dataclasses here. cattrs does not
    # really support unions with primitives in the same way.
    objects: list[Name | NameAlt] = []
    for i in range(count):
        if i % 2 == 0:
            objects.append(Name(first=f'first {i}', last=f'last {i}'))
        else:
            objects.append(NameAlt(full_name=f'full name {i}', label=f'label {i}'))
    return objects


def make_primitive_int_objects(count: int) -> list[int]:
    return list(range(count))


PRIMITIVE_INT_SPEC = CaseSpec(
    id='primitive-int',
    Type=int,
    build_objects=make_primitive_int_objects,
    validate_deserialized=_validate_sample_equal,
)
NESTED_DATACLASS_SPEC = CaseSpec(
    id='nested-dataclass',
    Type=TE2,
    build_objects=make_nested_dataclass_objects,
    validate_deserialized=_validate_sample_equal,
)
DATETIME_SPEC = CaseSpec(
    id='datetimes',
    Type=datetime,
    build_objects=make_datetime_objects,
    validate_deserialized=_validate_datetimes,
)
UNION_DATACLASS_SPEC = CaseSpec(
    id='union-dataclass',
    Type=Name | NameAlt,
    build_objects=make_union_dataclass_objects,
    validate_deserialized=_validate_sample_equal,
    unsupported_impls=frozenset({'msgspec-json', 'msgspec-msgpack'}),
    unsupported_reason='msgspec only supports multi-struct unions via tagged msgspec.Struct types',
)
CASE_SPECS = (
    PRIMITIVE_INT_SPEC,
    NESTED_DATACLASS_SPEC,
    DATETIME_SPEC,
    UNION_DATACLASS_SPEC,
)  # fmt: skip

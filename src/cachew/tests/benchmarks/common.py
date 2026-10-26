import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
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
    'pickle',
    'msgspec-json',
    'msgspec-msgpack',
]  # fmt: skip
Impls = cast(Sequence[Impl], get_args(Impl.__value__))
# benchmark stages, in pipeline order
type Operation = Literal[
    'build',
    'encode',
    'json-dump',
    'storage-dump',
    'dump-e2e',
    'storage-load',
    'json-load',
    'decode',
    'load-e2e',
]  # fmt: skip
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
    marshall: Callable[[Any], Marshalled]
    unmarshall: Callable[[Marshalled], Any]
    # None for impls that marshall to bytes straightaway, i.e. without a separate json step
    payload_to_blob: Callable[[Marshalled], bytes] | None
    blob_to_payload: Callable[[bytes], Marshalled] | None

    # NOTE: these are deliberately written out rather than sharing code: an extra function call
    # or generator step per object adds measurable overhead for the faster impls (e.g. msgspec).
    # iter_encode is the streaming version for the e2e write benchmark, which writes blobs as it
    # encodes (like cachew does) rather than materializing them all first.
    # fmt: off
    def encode_all(self, objects: Iterable[Any]) -> list[bytes]:
        marshall, payload_to_blob = self.marshall, self.payload_to_blob
        if payload_to_blob is None:
            return [                marshall(obj)  for obj in objects]
        return     [payload_to_blob(marshall(obj)) for obj in objects]

    def iter_encode(self, objects: Iterable[Any]) -> Iterator[bytes]:
        marshall, payload_to_blob = self.marshall, self.payload_to_blob
        if payload_to_blob is None:
            return (                marshall(obj)  for obj in objects)
        return     (payload_to_blob(marshall(obj)) for obj in objects)

    def decode_all(self, blobs: Iterable[bytes]) -> list[Any]:
        unmarshall, blob_to_payload = self.unmarshall, self.blob_to_payload
        if blob_to_payload is None:
            return [unmarshall(                blob)  for blob in blobs]
        return     [unmarshall(blob_to_payload(blob)) for blob in blobs]
    # fmt: on

    def json_dump_all(self) -> list[bytes]:
        payload_to_blob = self.payload_to_blob
        assert payload_to_blob is not None
        return [payload_to_blob(payload) for payload in self.payloads]

    def json_load_all(self) -> list[Marshalled]:
        blob_to_payload = self.blob_to_payload
        assert blob_to_payload is not None
        return [blob_to_payload(blob) for blob in self.blobs]

    def serialized_size_bytes(self) -> int:
        return sum(len(blob) for blob in self.blobs)


def _sample(values: list[Any], *, sample_size: int = 100) -> list[Any]:
    # just to quickly sanity check/validate results
    return values[:sample_size] + values[-sample_size:]


def _attach_size_info(benchmark: BenchmarkFixture, *, total_bytes: int, count: int) -> None:
    benchmark.extra_info['serialized_avg_bytes'] = total_bytes / count


def attach_case_metadata(
    benchmark: BenchmarkFixture,
    *,
    count: int,
    impl: str,
    operation: Operation,
    spec: CaseSpec,
    case: MarshallCase,
) -> None:
    benchmark.extra_info['count'] = count
    benchmark.extra_info['shape'] = spec.id
    benchmark.extra_info['impl'] = impl
    benchmark.extra_info['operation'] = operation
    _attach_size_info(benchmark, total_bytes=case.serialized_size_bytes(), count=len(case.blobs))


def benchmark_pedantic[**P, R](
    benchmark: BenchmarkFixture,
    func: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    return benchmark.pedantic(func, args=args, kwargs=kwargs, **BENCHMARK_PEDANTIC)


_SDATETIME = SDatetime(type=datetime)


def _is_msgspec_impl(impl: Impl) -> bool:
    return impl in {'msgspec-json', 'msgspec-msgpack'}


def make_marshaller_impl(
    Type,
    *,
    impl: Impl,
) -> tuple[
    Callable[[Any], Marshalled],
    Callable[[Marshalled], Any],
    Callable[[Marshalled], bytes] | None,
    Callable[[bytes], Marshalled] | None,
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
    elif impl == 'pickle':
        # Keep the protocol explicit so cross-version benchmark results are
        # comparable even if pickle defaults change in the future.
        return partial(pickle.dumps, protocol=5), pickle.loads, None, None
    elif impl == 'msgspec-json':
        import msgspec

        json_encoder = msgspec.json.Encoder()
        json_decoder = msgspec.json.Decoder(type=Type)
        return json_encoder.encode, json_decoder.decode, None, None
    elif impl == 'msgspec-msgpack':
        import msgspec

        msgpack_encoder = msgspec.msgpack.Encoder()
        msgpack_decoder = msgspec.msgpack.Decoder(type=Type)
        return msgpack_encoder.encode, msgpack_decoder.decode, None, None
    else:
        assert_never(impl)


def _validate_sample_equal(_impl: Impl, result: list[Any], expected: list[Any]) -> None:
    assert _sample(result) == _sample(expected)


def _validate_datetimes(impl: Impl, result: list[Any], expected: list[Any]) -> None:
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


def skip_no_json_step(impl: Impl) -> None:
    # other impls marshall to bytes straightaway (make_marshaller_impl returns None for payload_to_blob)
    if impl not in {'cachew', 'cattrs'}:
        pytest.skip(f"{impl} doesn't have a separate json dump/load step")


def make_case(spec: CaseSpec, *, count: int, impl: Impl) -> MarshallCase:
    objects = spec.build_objects(count)
    marshall, unmarshall, payload_to_blob, blob_to_payload = make_marshaller_impl(spec.Type, impl=impl)
    payloads = [marshall(obj) for obj in objects]
    blobs = payloads if payload_to_blob is None else [payload_to_blob(payload) for payload in payloads]
    return MarshallCase(
        objects=objects,
        payloads=payloads,
        blobs=blobs,
        marshall=marshall,
        unmarshall=unmarshall,
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

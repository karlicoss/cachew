#!/usr/bin/env python3
from dataclasses import dataclass
from typing import NamedTuple, Union


def test_dataclasses_json():
    # pip install dataclasses-json
    from dataclasses_json import dataclass_json

    @dataclass
    class Inner:
        value: int

    @dataclass
    class Outer:
        inner: Inner

    ### issue 1: requires @dataclass_json annotation on all involved dataclasses
    obj = Outer(inner=Inner(value=123))  # noqa: F841

    # we don't control the types that are passed to us, so we can't use the @dataclass_json
    # but we can just call the decorator directly

    # HOWEVER: this modifies the original class, Outer!!
    OuterJson = dataclass_json(Outer)  # noqa: F841
    # it adds 'from_dict', 'from_json', 'schema', 'to_dict', 'to_json' attributes to it

    # now if you try
    # print(OuterJson.schema().dump(obj))
    # you get a warning that it wants you to add annotations to Inner classes too.
    # this isn't really an option for us.
    ###

    ### issue 2: can't dump anything unless the top level type is a dataclass?
    ### could wrap into a dummy dataclass or something, but is wasteful in terms of performance
    ###

    ### nice thing: correctly serializes Union types, even if they share the same attributes
    @dataclass_json
    @dataclass
    class City:
        name: str

    @dataclass_json
    @dataclass
    class Country:
        name: str

    @dataclass_json
    @dataclass
    class WithUnion:
        union: Union[City, Country]  # noqa: UP007

    objs = [
        WithUnion(union=City(name='London')),
        WithUnion(union=Country(name='UK')),
    ]

    schema = WithUnion.schema()
    json = schema.dumps(objs, many=True)
    objs2 = schema.loads(json, many=True)
    print("objects  ", objs)
    print("json     ", json)
    # NOTE: it dumps [{"union": {"name": "London", "__type": "City"}}, {"union": {"name": "UK", "__type": "Country"}}]
    # so types are correctly distinguished
    print("restored ", objs2)
    assert objs == objs2, (objs, objs2)
    ###


def test_marshmallow_dataclass():
    # pip3 install --user marshmallow-dataclass[union]
    import marshmallow_dataclass

    ### issue 1: the top level type has to be a dataclass?
    ### although possible that we could use regular marshmallow for that instead
    ###

    ### issue 2: doesn't handle unions correctly
    @dataclass
    class City:
        name: str

    @dataclass
    class Country:
        name: str

    @dataclass
    class WithUnion:
        union: Union[City, Country]  # noqa: UP007

    objs = [
        WithUnion(union=City(name="London")),
        WithUnion(union=Country(name="UK")),
    ]

    # NOTE: good, doesn't require adding annotations on the original classes
    schema = marshmallow_dataclass.class_schema(WithUnion)()

    json = schema.dumps(objs, many=True)
    objs2 = schema.loads(json, many=True)
    print("objects  ", objs)
    print("json     ", json)
    # NOTE: it dumps [{"union": {"value": 123}}, {"union": {"value": 123}}]
    # so it doesn't distingush based on types => won't deserialize correctly
    print("restored ", objs2)
    # assert objs == objs2, (objs, objs2)
    # ^ this assert fails!
    ###


def test_pydantic():
    from pydantic import TypeAdapter

    ### issue: doesn't handle Unions correctly
    @dataclass
    class City:
        name: str

    @dataclass
    class Country:
        name: str

    @dataclass
    class WithUnion:
        union: Union[City, Country]  # noqa: UP007

    objs = [
        WithUnion(union=City(name="London")),
        WithUnion(union=Country(name="UK")),
    ]

    # NOTE: nice, doesn't require annotating the original classes with anything
    Schema = TypeAdapter(list[WithUnion])

    json = Schema.dump_python(
        objs,
        # round_rtip: Whether to output the serialized data in a way that is compatible with deserialization
        # not sure, doesn't seem to impact anything..
        round_trip=True,
    )
    objs2 = Schema.validate_python(json)

    print("objects  ", objs)
    print("json     ", json)
    print("restored ", objs2)

    # assert objs == objs2, (objs, objs2)
    # ^ this assert fails!
    # created an issue https://github.com/pydantic/pydantic/issues/7391
    ###


def test_cattrs():
    from cattrs import Converter
    from cattrs.strategies import configure_tagged_union

    converter = Converter()

    ### issue: NamedTuples aren't unstructured? asked here https://github.com/python-attrs/cattrs/issues/425
    class X(NamedTuple):
        value: int

    d = converter.unstructure(X(value=123), X)  # noqa: F841
    # NOTE: this assert doesn't pass!
    # assert isinstance(d, dict)
    ###

    ### good: handles Union correctly (although some extra configuring required)
    @dataclass
    class City:
        name: str

    @dataclass
    class Country:
        name: str

    @dataclass
    class WithUnion:
        union: Union[City, Country]  # noqa: UP007

    objs = [
        WithUnion(union=City(name="London")),
        WithUnion(union=Country(name="UK")),
    ]

    configure_tagged_union(
        union=City | Country,
        converter=converter,
    )
    # NOTE: nice -- doesn't require decorating original classes
    json = converter.unstructure(objs, list[WithUnion])
    assert isinstance(json, list)
    objs2 = converter.structure(json, list[WithUnion])

    print("objects  ", objs)
    # NOTE: dumps it as [{'union': {'name': 'London', '_type': 'City'}}, {'union': {'name': 'UK', '_type': 'Country'}}]
    print("json     ", json)
    print("restored ", objs2)

    assert objs == objs2, (objs, objs2)
    ###

    ### issue: unions of simple types aren't supported?
    # see https://github.com/python-attrs/cattrs/issues/423
    mixed: list[int | str] = [
        123,
        'Jakarta',
    ]
    json = converter.unstructure(mixed, list[int | str])
    # NOTE: this fails
    # mixed2 = converter.structure(json , list[int | str])
    ###


test_dataclasses_json()
test_marshmallow_dataclass()
test_pydantic()
test_cattrs()

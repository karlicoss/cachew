from ..utils import resolve_type_parameters


def test_simple_generic_alias() -> None:
    # if you define types ad-hoc, they resolve to GenericAlias, not TypeAliasType
    assert resolve_type_parameters(int) == int  # noqa: E721
    assert resolve_type_parameters(list[bool]) == list[bool]
    assert resolve_type_parameters(dict[str, list[float]]) == dict[str, list[float]]


def test_simple_type_keyword() -> None:
    type Int = int

    assert resolve_type_parameters(Int) == int  # noqa: E721
    assert resolve_type_parameters(list[Int]) == list[int]
    assert resolve_type_parameters(dict[str, list[Int]]) == dict[str, list[int]]


def test_generic_collections() -> None:
    type ListInt = list[int]
    assert resolve_type_parameters(ListInt) == list[int]
    assert resolve_type_parameters(dict[str, ListInt]) == dict[str, list[int]]

    type TupleInt = tuple[int, bool]
    assert resolve_type_parameters(TupleInt) == tuple[int, bool]
    type TupleIntStr = tuple[TupleInt, str]
    assert resolve_type_parameters(TupleIntStr) == tuple[tuple[int, bool], str]

    type SetStr = set[str]
    assert resolve_type_parameters(SetStr) == set[str]

    type DictAlias[K, V] = dict[K, V]
    assert resolve_type_parameters(DictAlias[str, int]) == dict[str, int]
    assert resolve_type_parameters(DictAlias[int, list[str]]) == dict[int, list[str]]

    type ComplexDict = dict[str, tuple[ListInt, SetStr]]
    assert resolve_type_parameters(ComplexDict) == dict[str, tuple[list[int], set[str]]]


def test_generic_type_keyword() -> None:
    type Id[T] = T
    type IdInt = Id[int]

    assert resolve_type_parameters(IdInt) == int  # noqa: E721
    assert resolve_type_parameters(list[IdInt]) == list[int]

    # check multiple uses of type params
    type Pair[T] = tuple[T, T]
    type PairInt = Pair[int]
    assert resolve_type_parameters(PairInt) == tuple[int, int]
    assert resolve_type_parameters(Pair[str]) == tuple[str, str]
    assert resolve_type_parameters(list[Pair[int]]) == list[tuple[int, int]]

    # check if type params aren't used
    type NotUsing1[T, V] = int
    type NotUsing2[V, W] = NotUsing1[bool, float]
    type ListInt1 = list[NotUsing2[list, str]]
    assert resolve_type_parameters(ListInt1) == list[int]

    # Test generic alias with alias as parameter
    type Container[T] = list[T]
    type Int = int
    assert resolve_type_parameters(Container[Int]) == list[int]


def test_chaining() -> None:
    type Int = int
    type Int2 = Int
    type Int3 = Int2
    assert resolve_type_parameters(Int3) == int  # noqa: E721

    type ListInt3 = list[Int3]
    assert resolve_type_parameters(ListInt3) == list[int]

    type Box[T] = list[T]
    type DoubleBox[T] = Box[Box[T]]
    type DoubleBoxFloat = DoubleBox[float]
    assert resolve_type_parameters(DoubleBoxFloat) == list[list[float]]


def test_optional_and_union() -> None:
    type Int = int
    type MaybeInt = int | None
    assert resolve_type_parameters(MaybeInt) == (int | None)
    assert resolve_type_parameters(list[MaybeInt]) == list[int | None]

    type Str = str  # FIXME extract outside?

    type StrOrInt = Str | Int
    assert resolve_type_parameters(StrOrInt) == (str | int)

    type UnionWithAlias = int | Str
    assert resolve_type_parameters(UnionWithAlias) == (int | str)

    # Test union in generic contexts
    type OptionalList[T] = list[T] | None
    assert resolve_type_parameters(OptionalList[int]) == (list[int] | None)
    assert resolve_type_parameters(OptionalList[str]) == (list[str] | None)

    # Test nested unions with aliases
    type Bool = bool
    type StrOrIntOrBool = StrOrInt | Bool
    assert resolve_type_parameters(StrOrIntOrBool) == (int | str | bool)

    # Test union with complex aliased types
    type ListInt = list[int]
    type DictStrInt = dict[str, int]
    type ComplexUnion = ListInt | DictStrInt | None
    assert resolve_type_parameters(ComplexUnion) == (list[int] | dict[str, int] | None)


def test_old_aliases() -> None:
    """
    Old style typing.* aliases get 'normalised' by typing.get_origin call.
    This shouldn't really be a problem, so just highihghting it here.
    """
    from typing import Dict, List, Optional  # noqa: UP035

    type OptionalInt = Optional[int]  # noqa: UP045
    assert resolve_type_parameters(OptionalInt) == int | None

    type ListInt = List[int]  # noqa: UP006
    assert resolve_type_parameters(ListInt) == list[int]

    type DictIntStr = Dict[int, str]  # noqa: UP006
    assert resolve_type_parameters(DictIntStr) == dict[int, str]


def test_old_union() -> None:
    from typing import Union

    type IntUnion[T] = Union[int, T, bool]  # noqa: UP007

    assert resolve_type_parameters(IntUnion[str]) == (int | str | bool)


def test_typevar() -> None:
    from typing import TypeVar

    X = TypeVar('X')

    ListX = list[X]
    type ListInt = ListX[int]
    assert resolve_type_parameters(ListInt) == list[int]

    SetX = set[X]
    SetFloat = SetX[float]
    assert resolve_type_parameters(SetFloat) == set[float]


def test_misc() -> None:
    """
    Miscellaneous more complex tests.
    """

    # Test union inside list/dict
    type MaybeStr = str | None
    assert resolve_type_parameters(list[MaybeStr]) == list[str | None]
    assert resolve_type_parameters(dict[str, MaybeStr]) == dict[str, str | None]

    # Test union with nested generic aliases
    type Container[T] = list[T]
    type OptionalContainer[T] = Container[T] | None
    assert resolve_type_parameters(OptionalContainer[int]) == (list[int] | None)

    # Test union with multiple aliased generics
    type ListAlias[T] = list[T]
    type SetAlias[T] = set[T]
    type CollectionUnion[T] = ListAlias[T] | SetAlias[T]
    assert resolve_type_parameters(CollectionUnion[str]) == (list[str] | set[str])

    # Test union in tuple
    type IntOrStr = int | str
    assert resolve_type_parameters(tuple[IntOrStr, bool]) == tuple[int | str, bool]

    # Test deeply nested union with aliases
    type Middle = list[IntOrStr]
    type Outer = Middle | None
    assert resolve_type_parameters(Outer) == (list[int | str] | None)

    # Test union with chained aliases
    type Level1 = int
    type Level2 = Level1
    type Level3 = Level2
    type UnionChained = Level3 | str | None
    assert resolve_type_parameters(UnionChained) == (int | str | None)

    # Test union with generic that resolves to union
    type MaybeList[T] = list[T] | None
    type NestedMaybe = MaybeList[int | str]
    assert resolve_type_parameters(NestedMaybe) == (list[int | str] | None)

    # Test union with aliased union
    type NumberOrStr = int | float | str
    type ExtendedUnion = NumberOrStr | bool
    assert resolve_type_parameters(ExtendedUnion) == (int | float | str | bool)

    # Test union in dict values and keys
    type FlexibleKey = str | int
    type FlexibleValue = list[int] | dict[str, str] | None
    assert (
        resolve_type_parameters(dict[FlexibleKey, FlexibleValue]) == dict[str | int, list[int] | dict[str, str] | None]
    )

    # Test union with same type repeated (Python may or may not normalize this)
    type RepeatUnion = int | int | str  # noqa: PYI016
    # Python's union implementation may deduplicate, so we accept both
    assert resolve_type_parameters(RepeatUnion) == (int | str) or resolve_type_parameters(RepeatUnion) == (int | int | str)  # fmt: skip

    # Test union with TypeAliasType in multiple positions
    type AliasA = list[int]
    type AliasB = dict[str, int]
    type AliasC = set[str]
    type MultiAliasUnion = AliasA | AliasB | AliasC
    assert resolve_type_parameters(MultiAliasUnion) == (list[int] | dict[str, int] | set[str])

    # Test generic union with substitution
    type Result[T, E] = T | E
    assert resolve_type_parameters(Result[int, str]) == (int | str)
    assert resolve_type_parameters(Result[list[int], dict[str, str]]) == (list[int] | dict[str, str])

    # Test union with None (Optional pattern) in various positions
    type OptionalInt = int | None
    type ListOfOptional = list[OptionalInt]
    assert resolve_type_parameters(ListOfOptional) == list[int | None]

    # Test union with multiple levels of aliased unions
    type UnionA = int | str
    type UnionB = bool | float
    type CombinedUnion = UnionA | UnionB
    assert resolve_type_parameters(CombinedUnion) == (int | str | bool | float)

    # Test union as generic parameter with nested aliases
    type NestedAlias = list[int]
    type UnionParam[T] = dict[str, T | None]
    assert resolve_type_parameters(UnionParam[NestedAlias]) == dict[str, list[int] | None]

    # Test complex scenario: generic alias that returns a union, used in another union
    type ComplexUnion[T] = MaybeList[T] | dict[str, T]
    assert resolve_type_parameters(ComplexUnion[int]) == (list[int] | None | dict[str, int])

    # Test union in tuple with multiple aliased elements
    type AliasInt = int
    type AliasStr = str
    type TupleWithUnions = tuple[AliasInt | None, list[AliasStr | bool]]
    assert resolve_type_parameters(TupleWithUnions) == tuple[int | None, list[str | bool]]

    # Test three-way union with all aliased types
    type TypeA = list[int]
    type TypeB = dict[str, str]
    type TypeC = set[bool]
    type ThreeWayUnion = TypeA | TypeB | TypeC
    assert resolve_type_parameters(ThreeWayUnion) == (list[int] | dict[str, str] | set[bool])

    # Test union where members themselves contain unions
    type InnerUnion1 = int | str
    type InnerUnion2 = bool | float
    type OuterUnion = list[InnerUnion1] | dict[str, InnerUnion2]
    assert resolve_type_parameters(OuterUnion) == (list[int | str] | dict[str, bool | float])

    # Test generic union with nested type aliases in parameters
    type Box[T] = list[T]
    type OptionBox[T] = Box[T] | None
    assert resolve_type_parameters(OptionBox[int | str]) == (list[int | str] | None)

    # Test union with mix of generic and non-generic aliases
    type SimpleAlias = int
    type GenericAlias[T] = list[T]
    type MixedUnion[T] = SimpleAlias | GenericAlias[T]
    assert resolve_type_parameters(MixedUnion[str]) == (int | list[str])

    # Test generic alias that returns the parameter unchanged
    type Same[T] = T
    assert resolve_type_parameters(Same[int]) == int  # noqa: E721
    assert resolve_type_parameters(Same[list[str]]) == list[str]
    assert resolve_type_parameters(Same[Same[int]]) == int  # noqa: E721

    # Test deeply nested generics
    type Deep = dict[str, list[tuple[int, set[str]]]]
    assert resolve_type_parameters(Deep) == dict[str, list[tuple[int, set[str]]]]

    # Test union in complex nested structure
    type Data[T] = dict[str, list[T] | None]
    assert resolve_type_parameters(Data[int | str]) == dict[str, list[int | str] | None]

    # Test alias in tuple with mixed types
    type Mixed = tuple[int, list[str], dict[str, int]]
    assert resolve_type_parameters(Mixed) == tuple[int, list[str], dict[str, int]]

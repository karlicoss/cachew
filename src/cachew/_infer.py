from collections.abc import Callable, Iterable
from typing import Any, Literal, get_args, get_origin, get_type_hints

from ._types import resolve_type_parameters
from .common import TypeNotSupported
from .marshall.cachew import build_schema

Failure = str  # deliberately not a type =, used in type checks
type Kind = Literal['single', 'multiple']
type Inferred = tuple[Kind, type[Any]]


def infer_return_type(func: Callable[..., Any]) -> Failure | Inferred:
    """
    >>> def const() -> int:
    ...     return 123
    >>> infer_return_type(const)
    ('single', <class 'int'>)

    >>> from typing import Optional
    >>> def first_character(s: str) -> Optional[str]:
    ...     return None if len(s) == 0 else s[0]
    >>> kind, opt = infer_return_type(first_character)
    >>> # in 3.8, Optional[str] is printed as Union[str, None], so need to hack around this
    >>> (kind, opt == Optional[str])
    ('single', True)

    # tuple is an iterable.. but presumably should be treated as a single value
    >>> from typing import Tuple
    >>> def a_tuple() -> Tuple[int, str]:
    ...     return (123, 'hi')
    >>> infer_return_type(a_tuple)
    ('single', tuple[int, str])

    >>> from typing import Collection, NamedTuple
    >>> class Person(NamedTuple):
    ...     name: str
    ...     age: int
    >>> def person_provider() -> Collection[Person]:
    ...     return []
    >>> infer_return_type(person_provider)
    ('multiple', <class 'cachew._infer.Person'>)

    >>> def single_str() -> str:
    ...     return 'hello'
    >>> infer_return_type(single_str)
    ('single', <class 'str'>)

    >>> def single_person() -> Person:
    ...     return Person(name="what", age=-1)
    >>> infer_return_type(single_person)
    ('single', <class 'cachew._infer.Person'>)

    >>> from typing import Sequence
    >>> def int_provider() -> Sequence[int]:
    ...     return (1, 2, 3)
    >>> infer_return_type(int_provider)
    ('multiple', <class 'int'>)

    >>> from typing import Iterator
    >>> def union_provider() -> Iterator[str | int]:
    ...     yield 1
    ...     yield 'aaa'
    >>> infer_return_type(union_provider)
    ('multiple', str | int)

    >>> from typing import Iterator
    >>> type Str = str
    >>> type Int = int
    >>> type IteratorStrInt = Iterator[Str | Int]
    >>> def iterator_str_int() -> IteratorStrInt:
    ...     yield 1
    ...     yield 'aaa'
    >>> infer_return_type(iterator_str_int)
    ('multiple', str | int)

    # a bit of an edge case
    >>> from typing import Tuple
    >>> def empty_tuple() -> Iterator[Tuple[()]]:
    ...     yield ()
    >>> infer_return_type(empty_tuple)
    ('multiple', tuple[()])

    ... # doctest: +ELLIPSIS

    >>> def untyped():
    ...     return 123
    >>> infer_return_type(untyped)
    'no return type annotation...'

    >>> from typing import List
    >>> class Custom:
    ...     pass
    >>> def unsupported() -> Custom:
    ...     return Custom()
    >>> infer_return_type(unsupported)
    "can't infer type from <class 'cachew._infer.Custom'>: can't cache <class 'cachew._infer.Custom'>"

    >>> def unsupported_list() -> List[Custom]:
    ...     return [Custom()]
    >>> infer_return_type(unsupported_list)
    "can't infer type from list[cachew._infer.Custom]: can't cache <class 'cachew._infer.Custom'>"
    """
    try:
        hints = get_type_hints(func)
    except Exception as ne:
        # get_type_hints might fail if types are forward defined or missing
        # see test_future_annotation for an example
        return str(ne)
    rtype = hints.get('return', None)
    if rtype is None:
        return f"no return type annotation on {func}"

    rtype = resolve_type_parameters(rtype)

    def bail(reason: str) -> str:
        return f"can't infer type from {rtype}: " + reason

    # first we wanna check if the top level type is some sort of iterable that makes sense ot cache
    # e.g. List/Sequence/Iterator etc
    return_multiple = _returns_multiple(rtype)

    if return_multiple:
        # then the actual type to cache will be the argument of the top level one
        args = get_args(rtype)
        if len(args) != 1:
            return bail(f"wrong number of __args__: {args}")

        (cached_type,) = args
    else:
        cached_type = rtype

    try:
        build_schema(Type=cached_type)
    except TypeNotSupported as ex:
        return bail(f"can't cache {ex.type_}")

    return ('multiple' if return_multiple else 'single', cached_type)


def _returns_multiple(rtype: object) -> bool:
    origin = get_origin(rtype)
    if origin is None:
        return False
    if origin is tuple:
        # usually tuples are more like single values rather than a sequence? (+ this works for namedtuple)
        return False
    try:
        return issubclass(origin, Iterable)
    except TypeError:
        # that would happen if origin is not a 'proper' type, e.g. is a Union or something
        # seems like exception is the easiest way to check
        return False

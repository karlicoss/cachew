from collections.abc import Mapping
from types import UnionType
from typing import TypeAliasType, TypeVar, get_args, get_origin


# https://stackoverflow.com/a/2166841/706389
def is_namedtuple(t) -> bool:
    b = getattr(t, '__bases__', None)
    if b is None:
        return False
    if len(b) != 1 or b[0] is not tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)  # noqa: E721


def resolve_type_parameters(t) -> type:
    return _resolve_type_parameters_aux(t, typevar_to_type={})


def _resolve_type_parameters_aux(t, *, typevar_to_type: Mapping[TypeVar, type]) -> type:
    if isinstance(t, TypeVar):
        return typevar_to_type[t]

    # This is the 'left hand side' case, i.e. in type ... =
    if isinstance(t, TypeAliasType):
        return _resolve_type_parameters_aux(t.__value__, typevar_to_type=typevar_to_type)

    # note: args is never none
    raw_args = get_args(t)
    resolved_args = tuple(_resolve_type_parameters_aux(arg, typevar_to_type=typevar_to_type) for arg in raw_args)

    # UnionType: resolve each member of the union
    if isinstance(t, UnionType):
        # Reconstruct the union with resolved args
        result = resolved_args[0]
        for arg in resolved_args[1:]:
            result = result | arg  # type: ignore[assignment]
        return result

    origin = get_origin(t)

    # Must be a non-generic type
    if origin is None:
        return t

    # This is the 'right hand side', e.g. '... = Id[int]' matches this
    if isinstance(origin, TypeAliasType):
        type_params = origin.__type_params__
        new_typevar_to_type: Mapping[TypeVar, type] = {
            **typevar_to_type,
            **dict(zip(type_params, resolved_args, strict=True)),  # type: ignore[arg-type]
        }
        return _resolve_type_parameters_aux(origin.__value__, typevar_to_type=new_typevar_to_type)

    # Just a regular generic type
    return origin[resolved_args]

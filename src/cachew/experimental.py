from typing import Optional

import sqlalchemy  # type: ignore

from . import PRIMITIVES


# NOTE ok, not ideal converting everything to str; but after all what else you can to with Exception??
class ExceptionAdapter(sqlalchemy.TypeDecorator):
    impl = sqlalchemy.String

    @property
    def python_type(self): return Exception

    def process_literal_param(self, value, dialect): raise NotImplementedError()  # make pylint happy

    def process_bind_param(self, value: Optional[Exception], dialect) -> Optional[str]:
        if value is None:
            return None
        return str(value.args)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[Exception]:
        if value is None:
            return None
        return Exception(value)


def enable_exceptions():
    """
    Enables support for caching Exceptions. Exception arguments are going to be serialized as strings.

    It's useful for defensive error handling, in case of cachew in particular for preserving error state.

    I elaborate on it here: [mypy-driven error handling](https://beepb00p.xyz/mypy-error-handling.html#kiss).
    """
    if Exception not in PRIMITIVES:
        PRIMITIVES[Exception] = ExceptionAdapter


def disable_exceptions():
    if Exception in PRIMITIVES:
        del PRIMITIVES[Exception]

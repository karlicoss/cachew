from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    from .compat import deprecated

    @deprecated("Exceptions are not an experimental feature anymore and enabled by default.")
    def enable_exceptions() -> None:
        pass

    @deprecated("Exceptions are not an experimental feature anymore and enabled by default.")
    def disable_exceptions() -> None:
        pass

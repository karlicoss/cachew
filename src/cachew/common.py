from dataclasses import dataclass

# TODO better name to represent what it means?
type SourceHash = str

DEPENDENCIES = 'dependencies'


class CachewException(RuntimeError):
    # TODO rename this to CachewError for consistency with concrete error subclasses.
    pass


class CacheReadError(CachewException):
    """
    Cache read failures are unrecoverable and do not respect settings.THROW_ON_ERROR.
    Once cached data starts yielding, falling back to the wrapped function can duplicate or mix results.
    """

    pass


@dataclass
class TypeNotSupported(CachewException):
    type_: type
    reason: str

    def __str__(self) -> str:
        return f"{self.type_} isn't supported by cachew: {self.reason}. See https://github.com/karlicoss/cachew#features for the list of supported types."

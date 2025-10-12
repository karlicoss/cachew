from dataclasses import dataclass

# TODO better name to represent what it means?
type SourceHash = str


class CachewException(RuntimeError):
    pass


@dataclass
class TypeNotSupported(CachewException):
    type_: type
    reason: str

    def __str__(self) -> str:
        return f"{self.type_} isn't supported by cachew: {self.reason}. See https://github.com/karlicoss/cachew#features for the list of supported types."

import sys

if sys.version_info[:2] >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


__all__ = ["deprecated"]

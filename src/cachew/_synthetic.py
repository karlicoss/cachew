import json
from typing import Any

from .common import DEPENDENCIES, SourceHash

CACHEW_CACHED = 'cachew_cached'  # TODO add to docs
SYNTHETIC_KEY = 'synthetic_key'
SYNTHETIC_KEY_VALUE = 'synthetic_key_value'


def missing_synthetic_key_values(cached: list[str], wanted: list[str]) -> list[str] | None:
    # FIXME assert both cached and wanted are sorted? since we rely on it
    # if not, then the user could use some custom key for caching (e.g. normalise filenames etc)
    # although in this case passing it into the function wouldn't make sense?

    if len(cached) == 0:
        # no point trying to reuse anything, cache should be empty?
        return None
    if len(wanted) == 0:
        # similar, no way to reuse cache
        return None
    if cached[0] != wanted[0]:
        # there is no common prefix, so no way to reuse cache really
        return None
    last_cached = cached[-1]
    # ok, now actually figure out which items are missing
    for i, k in enumerate(wanted):
        if k > last_cached:
            # ok, rest of items are missing
            return wanted[i:]
    # otherwise too many things are cached, and we seem to wante less
    return None


def missing_synthetic_key_values_for_hashes(
    *,
    old_hash: SourceHash | None,
    new_hash_d: dict[str, Any],
) -> list[str] | None:
    old_hash_d: dict[str, Any] = {}
    if old_hash is not None:
        try:
            old_hash_d = json.loads(old_hash)
        except json.JSONDecodeError:
            # possible if we used old cachew version (<=0.8.1), hash wasn't json
            pass

    hash_diffs = {
        k: new_hash_d.get(k) == old_hash_d.get(k)
        for k in (*new_hash_d.keys(), *old_hash_d.keys())
        # the only 'allowed' differences for hash, otherwise need to recompute (e.g. if schema changed)
        if k not in {SYNTHETIC_KEY_VALUE, DEPENDENCIES}
    }
    cache_compatible = all(hash_diffs.values())
    if not cache_compatible:
        return None

    new_values: list[str] = new_hash_d[SYNTHETIC_KEY_VALUE]
    old_values: list[str] = old_hash_d[SYNTHETIC_KEY_VALUE]
    return missing_synthetic_key_values(cached=old_values, wanted=new_values)

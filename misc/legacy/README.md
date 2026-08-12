# Legacy structural SQLite serializer

This directory preserves Cachew's former `NTBinder` implementation for design reference.
It flattened annotated Python values into schema-derived SQLite columns.
Cachew replaced it with single-column serialized payloads in 2023 and retired it from the installed package, tests, and active benchmark matrix in 2026.

This is frozen historical source, not a supported backend or public API.
Its imports and embedded tests reflect its former `cachew.legacy` location and may no longer run.
Direct imports such as `from cachew.legacy import NTBinder` are no longer supported.

The old representation contributed two ideas that remain worth exploring independently: recursive structural schema fingerprints and schema-ordered positional records.
Those experiments belong in the current marshaller rather than reviving this implementation.

See the [historical comparison](../../doc/benchmarks/20230912-comparison-with-legacy.org) and the committed [baseline benchmark](../../.benchmarks/Linux-CPython-3.14-64bit/0007_baseline.json) for recorded results.

Cachew still recognizes the old SQLite table name when refreshing caches, so archiving this Python source does not remove on-disk cleanup compatibility.

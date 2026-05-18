# Runnnig benchmarks


`uv run --group testing -m pytest --pyargs cachew.tests.benchmarks --benchmark-only 

- add `--benchmark-compare` to compare with the saved baseline
- use `--benchmark-save baseline` to update the baseline
- use `--benchmark-time-unit=ms` to force time units (otherwise each group might have different ones)
- use `uv run --python=3.xx` to pin a specific python version
- add `-k ...` to select a specific benchmark
- gc is disabled in some benchmarks via `disable_gc=True` parameter
- TODO python version
- to reduce CPU noise can be helpful to prepend the command with `taskset`, e.g. `taskset -c 2 ...`
- you might get `PytestBenchmarkWarning: Benchmark machine_info is different` -- thsi usually happens due to small frequency scaling artifacts.
  
  See https://github.com/ionelmc/pytest-benchmark/issues/255.
  
  It's possible to workaround by adjusting CPU governor/power policy, but I haven't bothered so far as you can't do it per CPU.




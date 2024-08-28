import gc
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

PROFILES = Path(__file__).absolute().parent / 'profiles'


@contextmanager
def profile(name: str):
    # ugh. seems like pyinstrument slows down code quite a bit?
    if os.environ.get('PYINSTRUMENT') is None:
        yield
        return

    from pyinstrument import Profiler

    with Profiler() as profiler:
        yield

    PROFILES.mkdir(exist_ok=True)
    results_file = PROFILES / f"{name}.html"

    print("results for " + name, file=sys.stderr)
    profiler.print()

    results_file.write_text(profiler.output_html())


def timer(name: str):
    from codetiming import Timer

    return Timer(name=name, text=name + ': ' + '{:.2f}s')


@pytest.fixture
def gc_control(*, gc_on: bool):
    if gc_on:
        # no need to do anything, should be on by default
        yield
        return

    gc.disable()
    try:
        yield
    finally:
        gc.enable()


running_on_ci = 'CI' in os.environ

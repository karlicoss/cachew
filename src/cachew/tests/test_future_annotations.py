from __future__ import annotations

import os
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from subprocess import check_output
from typing import Any

import pytest
from more_itertools import one

from .. import cachew


# fmt: off
@dataclass
class NewStyleTypes1:
    a_str   : str
    a_dict  : dict[str, Any]
    a_list  : list[Any]
    a_tuple : tuple[float, str]
# fmt: on


def test_types1(tmp_path: Path) -> None:
    obj = NewStyleTypes1(
        a_str   = 'abac',
        a_dict  = {'a': True, 'x': {'whatever': 3.14}},
        a_list  = ['aba', 123, None],
        a_tuple = (1.23, '3.2.1'),
    )  # fmt: skip

    @cachew(tmp_path)
    def get() -> Iterator[NewStyleTypes1]:
        yield obj

    assert one(get()) == obj
    assert one(get()) == obj


# fmt: off
@dataclass
class NewStyleTypes2:
    an_opt  : str | None
    a_union : str | int
# fmt: on


def test_types2(tmp_path: Path) -> None:
    obj = NewStyleTypes2(
        an_opt  = 'hello',
        a_union = 999,
    )  # fmt: skip

    @cachew(tmp_path)
    def get() -> Iterator[NewStyleTypes2]:
        yield obj

    assert one(get()) == obj
    assert one(get()) == obj


@pytest.mark.parametrize('use_future_annotations', [False, True])
@pytest.mark.parametrize('local', [False, True])
@pytest.mark.parametrize('throw', [False, True])
def test_future_annotations(
    *,
    use_future_annotations: bool,
    local: bool,
    throw: bool,
    tmp_path: Path,
) -> None:
    """
    Checks handling of postponed evaluation of annotations (from __future__ import annotations)
    """

    # NOTE: to avoid weird interactions with existing interpreter in which pytest is running
    #  , we compose a program and running in python directly instead
    #  (also not sure if it's even possible to tweak postponed annotations without doing that)

    if use_future_annotations and local and throw:
        # when annotation is local (like inner class), then they end up as strings
        #  so we can't eval it as we don't have access to a class defined inside function
        #  keeping this test just to keep track of whether this is fixed at some point
        #  possibly relevant:
        #  - https://peps.python.org/pep-0563/#keeping-the-ability-to-use-function-local-state-when-defining-annotations
        pytest.skip("local aliases/classses don't work with from __future__ import annotations")

    _PREAMBLE = f'''
from pathlib import Path
import tempfile

from cachew import cachew, settings
settings.THROW_ON_ERROR = {throw}

temp_dir = tempfile.TemporaryDirectory()
td = Path(temp_dir.name)

'''

    _TEST = '''
T = int

@cachew(td)
def fun() -> list[T]:
    print("called")
    return [1, 2]

assert list(fun()) == [1, 2]
assert list(fun()) == [1, 2]
'''

    if use_future_annotations:
        code = '''
from __future__ import annotations
'''
    else:
        code = ''

    code += _PREAMBLE

    if local:
        code += f'''
def test() -> None:
{textwrap.indent(_TEST, prefix=" ")}

test()
'''
    else:
        code += _TEST

    run_py = tmp_path / 'run.py'
    run_py.write_text(code)

    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()

    res = check_output(
        [sys.executable, run_py],
        env={'TMPDIR': str(cache_dir), **os.environ},
        text=True,
    )
    called = int(res.count('called'))
    if use_future_annotations and local and not throw:
        # cachew fails to set up, so no caching but at least it works otherwise
        assert called == 2
    else:
        assert called == 1

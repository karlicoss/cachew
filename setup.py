# -*- coding: utf-8 -*-
"""
    Setup file for cachew.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import require, VersionConflict
from setuptools import setup

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    setup(
        use_pyscaffold=True,
        package_data={
            'cachew': [
                'py.typed',
            ],
        },
        extras_require={
            # ugh, unclear how to specify python version dependnt code in setup.cfg
            ':python_version<"3.7"': [
                'dataclasses', # TODO could make optional?
            ],
            'testing': [
                'pytest',
                'pytz',

                'pylint',
                'mypy',
                'bandit',

                'patchy',
            ]
        },
    )

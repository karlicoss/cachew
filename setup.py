# see https://github.com/karlicoss/pymplate for up-to-date reference


url = 'https://github.com/karlicoss/cachew'
author = 'Dima Gerasimov'
author_email = 'karlicoss@gmail.com'
description = 'Easy sqlite-backed persistent cache for dataclasses'

install_requires = [
    'appdirs'        ,  # default cache dir
    'sqlalchemy>=1.0',  # cache DB interaction
]


from setuptools import setup, find_namespace_packages # type: ignore


def main() -> None:
    pkgs = find_namespace_packages('src', exclude=['*.tests'])
    pkg = min(pkgs)
    setup(
        name=pkg,
        use_scm_version={
            'version_scheme': 'python-simplified-semver',
            'local_scheme': 'dirty-tag',
        },
        setup_requires=['setuptools_scm'],

        url=url,
        author=author,
        author_email=author_email,
        description=description,

        packages=[pkg],
        package_dir={'': 'src'},
        package_data={pkg: ['py.typed']},

        python_requires='>=3.8',

        install_requires=install_requires,
        extras_require={
            'testing': [
                'pytest', 'pytz', 'patchy',
                'pylint',
                'mypy', 'lxml',
                'bandit',

                'more-itertools',

                'enlighten',  # used in logging helper, but not really required

                'orjson',  # for now test only but may actually use soon
                'cattrs',  # benchmarking alternative marshalling implementation

                'pyinstrument',  # for profiling from within tests
                'codetiming',    # Timer context manager
            ],
            'optional': [
                'colorlog',
            ],
        },

        zip_safe=False,
    )


if __name__ == '__main__':
    main()

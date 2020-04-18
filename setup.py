# see https://github.com/karlicoss/pymplate for up-to-date reference


url = 'https://github.com/karlicoss/cachew'
author = 'Dima Gerasimov'
author_email = 'karlicoss@gmail.com'
description = 'Easy sqlite-backed persistent cache for dataclasses'

install_requires = [
    'sqlalchemy',
]


from setuptools import setup, find_packages # type: ignore


def main():
    pkgs = find_packages('src', exclude=['*.tests'])
    [pkg] = pkgs
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

        install_requires=install_requires,
        extras_require={
            # can't specify this in setup.cfg!
            ':python_version<"3.7"': ['dataclasses'],
            'testing': [
                'pytest', 'pytz', 'patchy',
                'pylint',
                'mypy',
                'bandit',
            ],
        },

        zip_safe=False,
    )


if __name__ == '__main__':
    main()

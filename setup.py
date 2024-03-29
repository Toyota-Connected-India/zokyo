# -*- coding: utf-8 -*-

# ============================================================================
# Copyright:    Toyota Connected, 2020.  All rights reserved.
# Authors:      Zokyo Developers
# ============================================================================

from setuptools import setup
from setuptools import find_packages

import sys

# This allows us to determine the package version without needing to install
# all zokyo dependencies
import builtins
builtins.__ZOKYO_SETUP__ = True
import zokyo

REQUIREMENTS = [
    'numpy>=1.17.3',
    'pyyaml>=5.4',
    'pandas>=1.1.5',
    'scikit-learn>=0.23.2',
    'opencv-python>=4.4.0',
    'Pillow>=5.2.0',
    'scipy>=1.5.0',
    'tqdm',
    'tensorflow>=2.5.0rc1',
    'Sphinx',
    'sphinx_rtd_theme'
]

SETUPTOOLS_COMMANDS = {
    'install', 'bdist_wheel', 'sdist'
}

# are we building from install or develop? Since "install" is not in the
# SETUPTOOLS_COMMANDS, we have to check that here...
we_be_buildin = 'install' in sys.argv

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    from setuptools.dist import Distribution

    class BinaryDistribution(Distribution):
        """Command class to indicate binary distribution.

        The goal is to avoid having to later build the C or Fortran code
        on the system itself, but to build the binary dist wheels on the
        CI platforms. This class helps us achieve just that.

        References
        ----------
        .. [1] How to avoid building a C library with my Python package:
               http://bit.ly/2vQkW47
        .. [2] https://github.com/spotify/dh-virtualenv/issues/113
        """

        def is_pure(self):
            """Return False (not pure).

            Since we are distributing binary (.so, .dll, .dylib) files for
            different platforms we need to make sure the wheel does not build
            without them! See 'Building Wheels':
            http://lucumr.pocoo.org/2014/1/27/python-on-wheels/
            """
            return False

        def has_ext_modules(self):
            """Return True (there are external modules).

            The package has external modules. Therefore, unsurprisingly,
            this returns True to indicate that there are, in fact, external
            modules.
            """
            return True

    # only import numpy (later) if we're developing
    if any(cmd in sys.argv for cmd in {'develop', 'bdist_wheel'}):
        we_be_buildin = True

    print('Adding extra setuptools args')
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        package_data=dict(DISTNAME=['*',
                                    'zokyo/datasets/uatg/*']),
        distclass=BinaryDistribution
    )
else:
    extra_setuptools_args = dict()


def do_setup():
    setup(
        name="zokyo",
        version=zokyo.__version__,
        description="Data augmentation library",
        author="Zokyo contributors",
        author_email=[
            "srinivas.v@toyotaconnected.co.in",
            "srivathsan.govindarajan@toyotaconnected.co.in",
            "harshavardhan.thirupathi@toyotaconnected.co.in"
            "ashok.ramadass@toyotaconnected.com"
        ],
        url="https://github.com/toyotaconnected-India/zokyo",
        license="Unlicense",
        packages=find_packages(),
        include_package_data=True,
        install_requires=REQUIREMENTS,
        python_requires=">=3.5, <4",
        **extra_setuptools_args
    )


if __name__ == '__main__':
    do_setup()

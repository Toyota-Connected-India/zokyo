# -*- coding: utf-8 -*-

# ============================================================================
# Copyright:    Toyota Connected, 2020.  All rights reserved.
# Authors:      Sphinx Developers
# Email:        <ashok.ramadass@toyotaconnected.com>
# Date:         10/28/20  23:34:54
# ============================================================================

import os
import six
import subprocess

import numpy as np

__all__ = [
    'get_git_revision_short_hash',
    'is_iterable',
    'str_to_random_state',
    'get_or_set_env',
]


def get_git_revision_short_hash():
    """Get the short revision hash of a git commit

    Examples
    --------
    >>> get_git_revision_short_hash()
    '2e708f7'
    """
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()


def is_iterable(x):
    """Determine whether ``x`` is a non-string iterable"""
    if isinstance(x, six.string_types):
        return False
    return hasattr(x, "__iter__")


def str_to_random_state(x, first_n=9):
    """Seed a Numpy random state from a seed

    Hashes the string, takes the first N characters of the absolute value of
    the integer hash result (since numpy random state seeds must be < 2**32-1)
    and seeds a random state. This allows us to create reproducible random
    states given a string as input (particularly for data creation).

    Parameters
    ----------
    x : str
        The string to hash.
    """
    return np.random.RandomState(int(str(abs(hash(x)))[:first_n]))


def get_or_set_env(env_var, default_value):
    """ Return either a environment variable or default value for the variable
    if the value is either None or an empty string.
    """
    env_val = os.environ.get(env_var)
    if env_val in ('', None):
        return default_value
    else:
        return env_val

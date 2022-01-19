# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,srivathsan.govindarajan@toyotaconnected.co.in,
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com ]

import os
import six
import subprocess

import numpy as np

__all__ = [
    'get_git_revision_short_hash',
    'is_iterable',
    'str_to_random_state',
    'get_or_set_env',
    'from_float',
    'to_float'
]

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def get_git_revision_short_hash():
    """Get the short revision hash of a git commit
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


def to_float(img, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[img.dtype]
        except KeyError:
            raise RuntimeError(
                '''Can't infer the maximum value for dtype {}. You need to
                specify the maximum value manually by
                passing the max_value argument'''.format(img.dtype)
            )
    return img.astype("float32") / max_value


def from_float(img, dtype, max_value=None):
    if max_value is None:
        try:
            max_value = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                '''Can't infer the maximum value for dtype {}. You need to
                specify the maximum value manually by
                passing the max_value argument'''.format(dtype)
            )
    return (img * max_value).astype(dtype)

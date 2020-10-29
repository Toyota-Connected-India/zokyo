# -*- coding: utf-8 -*-

"""Utilities for testing used across Sphinx"""

__all__ = ['pytest_err_msg']


def pytest_err_msg(err):
    """Get the error message from an exception that pytest catches

    Compatibility function for newer versions of pytest, where ``str(err)``
    no longer returns the expected ``__repr__`` of an exception.

    Parameters
    ----------
    err : BaseException
        The caught exception
    """
    try:
        return str(err.value)
    except AttributeError:
        return str(err)

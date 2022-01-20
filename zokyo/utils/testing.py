# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,srivathsan.govindarajan@toyotaconnected.co.in,
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com ]

"""Utilities for testing used across Zokyo"""

__all__ = ['pytest_err_msg']


def pytest_err_msg(err):
    """Get the error message from an exception that pytest catches

    Compatibility function for newer versions of pytest, where ``str(err)``
    no longer returns the expected ``__repr__`` of an exception.
    """
    try:
        return str(err.value)
    except AttributeError:
        return str(err)

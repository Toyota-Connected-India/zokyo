# -*- coding: utf-8 -*-
# Contributors : [srinivas.v@toyotaconnected.co.in,srivathsan.govindarajan@toyotaconnected.co.in,
# harshavardhan.thirupathi@toyotaconnected.co.in,
# ashok.ramadass@toyotaconnected.com ]

"""Utilities for Pandas frames"""

import json

from .misc import is_iterable

__all__ = ['json_field_parser']


def json_field_parser(data):
    """
    Parses a string field to JSON when reading the CSV from disk
    """
    # If it's already a dict, pass through.
    if isinstance(data, dict):
        return data

    # list or pd.Series (non-dict iterable)
    if is_iterable(data):
        return [json_field_parser(d) for d in data]

    # Otherwise just try it and let it fail out on its own...
    if data == "*":
        return None
    return json.loads(data)

# -*- coding: utf-8 -*-

# ============================================================================
# Copyright:    Toyota Connected, 2020.  All rights reserved.
# Authors:      Zokyo Developers
# Email:        <ashok.ramadass@toyotaconnected.com>
# Date:         10/28/20  23:28:54
# ============================================================================

"""Utilities for Pandas frames"""

import json

from .misc import is_iterable

__all__ = ['json_field_parser']


def json_field_parser(data):
    """Parses a string field to JSON when reading the CSV from disk

    Parameters
    ----------
    data : str or iterable
        The string or iterable that should be parsed to JSON

    Examples
    --------
    The intended form is for this to be used as a converter during the
    dataframe parse stage:

    >>> df = pd.read_csv('x.csv', converters={'entities': json_field_parser})

    Alternatively, an iterable of parseable JSONs can be passed:

    >>> json_field_parser(['{"key": "value"}'])
    [{'key': 'value'}]

    Passing an already-parsed JSON will also work:

    >>> json_field_parser({'key': 'value'})
    {'key': 'value'}
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

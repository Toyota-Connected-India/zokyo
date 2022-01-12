# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from ...utils.dataframes import json_field_parser

text_values = ['{"image": "1.png"}', '{"image": "2.png"}']
dict_values = [{'image': '1.png'}, {'image': '2.png'}]
with_star = ['{"image": "1.png"}', '*']
text_series = pd.Series(text_values)
dict_series = pd.Series(dict_values)


@pytest.mark.parametrize(
    'ipt,exp', [
        pytest.param(['{"key": "value"}'], [{'key': 'value'}]),
        pytest.param({'key': 'value'}, {'key': 'value'}),
        pytest.param(text_series, dict_values),
        pytest.param(with_star, [{'image': '1.png'}, None]),
    ]
)
def test_json_parsing(ipt, exp):
    converted = json_field_parser(ipt)
    assert converted == exp

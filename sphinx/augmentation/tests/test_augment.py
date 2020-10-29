# -*- coding: utf-8 -*-

from sphinx.augmentation import do_augmentation
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/../uatg"

def test_do_augmentation():
    # Pre-cache
    count = do_augmentation(dir_path=dir_path, no_of_sample=10)

    assert count == 10

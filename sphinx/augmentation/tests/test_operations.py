import pytest
import unittest
from ...augmentation import ColorEqualize, DarkenScene
from ...augmentation import Builder
import shutil
import os

class OperationsTest(unittest.TestCase):
    def test_color_equalize(self):
        builder = Builder("tests/color_equalize_test_config.json")
        builder.build_and_run()
        assert len(os.listdir('tests/images/output/color_equalize')) == 5
        shutil.rmtree('tests/images/output')
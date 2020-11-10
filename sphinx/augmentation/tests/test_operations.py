import pytest
import unittest
from ...augmentation import EqualizeScene, DarkenScene, BrightenScene
from ...augmentation import Builder
import shutil
import os

class OperationsTest(unittest.TestCase):
    def test_equalize_scene(self):
        builder = Builder("tests/color_equalize_test_config.json")
        builder.build_and_run()
        assert len(os.listdir('tests/images/output/color_equalize')) == 5
        shutil.rmtree('tests/images/output')

    def test_darken_scene(self):
        builder = Builder("tests/darkness_coeff_test_config.json")
        builder.build_and_run()
        assert len(os.listdir('tests/images/output/darken_scene')) == 5
        shutil.rmtree('tests/images/output')

    def test_brighten_scene(self):
        builder = Builder("tests/brighten_scene_test_config.json")
        builder.build_and_run()
        assert len(os.listdir('tests/images/output/brighten_scene')) == 5
        shutil.rmtree('tests/images/output')

    def test_random_brightness(self):
        builder = Builder("tests/random_brightness_test_config.json")
        builder.build_and_run()
        assert len(os.listdir('tests/images/output/random_brightness')) == 5
        shutil.rmtree('tests/images/output')

    def test_rain_scene(self):
        builder = Builder("tests/rain_scene_test_config.json")
        builder.build_and_run()
        assert len(os.listdir('tests/images/output/rain_scene')) == 5
        shutil.rmtree('tests/images/output')
import pytest
import unittest
from ...augmentation import EqualizeScene, DarkenScene, BrightenScene
from ...augmentation import Builder
import shutil
import os


class OperationsTest(unittest.TestCase):
    def test_equalize_scene(self):
        builder = Builder("tests/color_equalize_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def test_darken_scene(self):
        builder = Builder("tests/darkness_coeff_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def test_brighten_scene(self):
        builder = Builder("tests/brighten_scene_test_config.json")
        builder.process_and_save(internal_batch_size=2)
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def test_random_brightness(self):
        builder = Builder("tests/random_brightness_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def test_rain_scene(self):
        builder = Builder("tests/rain_scene_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def radial_lens_distortion(self):
        builder = Builder("tests/radial_lens_distortion_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def tangential_lens_distortion(self):
        builder = Builder("tests/tangential_lens_distortion_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

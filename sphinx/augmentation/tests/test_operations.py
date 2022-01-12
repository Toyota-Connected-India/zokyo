import unittest  # noqa: F401
from sphinx.augmentation import Builder
import shutil
import os
import pytest
from pathlib import Path

test_configs = [
    "tests/color_equalize_test_config.json",
    "tests/darkness_coeff_test_config.json",
    "tests/brighten_scene_test_config.json",
    "tests/random_brightness_test_config.json",
    "tests/rain_scene_test_config.json",
    "tests/radial_lens_distortion_test_config.json",
    "tests/tangential_lens_distortion_test_config.json",
    "tests/snow_scene_test_config.json",
    "tests/sun_flare_test_config.json"
]


@pytest.mark.parametrize(
    "test_config",
    [pytest.param(conf) for conf in test_configs],
    ids=[Path(conf).stem for conf in test_configs]
)
class TestOperations:

    @pytest.fixture(autouse=True)
    def builder_teardown(self, request):
        # request.cls.builder = Builder("tests/builder_test_config.json")
        def teardown():
            shutil.rmtree('tests/output/images')
            shutil.rmtree('tests/output/masks')
            shutil.rmtree('tests/output/annotations')

        request.addfinalizer(teardown)

    def test_operation(self, test_config):
        builder = Builder(test_config)
        builder.process_and_save()
        assert len(os.listdir('tests/output/images')) == builder.sample
        assert len(os.listdir('tests/output/masks')) == builder.sample

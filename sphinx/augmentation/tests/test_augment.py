# -*- coding: utf-8 -*-
import os
import pytest
import unittest
from sphinx.augmentation import Builder
import shutil


@pytest.fixture(scope="class")
def builder_class(request):
    request.cls.builder = Builder("tests/builder_test_config.json")


@pytest.mark.usefixtures("builder_class")
class BuilderTest(unittest.TestCase):
    def test_build_run(self):
        self.builder.process_and_save()
        assert len(os.listdir('tests/output/images/')) == 6
        assert len(os.listdir('tests/output/masks')) == 6
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')

    def test_save_annotation_run(self):
        builder = Builder("tests/save_annotation_test_config.json")
        builder.process_and_save()
        assert len(os.listdir('tests/output/images/')) == 7
        assert len(os.listdir('tests/output/masks')) == 7
        assert len(os.listdir('tests/output/annotation_mask/')) == 7
        shutil.rmtree('tests/output/images')
        shutil.rmtree('tests/output/masks')
        shutil.rmtree('tests/output/annotation_mask')

    def test_sequential_sampling_run(self):
        pipeline = Builder(
            config_json="tests/sequential_sampling_test_config.json")
        pipeline.calculate_and_set_generator_params(batch_size=2)
        gen = pipeline.process_and_generate()

        imgs = sorted(os.listdir("tests/images/"))
        i = 0

        while True:
            try:
                res = next(gen)
                for item in res:
                    assert item.name == imgs[i % len(imgs)][:-4]
                    i += 1

            except StopIteration:
                break

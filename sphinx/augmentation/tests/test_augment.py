# -*- coding: utf-8 -*-
import os
import pytest
import unittest
from ...augmentation import Builder
from ...utils.CustomExceptions import CrucialValueNotFoundError
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

# -*- coding: utf-8 -*-
import os
import pytest
import unittest
from ...augmentation import Builder
from ...utils.CustomExceptions import CrucialValueNotFoundError


@pytest.fixture(scope="class")
def builder_class(request):
    request.cls.builder = Builder("tests/builder_test_config.json")


@pytest.mark.usefixtures("builder_class")
class BuilderTest(unittest.TestCase):
    def test_build_run(self):
        self.builder.build_and_run()
        assert len(os.listdir('tests/images/output')) == 5
    

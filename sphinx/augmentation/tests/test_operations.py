import pytest
import unittest
from ...augmentation import ColorEqualize, DarkenScene
from Augmentor import pipeline

@pytest.fixture(scope="class")
def builder_class(request):

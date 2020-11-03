import pytest
import unittest
from ...augmentation import ColorEqualize, DarkenScene
import Augmentor


class OperationsTest(unittest.TestCase):
    def test_color_equalize(self):
        cq = ColorEqualize(probability=0.5)
        pipeline = Augmentor.Pipeline("tests/images")
        pipeline.add_operation(cq)

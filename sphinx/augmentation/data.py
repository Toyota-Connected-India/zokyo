from PIL import Image
from xml.etree import ElementTree
import copy


class SphinxData(object):
    def __init__(self, ):
        self.image: Image.Image = None
        self.mask: Image.Image = None
        self.annotation_mask: Image.Image = None
        self.annotation: ElementTree = None

    def copy(self):
        return copy.deepcopy(self)

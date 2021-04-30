from PIL import Image
from xml.etree import ElementTree


class SphinxData(object):
    image: Image.Image = None
    mask: Image.Image = None
    annotation_mask: Image.Image = None
    annotation: ElementTree = None

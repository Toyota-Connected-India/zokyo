import numpy as np
from xml.etree import ElementTree


class SphinxData(object):
    image: np.ndarray = None
    mask: np.ndarray = None
    annotation_mask: np.ndarray = None
    annotation: ElementTree = None

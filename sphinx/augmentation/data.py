import numpy as np
from xml.etree import ElementTree

class SphinxData(object):    
    image: np.ndarray
    mask: np.ndarray
    annotation_mask: np.ndarray
    annotation: ElementTree
        
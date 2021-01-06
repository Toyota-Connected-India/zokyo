# Contributors : [srinivas.v@toyotaconnected.co.in, ]
import numpy as np
from PIL import Image
import os


def apply_augmentation(image, mask, label, function):
    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)
    augmented_segment = function(image)
    image[mask == label] = augmented_segment[mask == label]
    return image

def change_pascal_annotation(annotation, image_dir, filename):
    root = annotation.getroot()
    for child in root:
        if child.tag == "folder":
            child.text = image_dir
        if child.tag == "filename":
            child.text = filename
        if child.tag == "path":
            child.text = os.path.join(image_dir, filename)
    return annotation

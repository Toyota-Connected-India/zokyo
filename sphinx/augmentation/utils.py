# Contributors : [srinivas.v@toyotaconnected.co.in, ]
import numpy as np
import cv2


def annotations_to_mask():
    pass


def parse_annotations():
    pass


def apply_augmentation(image, mask, label, function):
    image = np.array(image, dtype=np.uint8)
    mask = np.array(mask, dtype=np.uint8)
    augmented_segment = function(image)
    image[mask == label] = augmented_segment[mask == label]
    return image
